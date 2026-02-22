"""
models/mil_model.py

URAT-MIL: Complete modular MIL model.

Paper contributions per module:
  Module 1: Foundation encoder (ResNet18) — feature_dim → latent_dim projection
  Module 2: Domain alignment (GRL + MMD)  — Eq. 4–6
  Module 3: Variational uncertainty head  — Eq. 7–10
  Module 4: Dual-level OOD detection      — Eq. 11–12
  Module 5: Uncertainty-aware MIL attention + calibration — Eq. 13–15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import config as C


# ═══════════════════════════════════════════════════════════
# MODULE 2 — Domain Alignment (GRL + MMD)
# ═══════════════════════════════════════════════════════════

class _GRLFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha, dtype=torch.float32))
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        alpha = ctx.saved_tensors[0].item()
        return -alpha * grad, None


class GRL(nn.Module):
    """Gradient Reversal Layer — paper Eq. 4."""
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return _GRLFunction.apply(x, self.alpha)


class DomainDiscriminator(nn.Module):
    """Adversarial domain classifier after GRL — paper Eq. 4."""
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.grl = GRL()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(True),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(self.grl(x))

    def set_alpha(self, alpha: float):
        self.grl.alpha = alpha


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, n_kernels: int = 5) -> torch.Tensor:
    """Maximum Mean Discrepancy with RBF kernel — paper Eq. 5."""
    n, m = x.size(0), y.size(0)
    xy   = torch.cat([x, y], dim=0)
    xx   = xy.unsqueeze(0) - xy.unsqueeze(1)          # (n+m, n+m, D)
    d    = (xx ** 2).sum(-1)                            # (n+m, n+m)

    bw = d.detach().median().clamp_min(1e-4)
    kernels = sum(
        torch.exp(-d / (2 * bw * (2 ** i)))
        for i in range(n_kernels)
    )
    return (kernels[:n, :n].mean()
            + kernels[n:, n:].mean()
            - 2 * kernels[:n, n:].mean())


# ═══════════════════════════════════════════════════════════
# MODULE 3 — Variational Uncertainty Head
# ═══════════════════════════════════════════════════════════

class VariationalHead(nn.Module):
    """
    Diagonal Gaussian posterior q_φ(z|h) = N(μ, σ²I) — paper Eq. 7–10.
    Outputs stochastic latent z, uncertainty estimates U_alea and U_epis.
    """
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.mu_head     = nn.Linear(in_dim, latent_dim)
        self.logvar_head = nn.Linear(in_dim, latent_dim)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.constant_(self.logvar_head.bias, -2.0)

    def encode(self, h):
        mu      = self.mu_head(h)
        log_var = self.logvar_head(h).clamp(-8.0, 4.0)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        if self.training:
            std = (0.5 * log_var).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, h):
        """Single forward pass — used during training."""
        mu, log_var = self.encode(h)
        z  = self.reparameterize(mu, log_var)
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
        return z, mu, log_var, kl

    @torch.no_grad()
    def mc_sample(self, h, T: int = 20):
        """
        T Monte Carlo passes — paper Eq. 9–10.
        Returns:
          z_mean  : (N, D)  mean latent
          u_alea  : (N,)    aleatoric  = E[σ²]
          u_epis  : (N,)    epistemic  = Var[μ_t]
        """
        mu, log_var = self.encode(h)
        std = (0.5 * log_var).exp()

        mus = []
        for _ in range(T):
            eps = torch.randn_like(std)
            mus.append(mu + std * eps)

        stacked = torch.stack(mus)           # (T, N, D)
        z_mean  = stacked.mean(0)
        u_epis  = stacked.var(0).mean(-1)    # (N,)
        u_alea  = std.pow(2).mean(-1)        # (N,)
        kl      = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()

        return z_mean, u_alea, u_epis, kl


# ═══════════════════════════════════════════════════════════
# MODULE 4 — Dual-Level OOD Detection
# ═══════════════════════════════════════════════════════════

class OODModule(nn.Module):
    """
    Tier-1: cosine prototype distance (≈ OT score) — paper Eq. 11.
    Tier-2: VAE reconstruction error + log-likelihood — paper Eq. 12.
    Produces gate mask m_i ∈ {0, 1}.
    """
    def __init__(
        self,
        latent_dim:     int,
        embed_dim:      int,
        n_prototypes:   int   = 64,
        alpha:          float = 0.5,
        delta_artifact: float = 0.6,
        delta_near:     float = 0.4,
    ):
        super().__init__()
        self.prototypes = nn.Parameter(
            torch.randn(n_prototypes, latent_dim) * 0.02, requires_grad=True
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(True),
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, embed_dim),
        )
        self.alpha = alpha
        self.register_buffer("delta_artifact", torch.tensor(delta_artifact))
        self.register_buffer("delta_near",     torch.tensor(delta_near))

    def ot_score(self, z: torch.Tensor) -> torch.Tensor:
        """Eq. 11: minimum cosine distance to prototype set."""
        zn = F.normalize(z, dim=-1)
        pn = F.normalize(self.prototypes, dim=-1)
        cos_sim = zn @ pn.t()                    # (N, K)
        return (1.0 - cos_sim).min(-1).values     # (N,)

    def recon_score(self, h: torch.Tensor, z: torch.Tensor):
        """Eq. 12: reconstruction-based OOD score."""
        h_hat     = self.decoder(z)
        recon_err = ((h - h_hat) ** 2).mean(-1)       # (N,)
        log_px    = -0.5 * recon_err
        return -log_px + self.alpha * recon_err, h_hat  # (N,)

    def forward(self, h: torch.Tensor, z: torch.Tensor):
        s_ot         = self.ot_score(z)
        s_ood, h_hat = self.recon_score(h, z)

        gate       = (s_ot <= self.delta_artifact) & (s_ood <= self.delta_near)
        recon_loss = F.mse_loss(h_hat, h.detach())

        return gate, s_ot, s_ood, recon_loss

    @torch.no_grad()
    def calibrate(self, ot_scores, sood_scores, fpr: float = 0.05):
        """Set thresholds so that fpr fraction of in-distribution samples are flagged."""
        q = 1.0 - fpr
        self.delta_artifact.fill_(torch.quantile(ot_scores,  q).item())
        self.delta_near.fill_(    torch.quantile(sood_scores, q).item())


# ═══════════════════════════════════════════════════════════
# MODULE 5 — Uncertainty-Aware MIL Attention + Calibration
# ═══════════════════════════════════════════════════════════

class MILAttention(nn.Module):
    """
    Uncertainty-gated multi-head attention pooling — paper Eq. 13.
    Learnable temperature for ECE calibration — paper Eq. 14.
    """
    def __init__(
        self,
        latent_dim:    int,
        attention_dim: int,
        n_classes:     int,
        n_heads:       int   = 4,
        dropout:       float = 0.25,
        use_unc_gate:  bool  = True,
    ):
        super().__init__()
        self.n_heads      = n_heads
        self.use_unc_gate = use_unc_gate

        self.W = nn.ModuleList([nn.Linear(latent_dim, attention_dim) for _ in range(n_heads)])
        self.V = nn.ModuleList([nn.Linear(attention_dim, 1)          for _ in range(n_heads)])
        self.head_merge = nn.Linear(latent_dim * n_heads, latent_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, n_classes),
        )

        self.log_temp = nn.Parameter(torch.zeros(1))   # log τ — Eq. 14 calibration

    @property
    def temperature(self):
        return self.log_temp.exp().clamp(0.01, 10.0)

    def forward(self, z, u_total, gate_mask):
        """
        z        : (N, latent_dim)
        u_total  : (N,)   per-patch total uncertainty
        gate_mask: (N,)   bool — True = keep patch
        Returns  : logits (C,), p_hat (C,), z_bag (D,), attn (N,)
        """
        mask = gate_mask.float()
        if mask.sum() < 1:
            mask = torch.ones_like(mask)

        u_safe = u_total.clamp(min=1e-6)

        heads, attns = [], []
        for k in range(self.n_heads):
            a = torch.tanh(self.W[k](z))      # (N, A)
            a = self.V[k](a).squeeze(-1)       # (N,)

            if self.use_unc_gate:
                a = a / u_safe                 # Eq. 13 uncertainty gating

            a = a * mask
            a = F.softmax(a, dim=0)            # (N,)
            attns.append(a)
            heads.append((a.unsqueeze(-1) * z).sum(0))   # (D,)

        z_bag  = self.head_merge(torch.cat(heads, -1))    # (D,)
        attn_w = torch.stack(attns, 0).mean(0)            # (N,)

        logits = self.classifier(z_bag.unsqueeze(0)).squeeze(0)   # (C,)
        p_hat  = F.softmax(logits / self.temperature, -1)          # (C,)

        return logits, p_hat, z_bag, attn_w


# ═══════════════════════════════════════════════════════════
# FULL URAT-MIL MODEL
# ═══════════════════════════════════════════════════════════

class URATMIL(nn.Module):
    """
    Unified Robustness-Aware Transformer MIL.

    Assembles Modules 1–5. Supports ablation via config flags:
      - use_domain_align : enable/disable Module 2
      - use_variational  : enable/disable Module 3 (falls back to deterministic)
      - use_ood          : enable/disable Module 4 gate
      - use_unc_gate     : enable/disable uncertainty gating in Module 5
    """
    def __init__(
        self,
        feature_dim:      int   = 512,
        latent_dim:       int   = 256,
        attention_dim:    int   = 128,
        n_classes:        int   = 2,
        n_heads:          int   = 4,
        n_prototypes:     int   = 64,
        dropout:          float = 0.25,
        beta:             float = 0.0,
        gamma:            float = 1.0,
        use_domain_align: bool  = True,
        use_variational:  bool  = True,
        use_ood:          bool  = True,
        use_unc_gate:     bool  = True,
    ):
        super().__init__()
        self.latent_dim       = latent_dim
        self.beta             = beta
        self.gamma            = gamma
        self.use_domain_align = use_domain_align
        self.use_variational  = use_variational
        self.use_ood          = use_ood

        # Module 1 — linear projection of pre-extracted features
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Module 2 — domain alignment
        if use_domain_align:
            self.domain_disc = DomainDiscriminator(latent_dim)
            self.adv_crit    = nn.BCEWithLogitsLoss()

        # Module 3 — variational head
        if use_variational:
            self.vae = VariationalHead(latent_dim, latent_dim)

        # Module 4 — OOD
        if use_ood:
            self.ood = OODModule(
                latent_dim   = latent_dim,
                embed_dim    = latent_dim,
                n_prototypes = n_prototypes,
            )

        # Module 5 — MIL attention
        self.attn = MILAttention(
            latent_dim    = latent_dim,
            attention_dim = attention_dim,
            n_classes     = n_classes,
            n_heads       = n_heads,
            dropout       = dropout,
            use_unc_gate  = use_unc_gate and use_variational,
        )

    # ── Helpers ────────────────────────────────────────────
    def set_beta(self, beta: float):
        self.beta = beta

    def set_grl_alpha(self, alpha: float):
        if self.use_domain_align:
            self.domain_disc.set_alpha(alpha)

    # ── Forward ────────────────────────────────────────────
    def forward(self, feats: torch.Tensor, src_feats=None, tgt_feats=None):
        """
        feats      : (N, feature_dim) — one MIL bag
        src_feats  : (M, feature_dim) — source domain batch (Stage 2 only)
        tgt_feats  : (K, feature_dim) — target domain batch (Stage 2 only)
        """
        device = feats.device

        # ── Module 1 ───────────────────────────────────────
        h = self.proj(feats)                      # (N, latent_dim)

        # ── Module 2 ───────────────────────────────────────
        l_adv = l_mmd = torch.zeros(1, device=device)
        if self.use_domain_align and src_feats is not None and tgt_feats is not None:
            src_h = self.proj(src_feats)
            tgt_h = self.proj(tgt_feats)
            all_h = torch.cat([src_h, tgt_h])
            logits_d = self.domain_disc(all_h).squeeze(-1)
            dom_labels = torch.cat([
                torch.ones(src_h.size(0),  device=device),
                torch.zeros(tgt_h.size(0), device=device),
            ])
            l_adv = self.adv_crit(logits_d, dom_labels).unsqueeze(0)
            l_mmd = mmd_rbf(src_h, tgt_h).unsqueeze(0)

        l_align = l_adv + self.gamma * l_mmd

        # ── Module 3 ───────────────────────────────────────
        if self.use_variational:
            if self.training:
                z, mu, log_var, kl = self.vae(h)
                u_alea = (0.5 * log_var).exp().pow(2).mean(-1).detach()
                u_epis = torch.zeros_like(u_alea)
            else:
                z, u_alea, u_epis, kl = self.vae.mc_sample(h)
            kl = self.beta * kl
        else:
            z      = h
            u_alea = torch.zeros(h.size(0), device=device)
            u_epis = torch.zeros(h.size(0), device=device)
            kl     = torch.zeros(1, device=device)

        u_total = (u_alea + u_epis).clamp(min=1e-6)

        # ── Module 4 ───────────────────────────────────────
        l_recon = torch.zeros(1, device=device)
        if self.use_ood:
            gate_mask, s_ot, s_ood, l_recon = self.ood(h, z)
        else:
            gate_mask = torch.ones(z.size(0), dtype=torch.bool, device=device)
            s_ot = s_ood = torch.zeros(z.size(0), device=device)

        # ── Module 5 ───────────────────────────────────────
        logits, p_hat, z_bag, attn_w = self.attn(z, u_total, gate_mask)

        return {
            "logits":    logits,
            "p_hat":     p_hat,
            "z_bag":     z_bag,
            "attn_w":    attn_w,
            "gate_mask": gate_mask,
            "u_alea":    u_alea,
            "u_epis":    u_epis,
            "u_total":   u_total,
            "s_ot":      s_ot,
            "s_ood":     s_ood,
            "kl":        kl,
            "l_align":   l_align,
            "l_recon":   l_recon,
        }

    def compute_loss(self, out, label, lambda_kl, lambda_align, lambda_recon, lambda_cal):
        """
        Unified loss — paper Eq. 15:
        L_total = L_CE + λ1·KL + λ2·L_align + λ3·L_cal + λ4·L_recon

        L_CE uses label smoothing (ε = C.LABEL_SMOOTHING) to prevent
        overconfidence and improve calibration (paper Module 5).
        """
        label_t = label.view(1) if label.dim() == 0 else label[:1]
        logit_t = out["logits"].unsqueeze(0)
        p_t     = out["p_hat"].unsqueeze(0)

        # Label-smoothed cross-entropy — improves calibration
        l_ce = F.cross_entropy(
            logit_t, label_t,
            label_smoothing=C.LABEL_SMOOTHING,
        )

        # Calibration loss: push predicted confidence toward observed accuracy
        correct = (logit_t.argmax(-1) == label_t).float()
        l_cal   = F.mse_loss(p_t.max(-1).values, correct)

        l_total = (l_ce
                   + lambda_kl    * out["kl"].squeeze()
                   + lambda_align * out["l_align"].squeeze()
                   + lambda_cal   * l_cal
                   + lambda_recon * out["l_recon"].squeeze())

        return {
            "l_total": l_total,
            "l_ce":    l_ce.detach(),
            "kl":      out["kl"].detach().squeeze(),
            "l_align": out["l_align"].detach().squeeze(),
            "l_cal":   l_cal.detach(),
            "l_recon": out["l_recon"].detach().squeeze(),
        }
