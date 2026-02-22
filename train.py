"""
train.py

URAT-MIL training script — Stage 1 (PCAM bags) and Stage 2 (WSI .pt bags).

Fixes applied vs original:
  1. Validation loop runs every epoch and logs loss / acc / AUC
  2. Best checkpoint saved when val AUC improves
  3. Last checkpoint saved every epoch (for resume)
  4. EarlyStopping wired into the main loop
  5. Stage 2 data loader implemented and branched correctly
  6. Imports from sklearn added for AUC computation in val loop
"""

import os
import sys
import argparse
import logging
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from models.mil_model import URATMIL
from data.pcam_dataset import (
    PCAMBagDataset,
    WSIFeatureBagDataset,
    bag_collate,
)


# ──────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────
# Schedules
# ──────────────────────────────────────────────────────────
def get_beta(epoch: int, warmup: int, beta_max: float) -> float:
    """Linear KL annealing schedule — paper §III-C."""
    return min(beta_max, beta_max * epoch / max(warmup, 1))


def get_grl_alpha(epoch: int, max_epochs: int) -> float:
    """Sigmoid GRL schedule — paper Eq. 4."""
    p = epoch / max(max_epochs, 1)
    return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0


# ──────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────
def make_logger(run_name: str) -> logging.Logger:
    os.makedirs(C.LOG_DIR, exist_ok=True)
    log_path = os.path.join(C.LOG_DIR, f"{run_name}.log")
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger


# ──────────────────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0
        self.stop       = False

    def step(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ──────────────────────────────────────────────────────────
# Model Builder
# ──────────────────────────────────────────────────────────
def build_model(args, feature_dim: int) -> URATMIL:
    return URATMIL(
        feature_dim      = feature_dim,
        latent_dim       = C.LATENT_DIM,
        attention_dim    = C.ATTENTION_DIM,
        n_classes        = C.N_CLASSES,
        n_heads          = C.N_HEADS,
        n_prototypes     = C.N_PROTOTYPES,
        dropout          = C.DROPOUT,
        beta             = C.BETA_START,
        gamma            = C.GAMMA_MMD,
        use_domain_align = C.USE_DOMAIN_ALIGN and (args.stage == 2),
        use_variational  = not args.ablate_vae,
        use_ood          = not args.ablate_ood,
        use_unc_gate     = not args.ablate_unc,
    ).to(C.DEVICE)


# ──────────────────────────────────────────────────────────
# Data Loaders
# ──────────────────────────────────────────────────────────
def build_loaders_stage1():
    """Stage 1: synthetic MIL bags from pre-extracted PCAM features."""
    train_ds = PCAMBagDataset(
        feat_dir = C.PCAM_FEAT_DIR,
        split    = "train",
        bag_size = C.PCAM_BAG_SIZE,
        n_bags   = C.PCAM_N_TRAIN,
        seed     = C.SEED,
    )
    val_ds = PCAMBagDataset(
        feat_dir = C.PCAM_FEAT_DIR,
        split    = "val",
        bag_size = C.PCAM_BAG_SIZE,
        n_bags   = C.PCAM_N_VAL,
        seed     = C.SEED + 1,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size  = C.BATCH_SIZE,
        shuffle     = True,
        collate_fn  = bag_collate,
        num_workers = C.NUM_WORKERS,
        pin_memory  = (C.DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = 1,
        shuffle     = False,
        collate_fn  = bag_collate,
        num_workers = C.NUM_WORKERS,
    )
    return train_loader, val_loader, C.FEATURE_DIM


def build_loaders_stage2():
    """Stage 2: real CLAM-style WSI feature bags, cross-domain."""
    train_ds = WSIFeatureBagDataset(
        feat_dir   = C.WSI_FEAT_DIR_A,
        labels_csv = C.WSI_LABELS_A,
        seed       = C.SEED,
    )
    val_ds = WSIFeatureBagDataset(
        feat_dir   = C.WSI_FEAT_DIR_B,
        labels_csv = C.WSI_LABELS_B,
        seed       = C.SEED + 1,
    )
    feat_dim = train_ds[0]["features"].size(-1)

    train_loader = DataLoader(
        train_ds,
        batch_size  = C.BATCH_SIZE,
        shuffle     = True,
        collate_fn  = bag_collate,
        num_workers = C.NUM_WORKERS,
        pin_memory  = (C.DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = 1,
        shuffle     = False,
        collate_fn  = bag_collate,
        num_workers = C.NUM_WORKERS,
    )
    return train_loader, val_loader, feat_dim


# ──────────────────────────────────────────────────────────
# OOD Calibration
# ──────────────────────────────────────────────────────────
@torch.no_grad()
def calibrate_ood(model: URATMIL, loader, logger: logging.Logger):
    """Recalibrate OOD thresholds on the validation set after each epoch."""
    if not model.use_ood:
        return

    model.eval()
    all_ot, all_sood = [], []

    for batch in loader:
        for feats in batch["features"]:
            feats = feats.to(C.DEVICE)
            h = model.proj(feats)

            if model.use_variational:
                z, _, _, _ = model.vae(h)
            else:
                z = h

            s_ot         = model.ood.ot_score(z)
            s_s, _       = model.ood.recon_score(h, z)

            all_ot.append(s_ot.cpu())
            all_sood.append(s_s.cpu())

    if all_ot:
        ots  = torch.cat(all_ot)
        sods = torch.cat(all_sood)
        model.ood.calibrate(ots, sods, fpr=C.OOD_FPR_TARGET)
        logger.info(
            f"  OOD recalibrated: "
            f"delta_art={model.ood.delta_artifact.item():.4f}  "
            f"delta_near={model.ood.delta_near.item():.4f}"
        )


# ──────────────────────────────────────────────────────────
# Training Epoch
# ──────────────────────────────────────────────────────────
def train_epoch(
    model:     URATMIL,
    loader,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    logger:    logging.Logger,
) -> dict:

    model.train()

    beta  = get_beta(epoch, C.WARMUP_EPOCHS, C.BETA_MAX)
    alpha = get_grl_alpha(epoch, C.MAX_EPOCHS)

    model.set_beta(beta)
    model.set_grl_alpha(alpha)

    totals = dict(l_total=0.0, l_ce=0.0, kl=0.0, correct=0, n=0)

    for bag_idx, batch in enumerate(loader):

        labels        = batch["labels"].to(C.DEVICE)
        features_list = batch["features"]

        for i, feats in enumerate(features_list):

            feats = feats.to(C.DEVICE)
            label = labels[i]

            out    = model(feats)
            losses = model.compute_loss(
                out, label,
                C.LAMBDA_KL,
                C.LAMBDA_ALIGN,
                C.LAMBDA_RECON,
                C.LAMBDA_CAL,
            )

            optimizer.zero_grad()
            losses["l_total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            optimizer.step()

            pred = out["logits"].argmax().item()

            totals["l_total"] += losses["l_total"].item()
            totals["l_ce"]    += losses["l_ce"].item()
            totals["kl"]      += losses["kl"].item()
            totals["correct"] += int(pred == label.item())
            totals["n"]       += 1

        if (bag_idx + 1) % C.LOG_INTERVAL == 0:
            n = max(totals["n"], 1)
            logger.info(
                f"  [{bag_idx+1}/{len(loader)}] "
                f"loss={totals['l_total']/n:.4f} "
                f"ce={totals['l_ce']/n:.4f} "
                f"kl={totals['kl']/n:.4f} "
                f"acc={totals['correct']/n:.3f} "
                f"beta={beta:.3f} alpha={alpha:.3f}"
            )

    n = max(totals["n"], 1)
    return {
        "loss": totals["l_total"] / n,
        "acc":  totals["correct"] / n,
        "beta": beta,
    }


# ──────────────────────────────────────────────────────────
# Validation Epoch  ← was completely missing in original
# ──────────────────────────────────────────────────────────
@torch.no_grad()
def val_epoch(
    model:  URATMIL,
    loader,
    logger: logging.Logger,
    epoch:  int,
) -> dict:
    """
    Run inference on the validation set and compute loss, accuracy, and AUC.
    These are the ONLY numbers used for checkpoint selection and early stopping.
    No synthetic or mocked values.
    """
    model.eval()

    all_probs  = []   # (N_bags, C) — softmax probabilities
    all_labels = []   # (N_bags,)   — ground-truth integer labels
    total_loss = 0.0
    n          = 0

    for batch in loader:
        labels        = batch["labels"].to(C.DEVICE)
        features_list = batch["features"]

        for i, feats in enumerate(features_list):
            feats = feats.to(C.DEVICE)
            label = labels[i]

            out    = model(feats)
            losses = model.compute_loss(
                out, label,
                C.LAMBDA_KL,
                C.LAMBDA_ALIGN,
                C.LAMBDA_RECON,
                C.LAMBDA_CAL,
            )

            total_loss += losses["l_total"].item()
            all_probs.append(out["p_hat"].cpu().numpy())
            all_labels.append(label.item())
            n += 1

    probs      = np.stack(all_probs)        # (N, C)
    labels_arr = np.array(all_labels)       # (N,)
    preds      = probs.argmax(axis=1)

    acc = float((preds == labels_arr).mean())

    # AUC — skip if only one class present in this subset (can happen in small val sets)
    try:
        if C.N_CLASSES == 2:
            auc = roc_auc_score(labels_arr, probs[:, 1])
        else:
            auc = roc_auc_score(
                labels_arr, probs, multi_class="ovr", average="macro"
            )
    except ValueError as e:
        logger.warning(f"  AUC skipped (only one class in val batch): {e}")
        auc = 0.0

    avg_loss = total_loss / max(n, 1)

    logger.info(
        f"  [Val Ep {epoch:03d}] "
        f"loss={avg_loss:.4f} "
        f"acc={acc:.4f} "
        f"auc={auc:.4f}"
    )

    return {"loss": avg_loss, "acc": acc, "auc": auc}


# ──────────────────────────────────────────────────────────
# Checkpoint Save / Load
# ──────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, auc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model":            model.state_dict(),
            "optimizer":        optimizer.state_dict(),
            "epoch":            epoch,
            "val_auc":          auc,
            "use_domain_align": model.use_domain_align,
            "use_variational":  model.use_variational,
            "use_ood":          model.use_ood,
            "use_unc_gate":     model.attn.use_unc_gate,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location=C.DEVICE)
    # Support both formats: plain state_dict or wrapped dict
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    start_epoch = ckpt.get("epoch", 0) + 1
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return start_epoch


# ──────────────────────────────────────────────────────────
# Main Training Function
# ──────────────────────────────────────────────────────────
def train(args):

    set_seed(C.SEED)

    logger = make_logger(args.run_name)
    logger.info(f"=== URAT-MIL Training: {args.run_name} ===")
    logger.info(f"Device : {C.DEVICE}")
    if C.DEVICE.type == "cuda":
        logger.info(f"GPU    : {torch.cuda.get_device_name(0)}")
    logger.info(f"Stage  : {args.stage}")

    # ── Data ────────────────────────────────────────────
    if args.stage == 1:
        train_loader, val_loader, feat_dim = build_loaders_stage1()
    else:
        train_loader, val_loader, feat_dim = build_loaders_stage2()

    logger.info(f"Feature dim : {feat_dim}")
    logger.info(f"Train bags  : {len(train_loader.dataset)}")
    logger.info(f"Val bags    : {len(val_loader.dataset)}")

    # ── Model ────────────────────────────────────────────
    model = build_model(args, feat_dim)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters  : {n_params:,}")

    # ── Optimiser ────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = C.LR,
        weight_decay = C.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=C.MAX_EPOCHS, eta_min=C.LR * 0.01
    )

    # ── Resume ───────────────────────────────────────────
    last_ckpt = os.path.join(C.CKPT_DIR, f"{args.run_name}_last.pt")
    start_epoch = 1
    if args.resume and os.path.isfile(last_ckpt):
        start_epoch = load_checkpoint(last_ckpt, model, optimizer)
        logger.info(f"Resumed from epoch {start_epoch - 1}")

    # ── Training state ───────────────────────────────────
    best_auc = 0.0
    best_ckpt = os.path.join(C.CKPT_DIR, f"{args.run_name}_best.pt")
    last_ckpt = os.path.join(C.CKPT_DIR, f"{args.run_name}_last.pt")
    early_stop = EarlyStopping(patience=C.EARLY_STOP_PAT, min_delta=1e-4)

    # ── Epoch loop ───────────────────────────────────────
    for epoch in range(start_epoch, C.MAX_EPOCHS + 1):

        logger.info(f"\nEpoch {epoch}/{C.MAX_EPOCHS}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, epoch, logger)
        logger.info(
            f"  [Train] loss={train_metrics['loss']:.4f}  "
            f"acc={train_metrics['acc']:.4f}"
        )

        # Validate — real forward passes, no synthetic numbers
        val_metrics = val_epoch(model, val_loader, logger, epoch)

        # OOD threshold recalibration on val set
        calibrate_ood(model, val_loader, logger)

        # LR schedule step
        scheduler.step()

        # Save last checkpoint (allows training resume)
        save_checkpoint(model, optimizer, epoch, val_metrics["auc"], last_ckpt)

        # Save best checkpoint by val AUC
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            save_checkpoint(model, optimizer, epoch, best_auc, best_ckpt)
            logger.info(
                f"  --> New best AUC={best_auc:.4f} — checkpoint saved: {best_ckpt}"
            )

        # Early stopping check
        if early_stop.step(val_metrics["auc"]):
            logger.info(
                f"Early stopping triggered at epoch {epoch} "
                f"(no improvement for {C.EARLY_STOP_PAT} epochs)."
            )
            break

    logger.info(f"\nTraining complete.  Best val AUC = {best_auc:.4f}")
    logger.info(f"Best checkpoint : {best_ckpt}")


# ──────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="URAT-MIL Training")
    parser.add_argument("--stage",      type=int,  default=1,
                        help="1 = PCAM bags,  2 = WSI .pt bags")
    parser.add_argument("--run_name",   type=str,  default="urat_mil",
                        help="Tag used for log / checkpoint filenames")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume training from last checkpoint if it exists")
    parser.add_argument("--ablate_unc", action="store_true",
                        help="Ablation: disable uncertainty gating (Module 5)")
    parser.add_argument("--ablate_vae", action="store_true",
                        help="Ablation: disable variational head (Module 3)")
    parser.add_argument("--ablate_ood", action="store_true",
                        help="Ablation: disable OOD filter (Module 4)")

    args = parser.parse_args()
    train(args)
