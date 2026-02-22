"""
URAT-MIL  ·  Evaluation Script  (fixed load_model)
Infers ablation flags from checkpoint keys — works with any saved checkpoint.
"""

import os, sys, argparse, logging
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    ConfusionMatrixDisplay, classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from models.mil_model import URATMIL
from data.pcam_dataset import PCAMBagDataset, bag_collate


def make_logger(name):
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        log.addHandler(h)
    return log


def compute_ece(probs, labels, n_bins=10):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies  = (predictions == labels).astype(float)
    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs    = np.zeros(n_bins)
    bin_confs   = np.zeros(n_bins)
    bin_counts  = np.zeros(n_bins)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask   = (confidences > lo) & (confidences <= hi)
        nb     = mask.sum()
        if nb > 0:
            bin_accs[b]   = accuracies[mask].mean()
            bin_confs[b]  = confidences[mask].mean()
            bin_counts[b] = nb
    ece = np.sum(bin_counts / len(probs) * np.abs(bin_accs - bin_confs))
    return ece, bin_accs, bin_confs, bin_counts


def plot_roc(fpr, tpr, auc_val, run_name, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1A73E8", lw=2.2, label=f"URAT-MIL  AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random  AUC = 0.50")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#1A73E8")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title(f"ROC Curve — {run_name}", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right"); ax.set_xlim([0,1]); ax.set_ylim([0,1.02]); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run_name}_roc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def plot_confusion(cm, run_name, out_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {run_name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run_name}_cm.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def plot_reliability(bin_accs, bin_confs, bin_counts, ece, run_name, out_dir):
    n_bins  = len(bin_accs)
    centers = np.linspace(0.5/n_bins, 1-0.5/n_bins, n_bins)
    width   = 1.0 / n_bins
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.bar(centers, bin_accs, width=width*0.85, alpha=0.8, color="#1A73E8",
           label=f"URAT-MIL (ECE={ece:.4f})")
    ax.plot([0,1],[0,1],"k--", lw=1.4, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Confidence"); ax.set_ylabel("Fraction Correct")
    ax.set_title("Reliability Diagram", fontweight="bold")
    ax.legend(); ax.set_xlim([0,1]); ax.set_ylim([0,1.05]); ax.grid(alpha=0.3)
    ax2 = axes[1]
    total = bin_counts.sum()
    ax2.bar(centers, bin_counts / max(total, 1), width=width*0.85, alpha=0.8, color="#34A853")
    ax2.set_xlabel("Confidence bin"); ax2.set_ylabel("Fraction of samples")
    ax2.set_title("Confidence Distribution", fontweight="bold"); ax2.set_xlim([0,1]); ax2.grid(alpha=0.3)
    fig.suptitle(run_name, fontsize=11, style="italic"); fig.tight_layout()
    path = os.path.join(out_dir, f"{run_name}_reliability.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def plot_uncertainty(u_alea, u_epis, labels, run_name, out_dir):
    u_alea = np.array(u_alea); u_epis = np.array(u_epis); labels = np.array(labels)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["#1A73E8", "#E8710A"]
    for ax, (u, title) in zip(axes, [(u_alea, "Aleatoric U_alea"), (u_epis, "Epistemic U_epis")]):
        for c, col in zip([0,1], colors):
            mask = labels == c
            ax.hist(u[mask], bins=30, alpha=0.65, color=col,
                    label=f"Class {c} (n={mask.sum()})", density=True)
        ax.set_xlabel("Uncertainty"); ax.set_ylabel("Density")
        ax.set_title(title, fontweight="bold"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run_name}_uncertainty.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def plot_ood(ood_scores, labels, run_name, out_dir):
    ood_scores = np.array(ood_scores); labels = np.array(labels)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#1A73E8", "#E8710A"]
    for c, col in zip([0,1], colors):
        mask = labels == c
        ax.hist(ood_scores[mask], bins=40, alpha=0.7, color=col,
                density=True, label=f"Class {c}")
    ax.set_xlabel("OOD Score"); ax.set_ylabel("Density")
    ax.set_title(f"OOD Score Distribution — {run_name}", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    path = os.path.join(out_dir, f"{run_name}_ood_scores.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def load_model(ckpt_path, feature_dim, logger):
    """
    KEY FIX: infer which modules were active by inspecting state_dict keys.
    Uses strict=False so missing/extra keys never crash the loader.
    """
    ckpt = torch.load(ckpt_path, map_location=C.DEVICE)

    # unwrap if checkpoint is a dict wrapper
    if isinstance(ckpt, dict) and "model" in ckpt:
        state   = ckpt["model"]
        epoch   = ckpt.get("epoch", "?")
        val_auc = ckpt.get("val_auc", float("nan"))
    else:
        state   = ckpt
        epoch   = "?"
        val_auc = float("nan")

    # Infer flags from which keys are actually present
    use_domain_align = any("domain_disc" in k for k in state.keys())
    use_variational  = any(k.startswith("vae.") for k in state.keys())
    use_ood          = any(k.startswith("ood.") for k in state.keys())
    use_unc_gate     = any("temperature" in k   for k in state.keys())

    auc_str = f"{val_auc:.4f}" if not (isinstance(val_auc, float) and val_auc != val_auc) else "n/a"
    logger.info(f"  Loaded  : epoch={epoch}  val_auc={auc_str}")
    logger.info(f"  Modules : domain={use_domain_align}  vae={use_variational}  "
                f"ood={use_ood}  unc_gate={use_unc_gate}")

    model = URATMIL(
        feature_dim      = feature_dim,
        latent_dim       = C.LATENT_DIM,
        attention_dim    = C.ATTENTION_DIM,
        n_classes        = C.N_CLASSES,
        n_heads          = C.N_HEADS,
        n_prototypes     = C.N_PROTOTYPES,
        dropout          = C.DROPOUT,
        use_domain_align = use_domain_align,
        use_variational  = use_variational,
        use_ood          = use_ood,
        use_unc_gate     = use_unc_gate,
    ).to(C.DEVICE)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.info(f"  Missing keys  : {len(missing)}")
    if unexpected:
        logger.info(f"  Unexpected keys: {len(unexpected)}")

    model.eval()
    return model


@torch.no_grad()
def evaluate(args):
    logger  = make_logger(args.run_name)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "outputs", "plots")
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"=== URAT-MIL Evaluation: {args.run_name} ===")
    logger.info(f"Checkpoint : {args.ckpt}")
    logger.info(f"Stage      : {args.stage}")
    logger.info(f"Device     : {C.DEVICE}")

    # ── Dataset ───────────────────────────────────────────
    test_ds = PCAMBagDataset(
        feat_dir = C.PCAM_FEAT_DIR,
        split    = "test",
        bag_size = C.PCAM_BAG_SIZE,
        n_bags   = C.PCAM_N_TEST,
        seed     = C.SEED + 99,
    )
    feat_dim    = test_ds[0]["features"].shape[-1]
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=bag_collate, num_workers=0)
    logger.info(f"Test bags  : {len(test_ds)}")
    logger.info(f"Feature dim: {feat_dim}")

    # ── Model ─────────────────────────────────────────────
    model = load_model(args.ckpt, feat_dim, logger)

    # ── Inference ─────────────────────────────────────────
    all_probs, all_labels, all_preds = [], [], []
    all_u_alea, all_u_epis           = [], []
    all_ood_scores                   = []
    n_flagged = 0

    for batch in test_loader:
        # bag_collate may return a list of dicts OR a dict with batched tensors
        if isinstance(batch, (list, tuple)):
            item = batch[0]
        elif isinstance(batch, dict):
            # dict-style: features is a list, labels is a tensor
            item = {"features": batch["features"][0], "label": batch["labels"][0]}
        else:
            item = batch
        feats = item["features"].to(C.DEVICE)
        label = int(item["label"]) if torch.is_tensor(item["label"]) else int(item["label"])

        out  = model(feats)
        prob = out["p_hat"].cpu().numpy()
        pred = int(prob.argmax())

        all_probs.append(prob)
        all_labels.append(label)
        all_preds.append(pred)

        # uncertainty — safe get with fallback
        u_a = out.get("u_alea", None)
        u_e = out.get("u_epis", None)
        all_u_alea.append(float(u_a.mean()) if u_a is not None else 0.0)
        all_u_epis.append(float(u_e.mean()) if u_e is not None else 0.0)

        # OOD score — use scalar s_ood if available
        s = out.get("s_ood", None)
        if s is None:
            s = out.get("ood_score", None)
        all_ood_scores.append(float(s.mean()) if s is not None else 0.0)

        gate = out.get("gate_mask", None)
        if gate is not None and float(gate.sum()) / max(feats.size(0), 1) < 0.1:
            n_flagged += 1

    probs      = np.stack(all_probs)
    labels_arr = np.array(all_labels)
    preds      = np.array(all_preds)

    # ── Metrics ───────────────────────────────────────────
    acc = float((preds == labels_arr).mean())
    auc = roc_auc_score(labels_arr, probs[:, 1])
    fpr, tpr, _ = roc_curve(labels_arr, probs[:, 1])
    ece, bin_accs, bin_confs, bin_counts = compute_ece(probs, labels_arr)
    report = classification_report(labels_arr, preds,
                                   target_names=["Negative", "Positive"],
                                   zero_division=0)
    cm = confusion_matrix(labels_arr, preds)

    # ── Print ─────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"  Run          : {args.run_name}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  AUC          : {auc:.4f}")
    print(f"  ECE          : {ece:.4f}")
    print(f"  OOD-flagged  : {n_flagged}/{len(test_ds)}")
    print(f"  Mean U_alea  : {np.mean(all_u_alea):.6f}")
    print(f"  Mean U_epis  : {np.mean(all_u_epis):.6f}")
    print("="*55)
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    # ── Plots ─────────────────────────────────────────────
    plot_roc(fpr, tpr, auc, args.run_name, out_dir)
    plot_confusion(cm, args.run_name, out_dir)
    plot_reliability(bin_accs, bin_confs, bin_counts, ece, args.run_name, out_dir)
    plot_uncertainty(all_u_alea, all_u_epis, labels_arr, args.run_name, out_dir)
    plot_ood(all_ood_scores, labels_arr, args.run_name, out_dir)
    logger.info(f"All plots saved to {out_dir}")

    # ── Save results ──────────────────────────────────────
    import json
    metrics = {
        "run_name": args.run_name, "accuracy": round(acc, 4),
        "auc": round(auc, 4), "ece": round(ece, 4),
        "ood_flagged": n_flagged, "total_bags": len(test_ds),
        "mean_u_alea": float(np.mean(all_u_alea)),
        "mean_u_epis": float(np.mean(all_u_epis)),
        "confusion_matrix": cm.tolist(),
    }
    json_path = os.path.join(out_dir, f"{args.run_name}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics JSON saved to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     required=True)
    parser.add_argument("--stage",    type=int, default=1)
    parser.add_argument("--run_name", default="eval")
    args = parser.parse_args()
    evaluate(args)
