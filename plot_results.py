"""
plot_results.py

Reads the training log produced by train.py and generates training curve plots.
Works with the corrected train.py log format which includes both train and val metrics.

Usage:
    python plot_results.py --run_name pcam_full
"""

import os
import re
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(log_path: str):
    train_losses = []
    train_accs   = []
    val_losses   = []
    val_accs     = []
    val_aucs     = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            # Per-bag training loss lines: [bag_idx/total] loss=X ce=X ...
            if re.search(r"\[\d+/\d+\]", line) and "loss=" in line and "Val" not in line:
                m = re.search(r"loss=([0-9.]+)", line)
                if m:
                    train_losses.append(float(m.group(1)))

            # Per-epoch training summary: [Train] loss=X acc=X
            if "[Train]" in line:
                ml = re.search(r"loss=([0-9.]+)", line)
                ma = re.search(r"acc=([0-9.]+)",  line)
                if ml and ma:
                    train_accs.append(float(ma.group(1)))

            # Validation lines: [Val Ep XXX] loss=X acc=X auc=X
            if "[Val Ep" in line:
                m1 = re.search(r"loss=([0-9.]+)", line)
                m2 = re.search(r"acc=([0-9.]+)",  line)
                m3 = re.search(r"auc=([0-9.]+)",  line)
                if m1 and m2 and m3:
                    val_losses.append(float(m1.group(1)))
                    val_accs.append(float(m2.group(1)))
                    val_aucs.append(float(m3.group(1)))

    return train_losses, train_accs, val_losses, val_accs, val_aucs


def make_plots(run_name: str, log_dir: str = "outputs/logs", out_dir: str = "outputs/plots"):
    log_path = os.path.join(log_dir, f"{run_name}.log")
    if not os.path.isfile(log_path):
        print(f"[ERROR] Log file not found: {log_path}")
        return

    os.makedirs(out_dir, exist_ok=True)

    train_losses, train_accs, val_losses, val_accs, val_aucs = parse_log(log_path)

    print(f"Parsed: {len(train_losses)} train-loss points, "
          f"{len(val_losses)} val epochs")

    # ── Training loss (per bag iteration) ──────────────────
    if train_losses:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_losses, linewidth=0.8, alpha=0.7, color="#1A73E8")
        ax.set_title(f"Training Loss — {run_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Bag iterations")
        ax.set_ylabel("Total loss")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{run_name}_train_loss.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    epochs = list(range(1, len(val_losses) + 1))

    # ── Validation loss ────────────────────────────────────
    if val_losses:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, val_losses, marker="o", color="#EA4335", linewidth=1.8)
        ax.set_title(f"Validation Loss — {run_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{run_name}_val_loss.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # ── Validation accuracy ────────────────────────────────
    if val_accs:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, val_accs, marker="o", color="#34A853", linewidth=1.8)
        ax.set_title(f"Validation Accuracy — {run_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim([max(0, min(val_accs) - 0.05), 1.0])
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{run_name}_val_accuracy.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # ── Validation AUC ────────────────────────────────────
    if val_aucs:
        fig, ax = plt.subplots(figsize=(6, 4))
        best_epoch = val_aucs.index(max(val_aucs)) + 1
        ax.plot(epochs, val_aucs, marker="o", color="#1A73E8", linewidth=1.8,
                label=f"Best AUC={max(val_aucs):.4f} @ epoch {best_epoch}")
        ax.axvline(best_epoch, color="k", linestyle="--", linewidth=0.9, alpha=0.6)
        ax.set_title(f"Validation AUC — {run_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC")
        ax.set_ylim([max(0, min(val_aucs) - 0.05), 1.0])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{run_name}_val_auc.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # ── Combined overview ──────────────────────────────────
    if val_losses and val_accs and val_aucs:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (vals, ylabel, color) in zip(axes, [
            (val_losses, "Loss",     "#EA4335"),
            (val_accs,  "Accuracy", "#34A853"),
            (val_aucs,  "AUC",      "#1A73E8"),
        ]):
            ax.plot(epochs, vals, marker="o", color=color, linewidth=1.8)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Val {ylabel}")
            ax.grid(alpha=0.3)
        fig.suptitle(f"URAT-MIL Training Curves — {run_name}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(out_dir, f"{run_name}_overview.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="pcam_full")
    parser.add_argument("--log_dir",  type=str, default="outputs/logs")
    parser.add_argument("--out_dir",  type=str, default="outputs/plots")
    args = parser.parse_args()
    make_plots(args.run_name, args.log_dir, args.out_dir)
