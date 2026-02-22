"""
evaluation/ablation.py

Ablation study runner for URAT-MIL.

Trains and evaluates four variants:
  1. Full URAT-MIL          (all modules enabled)
  2. No uncertainty gating  (--ablate_unc)
  3. No variational head    (--ablate_vae)
  4. No OOD filter          (--ablate_ood)

Produces a summary table and bar chart comparing AUC / ECE / Acc.

Usage:
    python evaluation/ablation.py --stage 1
"""

import os
import sys
import argparse
import subprocess
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


VARIANTS = [
    {
        "name":        "full_urat_mil",
        "label":       "Full URAT-MIL",
        "train_flags": [],
    },
    {
        "name":        "ablate_unc",
        "label":       "No Unc. Gate",
        "train_flags": ["--ablate_unc"],
    },
    {
        "name":        "ablate_vae",
        "label":       "No VAE",
        "train_flags": ["--ablate_vae"],
    },
    {
        "name":        "ablate_ood",
        "label":       "No OOD Filter",
        "train_flags": ["--ablate_ood"],
    },
]


def run_variant(variant: dict, stage: int, python: str = sys.executable):
    run_name = f"{variant['name']}_s{stage}"
    ckpt     = os.path.join(C.CKPT_DIR, f"{run_name}_best.pt")

    if os.path.isfile(ckpt):
        print(f"[SKIP TRAIN] {run_name} — checkpoint exists")
        return ckpt

    cmd = [
        python, "train.py",
        "--stage",    str(stage),
        "--run_name", run_name,
    ] + variant["train_flags"]

    print(f"\n>>> Training: {run_name}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
    if result.returncode != 0:
        print(f"[ERROR] Training failed for {run_name}")
        return None
    return ckpt


def run_eval(ckpt: str, variant: dict, stage: int, python: str = sys.executable) -> dict:
    run_name = f"{variant['name']}_s{stage}"
    res_path = os.path.join(C.PLOT_DIR, f"{run_name}_results.pt")

    if not os.path.isfile(res_path):
        cmd = [
            python, "evaluation/evaluate.py",
            "--ckpt",     ckpt,
            "--stage",    str(stage),
            "--run_name", run_name,
        ]
        print(f"\n>>> Evaluating: {run_name}")
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
        if result.returncode != 0:
            print(f"[ERROR] Evaluation failed for {run_name}")
            return {}

    if os.path.isfile(res_path):
        return torch.load(res_path, map_location="cpu")
    return {}


def plot_ablation_comparison(results_list: list, out_dir: str, stage: int):
    labels = [r["label"] for r in results_list]
    aucs   = [r.get("auc",      0.0) for r in results_list]
    accs   = [r.get("accuracy", 0.0) for r in results_list]
    eces   = [r.get("ece",      1.0) for r in results_list]

    x      = np.arange(len(labels))
    width  = 0.22
    colors = ["#1A73E8", "#34A853", "#EA4335"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, aucs, width, label="AUC",      color=colors[0], alpha=0.85)
    bars2 = ax.bar(x,         accs, width, label="Accuracy", color=colors[1], alpha=0.85)
    bars3 = ax.bar(x + width, eces, width, label="ECE ↓",    color=colors[2], alpha=0.85)

    def autolabel(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    autolabel(bars1); autolabel(bars2); autolabel(bars3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"URAT-MIL Ablation Study — Stage {stage}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, f"ablation_stage{stage}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nAblation plot saved: {path}")
    return path


def print_table(results_list: list):
    header = f"{'Variant':<22} {'AUC':>8} {'Accuracy':>10} {'ECE':>8} {'OOD Flagged':>14}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results_list:
        print(
            f"{r['label']:<22} "
            f"{r.get('auc', 0):.4f}   "
            f"{r.get('accuracy', 0):.4f}     "
            f"{r.get('ece', 0):.4f}   "
            f"{r.get('n_ood_flagged', 0):>5}/{r.get('n_bags', 0):<5}"
        )
    print("=" * len(header))


def run_ablation(stage: int):
    os.makedirs(C.PLOT_DIR, exist_ok=True)
    combined = []

    for variant in VARIANTS:
        ckpt = run_variant(variant, stage)
        if ckpt is None or not os.path.isfile(ckpt):
            print(f"[WARN] No checkpoint for {variant['name']} — skipping eval")
            combined.append({"label": variant["label"]})
            continue

        res = run_eval(ckpt, variant, stage)
        res["label"] = variant["label"]
        combined.append(res)

    print_table(combined)
    if len(combined) > 1:
        plot_ablation_comparison(combined, C.PLOT_DIR, stage)

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1)
    args = parser.parse_args()
    run_ablation(args.stage)
