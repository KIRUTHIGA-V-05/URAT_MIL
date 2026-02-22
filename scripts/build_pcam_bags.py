"""
scripts/build_pcam_bags.py

Step-by-step guide for Stage 1 data preparation.

STEP 1: Download PatchCAMELYON from Kaggle.
    kaggle datasets download -d andrewmvd/camelyon-patch-level-2-split
    Unzip into:  data/pcam/

STEP 2: Run this script to extract ResNet18 features (GPU-accelerated).
    python scripts/extract_pcam_features.py

STEP 3: This script validates the extracted features and reports bag statistics.
    python scripts/build_pcam_bags.py

Paper: Stage 1 validation on PCAM follows the bag-construction rule
       from §IV-A — a bag is labelled positive iff at least one patch is positive.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from data.pcam_dataset import PCAMBagDataset


def verify_split(split: str, n_bags: int, seed_offset: int = 0):
    feat_file = os.path.join(C.PCAM_FEAT_DIR, f"{split}_features.pt")
    if not os.path.isfile(feat_file):
        print(f"[MISSING] {feat_file}")
        print(f"  → Run: python scripts/extract_pcam_features.py")
        return

    data    = torch.load(feat_file, map_location="cpu")
    feats   = data["features"]
    labels  = data["labels"]
    n_pos   = labels.sum().item()
    n_total = len(labels)

    print(f"\n[{split.upper()}] Feature file: OK")
    print(f"  Total patches : {n_total:,}")
    print(f"  Feature dim   : {feats.size(1)}")
    print(f"  Positive rate : {n_pos/n_total:.3f}  ({n_pos:,}/{n_total:,})")

    try:
        ds = PCAMBagDataset(
            feat_dir  = C.PCAM_FEAT_DIR,
            split     = split,
            bag_size  = C.PCAM_BAG_SIZE,
            n_bags    = n_bags,
            seed      = C.SEED + seed_offset,
        )
        bag_labels = torch.tensor([ds[i]["label"].item() for i in range(len(ds))])
        print(f"  Bags constructed : {len(ds)}")
        print(f"  Bag positive rate: {bag_labels.float().mean().item():.3f}")
        print(f"  Sample bag shape : {ds[0]['features'].shape}")
    except Exception as e:
        print(f"  [ERROR] Bag construction failed: {e}")


def main():
    print("=" * 55)
    print("URAT-MIL — Stage 1 Data Verification")
    print("=" * 55)

    for split, n, off in [("train", C.PCAM_N_TRAIN, 0),
                          ("val",   C.PCAM_N_VAL,   1),
                          ("test",  C.PCAM_N_TEST,  99)]:
        verify_split(split, n, off)

    print("\n" + "=" * 55)
    print("If all splits show OK, run:")
    print("  python train.py --stage 1 --run_name pcam_full")
    print("  python train.py --stage 1 --run_name pcam_nounc --ablate_unc")
    print("  python evaluation/evaluate.py --ckpt outputs/checkpoints/pcam_full_best.pt --stage 1 --run_name eval_stage1")
    print("=" * 55)


if __name__ == "__main__":
    main()
