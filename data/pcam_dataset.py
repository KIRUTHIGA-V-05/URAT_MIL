"""
data/pcam_dataset.py

Stage 1: PatchCAMELYON (PCAM) → MIL bags.

PCAM original format: HDF5 files from Kaggle / TensorFlow Datasets.
  - camelyonpatch_level_2_split_train_x.h5  shape (262144, 96, 96, 3)
  - camelyonpatch_level_2_split_train_y.h5  shape (262144, 1, 1, 1)

Bag construction rule (paper §IV-A):
  A bag is positive (y=1) if ≥1 patch in the bag is positive.
  Bags are constructed with ~50% positive / 50% negative bag label balance
  to prevent the model from exploiting label frequency as a shortcut.

FIX vs original:
  - Bag labels are now explicitly balanced 50/50 (positive/negative).
    Original used randint(0, bag_size) which caused ~97% positive bags,
    leading to the frozen val accuracy of 0.9625 observed in training logs.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from torchvision import transforms


# ── Transforms ────────────────────────────────────────────
def pcam_train_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def pcam_val_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


class PCAMPatchDataset(Dataset):
    """
    Raw PatchCAMELYON patch dataset (used only for feature extraction).
    Each item is a single 96×96 patch and its binary label.
    """
    def __init__(self, h5_x: str, h5_y: str, transform=None, max_samples: int = None):
        self.h5_x      = h5_x
        self.h5_y      = h5_y
        self.transform = transform

        with h5py.File(h5_x, "r") as f:
            self.n = f["x"].shape[0]
        if max_samples:
            self.n = min(self.n, max_samples)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        with h5py.File(self.h5_x, "r") as fx:
            img = fx["x"][idx]                    # (96, 96, 3) uint8
        with h5py.File(self.h5_y, "r") as fy:
            label = int(fy["y"][idx].squeeze())   # 0 or 1

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, label


class PCAMBagDataset(Dataset):
    """
    MIL bag dataset from pre-extracted PCAM features.

    Expects a directory of .pt files where each file contains:
        {"features": Tensor(N, D), "labels": Tensor(N,)}

    Bag label = 1 if ANY patch label == 1, else 0.

    Bags are explicitly balanced: exactly 50% positive, 50% negative bags
    so the model cannot exploit label frequency.

    Positive bags: n_pos ∈ [1, bag_size] positive patches (min 1 guaranteed)
    Negative bags: all patches are negative (n_pos = 0)
    """
    def __init__(
        self,
        feat_dir:    str,
        split:       str  = "train",
        bag_size:    int  = 32,
        n_bags:      int  = 2000,
        seed:        int  = 42,
        max_patches: int  = 512,
        pos_ratio:   float = 0.5,   # fraction of bags that are positive
    ):
        self.feat_dir    = feat_dir
        self.bag_size    = bag_size
        self.max_patches = max_patches

        feat_file = os.path.join(feat_dir, f"{split}_features.pt")
        if not os.path.isfile(feat_file):
            raise FileNotFoundError(
                f"Feature file not found: {feat_file}\n"
                f"Run scripts/extract_pcam_features.py first."
            )

        data       = torch.load(feat_file, map_location="cpu")
        all_feats  = data["features"]   # (N_total, D)
        all_labels = data["labels"]     # (N_total,)

        pos_idx = (all_labels == 1).nonzero(as_tuple=True)[0].tolist()
        neg_idx = (all_labels == 0).nonzero(as_tuple=True)[0].tolist()

        rng = random.Random(seed)
        self.bags = []

        n_pos_bags = int(round(n_bags * pos_ratio))
        n_neg_bags = n_bags - n_pos_bags

        # ── Positive bags (at least 1 positive patch) ──────
        for _ in range(n_pos_bags):
            n_pos_patches = rng.randint(1, bag_size)          # guaranteed ≥ 1
            n_neg_patches = bag_size - n_pos_patches

            chosen_pos = rng.choices(pos_idx, k=n_pos_patches)
            chosen_neg = rng.choices(neg_idx, k=n_neg_patches) if n_neg_patches > 0 else []
            chosen     = chosen_pos + chosen_neg
            rng.shuffle(chosen)

            self.bags.append({
                "features":     all_feats[chosen],
                "patch_labels": all_labels[chosen],
                "label":        1,
            })

        # ── Negative bags (all patches negative) ───────────
        for _ in range(n_neg_bags):
            chosen = rng.choices(neg_idx, k=bag_size)

            self.bags.append({
                "features":     all_feats[chosen],
                "patch_labels": all_labels[chosen],
                "label":        0,
            })

        # Shuffle so pos/neg bags are interleaved
        rng.shuffle(self.bags)

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        return {
            "features":     bag["features"],           # (N, D)
            "patch_labels": bag["patch_labels"],       # (N,)
            "label":        torch.tensor(bag["label"], dtype=torch.long),
        }


class WSIFeatureBagDataset(Dataset):
    """
    Stage 2: CLAM-style WSI feature bags.

    Directory layout:
        feat_dir/
          slide_001.pt   → Tensor(N, feature_dim)
          slide_002.pt
          ...

    Labels CSV: columns [slide_id, label]
    """
    def __init__(
        self,
        feat_dir:    str,
        labels_csv:  str,
        max_patches: int  = 4096,
        shuffle:     bool = True,
        seed:        int  = 42,
    ):
        import pandas as pd

        self.feat_dir    = feat_dir
        self.max_patches = max_patches
        self.shuffle     = shuffle
        self.rng         = random.Random(seed)

        df = pd.read_csv(labels_csv)
        self.samples = []
        for _, row in df.iterrows():
            pt = os.path.join(feat_dir, f"{row['slide_id']}.pt")
            if os.path.isfile(pt):
                self.samples.append({
                    "path":     pt,
                    "label":    int(row["label"]),
                    "slide_id": str(row["slide_id"]),
                })

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No .pt files matched entries in {labels_csv}.\n"
                f"Check feat_dir={feat_dir} and CSV slide_id column."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        feats = torch.load(s["path"], map_location="cpu")

        if isinstance(feats, dict):
            feats = feats.get("features", feats.get("feat", list(feats.values())[0]))
        feats = feats.float()

        if feats.size(0) > self.max_patches:
            perm  = torch.randperm(feats.size(0))[:self.max_patches]
            feats = feats[perm]

        return {
            "features": feats,
            "label":    torch.tensor(s["label"], dtype=torch.long),
            "slide_id": s["slide_id"],
        }


def bag_collate(batch):
    """Variable-length bag collate — keeps bags as a list."""
    return {
        "features":  [b["features"]  for b in batch],
        "labels":    torch.stack([b["label"] for b in batch]),
        "slide_ids": [b.get("slide_id", "") for b in batch],
    }
