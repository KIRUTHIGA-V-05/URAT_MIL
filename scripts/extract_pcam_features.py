"""
scripts/extract_pcam_features.py

GPU-accelerated ResNet18 feature extraction from raw PatchCAMELYON HDF5 files.
Produces train_features.pt, val_features.pt, test_features.pt in data/pcam_features/.

Usage:
    python scripts/extract_pcam_features.py

Paper contribution: Module 1 feature encoding backbone.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from data.pcam_dataset import PCAMPatchDataset, pcam_val_transform


def build_resnet18_extractor(device: torch.device) -> nn.Module:
    """
    ResNet18 with classifier head removed.
    Output: (N, 512) average-pooled features.
    Paper: Module 1 backbone (ResNet18 used for Stage 1 validation).
    """
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net.eval()
    return net.to(device)


@torch.no_grad()
def extract_split(
    net:        nn.Module,
    h5_x:       str,
    h5_y:       str,
    split_name: str,
    out_dir:    str,
    device:     torch.device,
    batch_size: int,
    max_samples: int = None,
):
    if not os.path.isfile(h5_x):
        print(f"[SKIP] {split_name}: file not found → {h5_x}")
        return

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split_name}_features.pt")

    if os.path.isfile(out_path):
        print(f"[SKIP] {split_name}: features already exist → {out_path}")
        return

    print(f"\n[{split_name.upper()}] Loading patches …")
    ds = PCAMPatchDataset(h5_x, h5_y, transform=pcam_val_transform(),
                          max_samples=max_samples)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=C.NUM_WORKERS, pin_memory=True)

    all_feats  = []
    all_labels = []

    for i, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        feats = net(imgs).cpu()
        all_feats.append(feats)
        all_labels.append(labels)

        if (i + 1) % 20 == 0:
            done = min((i + 1) * batch_size, len(ds))
            print(f"  {done}/{len(ds)} patches processed …", flush=True)

    all_feats  = torch.cat(all_feats,  dim=0)   # (N, 512)
    all_labels = torch.cat(all_labels, dim=0)   # (N,)

    torch.save({"features": all_feats, "labels": all_labels}, out_path)
    print(f"  Saved → {out_path}  shape={tuple(all_feats.shape)}")


def main():
    device = C.DEVICE
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    net = build_resnet18_extractor(device)
    print(f"Feature extractor: ResNet18 (output_dim=512)")

    splits = [
        ("train", C.PCAM_H5_TRAIN,  C.PCAM_H5_TRAIN_Y, None),
        ("val",   C.PCAM_H5_VAL,    C.PCAM_H5_VAL_Y,   None),
        ("test",  C.PCAM_H5_TEST,   C.PCAM_H5_TEST_Y,  None),
    ]

    for name, hx, hy, max_s in splits:
        extract_split(net, hx, hy, name, C.PCAM_FEAT_DIR,
                      device, C.FEAT_BATCH_SIZE, max_s)

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()
