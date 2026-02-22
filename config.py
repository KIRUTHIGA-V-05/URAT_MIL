"""
config.py
Central configuration for URAT-MIL v2.
All paths, hyperparameters, and experiment flags live here.
"""

import os
import torch

# ── Hardware ──────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0          # set to 0 on Windows if DataLoader hangs

# ── Paths ─────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
CKPT_DIR   = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR    = os.path.join(OUTPUT_DIR, "logs")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")

for d in [OUTPUT_DIR, CKPT_DIR, LOG_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Stage 1: PatchCAMELYON → MIL bags ───────────────────
PCAM_H5_TRAIN   = os.path.join(DATA_DIR, "pcam", "camelyonpatch_level_2_split_train_x.h5")
PCAM_H5_TRAIN_Y = os.path.join(DATA_DIR, "pcam", "camelyonpatch_level_2_split_train_y.h5")
PCAM_H5_VAL     = os.path.join(DATA_DIR, "pcam", "camelyonpatch_level_2_split_valid_x.h5")
PCAM_H5_VAL_Y   = os.path.join(DATA_DIR, "pcam", "camelyonpatch_level_2_split_valid_y.h5")
PCAM_H5_TEST    = os.path.join(DATA_DIR, "pcam", "camelyonpatch_level_2_split_test_x.h5")
PCAM_H5_TEST_Y  = os.path.join(DATA_DIR, "pcam", "camelyonpatch_level_2_split_test_y.h5")

PCAM_FEAT_DIR   = os.path.join(DATA_DIR, "pcam_features")   # pre-extracted .pt files

# bag construction
PCAM_BAG_SIZE   = 32       # patches per synthetic MIL bag
PCAM_N_TRAIN    = 2000     # bags to build for training
PCAM_N_VAL      = 400
PCAM_N_TEST     = 400

# ── Stage 2: CLAM-style WSI feature bags ─────────────────
WSI_FEAT_DIR_A  = os.path.join(DATA_DIR, "wsi_feats_source")
WSI_FEAT_DIR_B  = os.path.join(DATA_DIR, "wsi_feats_target")
WSI_LABELS_A    = os.path.join(DATA_DIR, "wsi_labels_source.csv")
WSI_LABELS_B    = os.path.join(DATA_DIR, "wsi_labels_target.csv")

# ── Feature extractor ─────────────────────────────────────
FEATURE_DIM     = 512      # ResNet18 avgpool output
PCAM_IMG_SIZE   = 96       # PatchCAMELYON native size
FEAT_BATCH_SIZE = 256      # patches per batch for feature extraction

# ── MIL model ─────────────────────────────────────────────
LATENT_DIM      = 256
ATTENTION_DIM   = 128
N_CLASSES       = 2
N_HEADS         = 4
DROPOUT         = 0.25

# ── Variational head (Module 3) ───────────────────────────
LATENT_DIM_VAE  = 256
BETA_START      = 0.0
BETA_MAX        = 0.5      # FIX: was 1.0 — lower prevents posterior collapse
WARMUP_EPOCHS   = 25       # FIX: was 15  — more warmup keeps KL meaningful
MC_SAMPLES      = 20       # Monte Carlo forward passes at inference

# ── OOD module (Module 4) ─────────────────────────────────
N_PROTOTYPES    = 64
OOD_ALPHA       = 0.5
DELTA_ARTIFACT  = 0.6
DELTA_NEAR      = 0.4
OOD_FPR_TARGET  = 0.05

# ── Domain alignment (Module 2) ──────────────────────────
ALPHA_GRL       = 1.0
GAMMA_MMD       = 1.0

# ── Training ──────────────────────────────────────────────
SEED            = 42
MAX_EPOCHS      = 50
LR              = 1e-4
WEIGHT_DECAY    = 1e-5
GRAD_CLIP       = 1.0
BATCH_SIZE      = 1         # one bag per step (standard MIL)
EARLY_STOP_PAT  = 10        # patience in epochs
LABEL_SMOOTHING = 0.05

# ── Loss weights ──────────────────────────────────────────
LAMBDA_KL       = 1.0
LAMBDA_ALIGN    = 1.0
LAMBDA_CAL      = 0.1
LAMBDA_RECON    = 0.1

# ── Ablation flags ────────────────────────────────────────
USE_UNCERTAINTY_GATE = True
USE_DOMAIN_ALIGN     = True
USE_OOD_FILTER       = True
USE_CALIBRATION      = True

# ── ECE / calibration ─────────────────────────────────────
N_CAL_BINS      = 15

# ── Logging ───────────────────────────────────────────────
LOG_INTERVAL    = 10    # log every N bags
