# -*- coding: utf-8 -*-
"""
Central configuration for the ResNet50 baseline reproduction (Target A).

EVERY value below is copied verbatim from the released reference code
(Skin-NeuralNET-main/src/train_cnn_final.py). Source line numbers are given.
DO NOT change these for Target A — they define "the released public code".
Deviations from the paper's prose are documented in PHASE1_AUDIT.md §4.

This module imports NOTHING heavy (no torch), so it is safe to import from
the torch-free validation/split scripts and from the Colab notebook.
"""
from pathlib import Path

# --- Reproducibility (train_cnn_final.py:57, 71-72) ---
SEED        = 42          # random / numpy / torch / cuda
BATCH_SIZE  = 16
EPOCHS      = 20          # hard cap; early stopping usually fires first
LR          = 1e-4        # base LR; group LRs are LR*0.06 (backbone), LR*0.6 (head)
NUM_WORKERS = 0           # released value (macOS-safe). On Colab use 2 (see notebook).
PATIENCE    = 7           # early stopping on val AUC  (paper §3.4 says 10 — MISMATCH D4)
MIN_DELTA   = 1e-4

# --- Optimizer / scheduler (train_cnn_final.py:658-662) ---
BACKBONE_LR_MULT   = 0.06   # backbone group LR = LR * 0.06
HEAD_LR_MULT       = 0.6    # head     group LR = LR * 0.6
BACKBONE_WD        = 0.1
HEAD_WD            = 0.01
SCHED_PATIENCE     = 3      # ReduceLROnPlateau (paper says cosine+warmup — MISMATCH D1)
SCHED_FACTOR       = 0.5
GRAD_CLIP_NORM     = 0.5    # train_cnn_final.py:306

# --- Classifier head (train_cnn_final.py:194-200) ---
HEAD_DROPOUT_1 = 0.6
HEAD_HIDDEN    = 256
HEAD_DROPOUT_2 = 0.4

# --- Train augmentation (train_cnn_final.py:163-170) ---
AUG_HFLIP_P          = 0.5
AUG_COLORJITTER      = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
AUG_AFFINE           = dict(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
AUG_RANDOM_ERASING_P = 0.2
AUG_RANDOM_ERASING_SCALE = (0.02, 0.1)

# --- Label mapping (train_cnn_final.py:578-585; paper §3.1) ---
CANCER_LABELS     = {"BCC", "MEL", "SCC"}   # is_cancer = 1
NON_CANCER_LABELS = {"ACK", "NEV", "SEK"}   # is_cancer = 0
VALID_DIAGNOSTICS = CANCER_LABELS | NON_CANCER_LABELS

# --- Split (train_cnn_final.py:590-599; paper §3.1) ---
SPLIT_TRAIN = 0.70
SPLIT_VAL   = 0.15
# test = remainder (1 - 0.70 - 0.15), taken as the tail after int() truncation

# --- Expected dataset facts (paper §3.1) — used as validation assertions ---
EXPECT_PATIENTS = 1373
EXPECT_LESIONS  = 1641
EXPECT_IMAGES   = 2298
EXPECT_CLASSES  = 6

# --- Metadata column names (train_cnn_final.py:573, 581, 590) ---
COL_IMG   = "img_id"       # renamed to "ID" internally by the training script
COL_PATIENT = "patient_id"
COL_DIAG  = "diagnostic"
COL_LESION = "lesion_id"   # present in PAD-UFES-20 metadata; used only for reporting

# --- Image search behaviour (train_cnn_final.py:116-131) ---
IMG_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")

# --- Paths (resolved relative to the project root = parent of src/) ---
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def data_dir() -> Path:
    return project_root() / "data"

def image_dirs():
    d = data_dir()
    # Doubled nesting, exactly as the released loader expects (train_cnn_final.py:619-623)
    return [
        d / "imgs_part_1" / "imgs_part_1",
        d / "imgs_part_2" / "imgs_part_2",
        d / "imgs_part_3" / "imgs_part_3",
    ]

def metadata_path() -> Path:
    return data_dir() / "metadata.csv"

def split_path() -> Path:
    return project_root() / "splits" / "padufes20_patient_split.csv"
