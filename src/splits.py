# -*- coding: utf-8 -*-
"""
Torch-free dataset logic: metadata loading, binary labels, and the
patient-level 70/15/15 split.

The split FAITHFULLY reproduces the released code
(train_cnn_final.py:571-603). Reproduction notes:

* We read metadata.csv WITHOUT reordering rows. The original computed the
  split on the row order of metadata.csv after filtering to the 6 valid
  diagnostics; `patient_id.unique()` returns patients in first-appearance
  order, which is what the shuffle permutes.
* `np.random.RandomState(SEED)` is the legacy Mersenne-Twister generator.
  Its `.shuffle` permutation depends only on the array LENGTH and the seed
  (not on numpy version or dtype), so the split is reproducible across
  environments PROVIDED metadata.csv row order is preserved.
* We therefore freeze the split to a CSV once; downstream code loads that
  file and never regenerates, removing even the row-order dependency.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import config as C


def load_metadata(path: Path | None = None) -> pd.DataFrame:
    """Load metadata.csv and apply the EXACT filtering the training code uses."""
    path = Path(path) if path is not None else C.metadata_path()
    meta = pd.read_csv(path)
    if C.COL_IMG not in meta.columns:
        raise ValueError(f"metadata.csv must contain a '{C.COL_IMG}' column; "
                         f"found: {list(meta.columns)}")
    if C.COL_PATIENT not in meta.columns:
        raise ValueError(f"metadata.csv must contain a '{C.COL_PATIENT}' column")
    if C.COL_DIAG not in meta.columns:
        raise ValueError(f"metadata.csv must contain a '{C.COL_DIAG}' column")

    # Match train_cnn_final.py:581-585 exactly (uppercase/strip, filter, is_cancer)
    meta[C.COL_DIAG] = meta[C.COL_DIAG].astype(str).str.upper().str.strip()
    meta = meta[meta[C.COL_DIAG].isin(C.VALID_DIAGNOSTICS)].copy()
    meta["is_cancer"] = meta[C.COL_DIAG].isin(C.CANCER_LABELS).astype(np.float32)
    return meta.reset_index(drop=True)


def make_patient_split(meta: pd.DataFrame, seed: int = C.SEED):
    """Reproduce train_cnn_final.py:590-603. Returns dict of patient-id sets."""
    # .unique() preserves first-appearance order. On modern pandas it returns a
    # StringArray (ExtensionArray); RandomState.shuffle is "not guaranteed" on
    # non-ndarrays. Coerce to a plain object ndarray — this reproduces exactly
    # what the released code got on older pandas, and makes the legacy-RandomState
    # permutation (which depends only on length + seed, not dtype) deterministic.
    unique_patients = np.asarray(meta[C.COL_PATIENT].unique(), dtype=object)
    rng = np.random.RandomState(seed)                       # legacy generator
    rng.shuffle(unique_patients)                            # in-place permutation
    n = len(unique_patients)
    n_train = int(C.SPLIT_TRAIN * n)
    n_val   = int(C.SPLIT_VAL * n)
    train = set(unique_patients[:n_train])
    val   = set(unique_patients[n_train:n_train + n_val])
    test  = set(unique_patients[n_train + n_val:])
    return {"train": train, "val": val, "test": test}, unique_patients


def assert_disjoint(split: dict) -> None:
    """Hard guarantee: no patient in more than one split."""
    tr, va, te = split["train"], split["val"], split["test"]
    assert tr.isdisjoint(va), "LEAKAGE: train∩val not empty"
    assert tr.isdisjoint(te), "LEAKAGE: train∩test not empty"
    assert va.isdisjoint(te), "LEAKAGE: val∩test not empty"


def split_to_frame(meta: pd.DataFrame, split: dict) -> pd.DataFrame:
    """One row per patient: patient_id, split."""
    rows = []
    for name, pts in split.items():
        for p in pts:
            rows.append((p, name))
    df = pd.DataFrame(rows, columns=[C.COL_PATIENT, "split"])
    # stable order for a clean, diff-able frozen file
    return df.sort_values([C.COL_PATIENT]).reset_index(drop=True)


def save_split(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = Path(path) if path is not None else C.split_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_split(path: Path | None = None) -> dict:
    """Load the FROZEN split from CSV into patient-id sets (canonical source)."""
    path = Path(path) if path is not None else C.split_path()
    df = pd.read_csv(path)
    split = {name: set(g[C.COL_PATIENT]) for name, g in df.groupby("split")}
    for k in ("train", "val", "test"):
        split.setdefault(k, set())
    return split


def labels_for_split(meta: pd.DataFrame, split: dict):
    """Slice metadata rows (lesion/image level) into the three splits by patient."""
    out = {}
    for name in ("train", "val", "test"):
        out[name] = meta[meta[C.COL_PATIENT].isin(split[name])].reset_index(drop=True)
    return out
