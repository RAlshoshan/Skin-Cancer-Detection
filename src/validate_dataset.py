# -*- coding: utf-8 -*-
"""
Dataset validation for PAD-UFES-20. Torch-free (runs anywhere).

Checks performed:
  1. metadata.csv exists and has the required columns.
  2. Diagnostic values are within the 6 expected classes (reports any extras).
  3. Patient / lesion / image counts vs the paper's expected figures.
  4. Every metadata row resolves to an image file on disk (and which naming
     pattern matched: img_id-with-extension, img_id+ext, or MISSING).
  5. Orphan image files on disk not referenced by metadata.
  6. Class balance and cancer/non-cancer balance.

Writes a JSON + human-readable report to results/validation_report.{json,txt}.
Exit code is non-zero if any hard check fails, so Colab stops early.

Usage:  python validate_dataset.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
import config as C
import splits as S


def _resolve_image(img_id: str, image_dirs) -> tuple[str, str | None]:
    """Mirror the loader's search (train_cnn_final.py:116-131)."""
    for d in image_dirs:
        for ext in C.IMG_EXTS:
            if (d / f"{img_id}{ext}").exists():
                return "id+ext", str(d / f"{img_id}{ext}")
    for d in image_dirs:                      # fallback: id already has extension
        if (d / img_id).exists():
            return "id-as-is", str(d / img_id)
    return "MISSING", None


def main() -> int:
    results_dir = C.project_root() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"ok": True, "hard_failures": [], "warnings": [], "stats": {}}

    def fail(msg):
        report["ok"] = False
        report["hard_failures"].append(msg)

    def warn(msg):
        report["warnings"].append(msg)

    # 1. metadata
    mpath = C.metadata_path()
    if not mpath.exists():
        fail(f"metadata.csv not found at {mpath}")
        _dump(report, results_dir); return 1
    meta_raw = pd.read_csv(mpath)
    report["stats"]["metadata_rows_raw"] = int(len(meta_raw))
    report["stats"]["metadata_columns"] = list(meta_raw.columns)

    # 2. filtered labels
    try:
        meta = S.load_metadata(mpath)
    except ValueError as e:
        fail(str(e)); _dump(report, results_dir); return 1

    extras = set(meta_raw[C.COL_DIAG].astype(str).str.upper().str.strip().unique()) - C.VALID_DIAGNOSTICS
    if extras:
        warn(f"metadata contains diagnostics outside the 6 expected classes "
             f"(dropped by the filter): {sorted(extras)}")

    n_patients = meta[C.COL_PATIENT].nunique()
    n_images   = len(meta)
    n_lesions  = meta[C.COL_LESION].nunique() if C.COL_LESION in meta.columns else None
    report["stats"].update(
        patients=int(n_patients), images=int(n_images),
        lesions=(int(n_lesions) if n_lesions is not None else "column_absent"),
        classes=int(meta[C.COL_DIAG].nunique()),
    )

    # 3. counts vs paper (warn, not hard-fail — your copy may differ slightly)
    for label, got, exp in [("patients", n_patients, C.EXPECT_PATIENTS),
                            ("images", n_images, C.EXPECT_IMAGES),
                            ("classes", meta[C.COL_DIAG].nunique(), C.EXPECT_CLASSES)]:
        if got != exp:
            warn(f"{label} = {got}, paper reports {exp}")
    if n_lesions is not None and n_lesions != C.EXPECT_LESIONS:
        warn(f"lesions = {n_lesions}, paper reports {C.EXPECT_LESIONS}")

    # 4. image resolution
    image_dirs = C.image_dirs()
    missing_dirs = [str(d) for d in image_dirs if not d.exists()]
    if missing_dirs:
        warn(f"image directories not found (check the doubled nesting "
             f"imgs_part_N/imgs_part_N/): {missing_dirs}")
    patterns = {"id+ext": 0, "id-as-is": 0, "MISSING": 0}
    missing_examples = []
    for img_id in meta[C.COL_IMG].astype(str):
        patt, _ = _resolve_image(img_id, image_dirs)
        patterns[patt] += 1
        if patt == "MISSING" and len(missing_examples) < 10:
            missing_examples.append(img_id)
    report["stats"]["image_match_patterns"] = patterns
    if patterns["MISSING"] > 0:
        fail(f"{patterns['MISSING']} metadata rows have NO image file on disk "
             f"(examples: {missing_examples})")

    # 5. orphans on disk
    on_disk = set()
    for d in image_dirs:
        if d.exists():
            on_disk |= {p.name for p in d.glob("*") if p.is_file()}
    referenced = set()
    for img_id in meta[C.COL_IMG].astype(str):
        referenced.add(img_id)
        for ext in C.IMG_EXTS:
            referenced.add(f"{img_id}{ext}")
    orphans = on_disk - referenced
    report["stats"]["orphan_image_files"] = int(len(orphans))
    if orphans:
        warn(f"{len(orphans)} image files on disk are not referenced by metadata")

    # 6. balance
    vc = meta[C.COL_DIAG].value_counts().to_dict()
    report["stats"]["class_counts"] = {k: int(v) for k, v in vc.items()}
    n_cancer = int(meta["is_cancer"].sum())
    report["stats"]["cancer_images"] = n_cancer
    report["stats"]["non_cancer_images"] = int(n_images - n_cancer)
    report["stats"]["cancer_fraction"] = round(n_cancer / max(n_images, 1), 4)

    _dump(report, results_dir)
    return 0 if report["ok"] else 1


def _dump(report, results_dir):
    (results_dir / "validation_report.json").write_text(json.dumps(report, indent=2))
    lines = ["PAD-UFES-20 VALIDATION REPORT", "=" * 32,
             f"RESULT: {'PASS' if report['ok'] else 'FAIL'}", ""]
    if report["hard_failures"]:
        lines += ["HARD FAILURES:"] + [f"  - {m}" for m in report["hard_failures"]] + [""]
    if report["warnings"]:
        lines += ["WARNINGS:"] + [f"  - {m}" for m in report["warnings"]] + [""]
    lines += ["STATS:"]
    for k, v in report["stats"].items():
        lines.append(f"  {k}: {v}")
    txt = "\n".join(lines)
    (results_dir / "validation_report.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    sys.exit(main())
