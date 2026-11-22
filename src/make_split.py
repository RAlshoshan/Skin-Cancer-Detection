# -*- coding: utf-8 -*-
"""
Generate and FREEZE the patient-level 70/15/15 split, then verify no leakage.
Torch-free. Run once; downstream code loads the frozen CSV (never regenerates).

Reproduces train_cnn_final.py:590-603 via splits.make_patient_split().

Outputs:
  splits/padufes20_patient_split.csv   (patient_id, split)  <-- canonical
  results/split_report.{json,txt}

Idempotency guard: if the frozen file already exists, this script will
regenerate into memory and CONFIRM it matches the file on disk, rather than
silently overwriting. Use --force to overwrite.

Usage:  python make_split.py [--force]
"""
from __future__ import annotations
import json, sys, argparse
import pandas as pd
import config as C
import splits as S


def summarise(meta: pd.DataFrame, split: dict) -> dict:
    per = S.labels_for_split(meta, split)
    out = {"seed": C.SEED, "ratios": [C.SPLIT_TRAIN, C.SPLIT_VAL,
            round(1 - C.SPLIT_TRAIN - C.SPLIT_VAL, 2)], "splits": {}}
    total_imgs = 0
    for name in ("train", "val", "test"):
        df = per[name]
        n_cancer = int(df["is_cancer"].sum())
        out["splits"][name] = {
            "patients": len(split[name]),
            "images": int(len(df)),
            "lesions": int(df[C.COL_LESION].nunique()) if C.COL_LESION in df.columns else None,
            "cancer_images": n_cancer,
            "non_cancer_images": int(len(df) - n_cancer),
            "cancer_fraction": round(n_cancer / max(len(df), 1), 4),
        }
        total_imgs += len(df)
    out["total_patients"] = sum(len(split[n]) for n in ("train", "val", "test"))
    out["total_images"] = total_imgs
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing frozen split")
    args = ap.parse_args()

    meta = S.load_metadata()
    split, _ = S.make_patient_split(meta, seed=C.SEED)
    S.assert_disjoint(split)                       # hard leakage guarantee

    frame = S.split_to_frame(meta, split)
    spath = C.split_path()

    if spath.exists() and not args.force:
        existing = pd.read_csv(spath)
        same = (existing.sort_values(C.COL_PATIENT).reset_index(drop=True)
                .equals(frame.sort_values(C.COL_PATIENT).reset_index(drop=True)))
        print(f"[make_split] Frozen split already exists at {spath}")
        print(f"[make_split] Regenerated split MATCHES frozen file: {same}")
        if not same:
            print("[make_split] WARNING: regenerated split DIFFERS from frozen file. "
                  "Your metadata row order or filtering likely changed. "
                  "Use --force only if you intend to replace the canonical split.")
            return 2
    else:
        S.save_split(frame, spath)
        print(f"[make_split] Frozen split written to {spath}")

    # Verify by reloading the frozen file and re-checking disjointness
    reloaded = S.load_split(spath)
    S.assert_disjoint(reloaded)
    report = summarise(meta, reloaded)
    report["leakage_check"] = "PASS (train/val/test patient sets disjoint)"
    report["frozen_file"] = str(spath)

    results_dir = C.project_root() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "split_report.json").write_text(json.dumps(report, indent=2))

    lines = ["FROZEN SPLIT REPORT", "=" * 19,
             f"seed: {report['seed']}   ratios(train/val/test): {report['ratios']}",
             f"frozen file: {report['frozen_file']}",
             f"leakage: {report['leakage_check']}",
             f"total patients: {report['total_patients']}   total images: {report['total_images']}",
             ""]
    for name in ("train", "val", "test"):
        s = report["splits"][name]
        lines.append(f"  {name:5}  patients={s['patients']:<5} images={s['images']:<5} "
                     f"lesions={s['lesions']}  cancer={s['cancer_images']} "
                     f"non={s['non_cancer_images']} (cancer_frac={s['cancer_fraction']})")
    txt = "\n".join(lines)
    (results_dir / "split_report.txt").write_text(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
