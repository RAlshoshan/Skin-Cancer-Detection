# -*- coding: utf-8 -*-
"""
Data-loading smoke test. NO training. NO optimizer. NO weight updates.

Confirms, before any GPU time is spent:
  1. The frozen split loads and is leakage-free.
  2. Train/val/test datasets build from the frozen split.
  3. One batch loads from each split with correct shape/dtype/label range.
  4. A single forward pass through ResNet50 produces logits of shape (B, 1).
  5. Patient-level disjointness holds at the DataLoader level (sampled check).

Writes results/smoke_report.{json,txt}. Exit code non-zero on any failure.

Usage:  python smoke_test.py
"""
from __future__ import annotations
import json, sys
import torch
from torch.utils.data import DataLoader
import config as C
import splits as S
import torch_pipeline as TP


def main() -> int:
    report = {"ok": True, "device": None, "batches": {}, "forward_pass": None, "notes": []}
    def fail(m): report["ok"] = False; report["notes"].append("FAIL: " + m)

    TP.seed_everything(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report["device"] = str(device)

    # 1. frozen split
    if not C.split_path().exists():
        fail(f"frozen split not found at {C.split_path()} — run make_split.py first")
        _dump(report); return 1
    split = S.load_split()
    try:
        S.assert_disjoint(split)
    except AssertionError as e:
        fail(str(e)); _dump(report); return 1

    # 2. datasets from frozen split
    meta = S.load_metadata()
    per = S.labels_for_split(meta, split)
    img_dirs = C.image_dirs()
    tf_train = TP.get_resnet50_transforms(train=True)
    tf_eval  = TP.get_resnet50_transforms(train=False)
    datasets = {
        "train": TP.PADUFESDataset(img_dirs, per["train"], transform=tf_train),
        "val":   TP.PADUFESDataset(img_dirs, per["val"],   transform=tf_eval),
        "test":  TP.PADUFESDataset(img_dirs, per["test"],  transform=tf_eval),
    }

    # 3. one batch per split
    for name, ds in datasets.items():
        if len(ds) == 0:
            fail(f"{name} dataset is empty"); continue
        loader = DataLoader(ds, batch_size=C.BATCH_SIZE,
                            shuffle=(name == "train"), num_workers=0)
        imgs, labels = next(iter(loader))
        ok_shape = (imgs.ndim == 4 and imgs.shape[1] == 3)
        ok_label = (labels.ndim == 2 and labels.shape[1] == 1
                    and float(labels.min()) >= 0.0 and float(labels.max()) <= 1.0)
        report["batches"][name] = {
            "n_samples": len(ds),
            "img_batch_shape": list(imgs.shape),
            "img_dtype": str(imgs.dtype),
            "label_batch_shape": list(labels.shape),
            "label_values_seen": sorted(set(labels.flatten().tolist())),
            "shape_ok": bool(ok_shape), "label_ok": bool(ok_label),
        }
        if not ok_shape: fail(f"{name}: unexpected image batch shape {list(imgs.shape)}")
        if not ok_label: fail(f"{name}: labels out of [0,1] or wrong shape")

    # 4. single forward pass (no grad, no training)
    if report["ok"]:
        model = TP.build_resnet50(num_classes=1).to(device).eval()
        imgs, _ = next(iter(DataLoader(datasets["val"], batch_size=min(4, len(datasets["val"])))))
        with torch.no_grad():
            out = model(imgs.to(device))
        report["forward_pass"] = {"output_shape": list(out.shape),
                                  "expected": [imgs.shape[0], 1],
                                  "ok": list(out.shape) == [imgs.shape[0], 1]}
        if not report["forward_pass"]["ok"]:
            fail(f"forward pass produced {list(out.shape)}, expected {[imgs.shape[0],1]}")

    # 5. leakage re-check at image level: no patient_id crosses splits
    pat = {n: set(per[n][C.COL_PATIENT]) for n in ("train", "val", "test")}
    cross = (pat["train"] & pat["val"]) | (pat["train"] & pat["test"]) | (pat["val"] & pat["test"])
    report["image_level_leakage"] = "PASS" if not cross else f"FAIL ({len(cross)} shared patients)"
    if cross: fail("patient_id crosses splits at image level")

    _dump(report)
    return 0 if report["ok"] else 1


def _dump(report):
    rd = C.project_root() / "results"; rd.mkdir(parents=True, exist_ok=True)
    (rd / "smoke_report.json").write_text(json.dumps(report, indent=2))
    lines = ["DATA-LOADING SMOKE TEST", "=" * 23,
             f"RESULT: {'PASS' if report['ok'] else 'FAIL'}   device: {report['device']}", ""]
    for name, b in report.get("batches", {}).items():
        lines.append(f"  {name:5} n={b['n_samples']:<5} img={b['img_batch_shape']} "
                     f"{b['img_dtype']} labels={b['label_batch_shape']} "
                     f"values={b['label_values_seen']} "
                     f"shape_ok={b['shape_ok']} label_ok={b['label_ok']}")
    if report.get("forward_pass"):
        fp = report["forward_pass"]
        lines.append(f"  forward_pass: out={fp['output_shape']} expected={fp['expected']} ok={fp['ok']}")
    lines.append(f"  image_level_leakage: {report.get('image_level_leakage')}")
    if report["notes"]:
        lines += [""] + report["notes"]
    txt = "\n".join(lines); (rd / "smoke_report.txt").write_text(txt); print(txt)


if __name__ == "__main__":
    sys.exit(main())
