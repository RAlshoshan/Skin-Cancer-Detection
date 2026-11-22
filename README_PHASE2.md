# Phase 2 — Data Preparation (ResNet50 baseline, Target A)

Data prep only. **No training.** Target A = reproduce the released public code
(`train_cnn_final.py`), documenting deviations from the paper (see
`PHASE1_AUDIT.md` §4).

## What's here
```
src/config.py           all hyperparameters, copied verbatim from the released code (sourced by line)
src/splits.py           torch-free: metadata load, binary labels, patient split (faithful to code)
src/validate_dataset.py torch-free: dataset validation report
src/make_split.py       torch-free: freeze the 70/15/15 split + leakage check
src/torch_pipeline.py   torch: PADUFESDataset, ResNet50 transforms, ResNet50 builder (faithful copies)
src/smoke_test.py       torch: one-batch load + one forward pass, NO training
env/requirements.txt    pinned light deps (use Colab's preinstalled torch/torchvision)
notebooks/colab_phase2.ipynb  orchestration notebook for Colab
```

## Step 1 — Get the data (the repo cannot download it)
The reference repo's `download_data.py` is broken — it points at a dead personal
server and a *thyroid* dataset, not PAD-UFES-20. Download manually:

1. Go to the Mendeley record `zr7vgbcyr2` (PAD-UFES-20) and download:
   `metadata.csv`, `imgs_part_1.zip`, `imgs_part_2.zip`, `imgs_part_3.zip`.
2. Extract so the final layout is **exactly** (note the doubled nesting):
```
data/
  metadata.csv
  imgs_part_1/imgs_part_1/*.png
  imgs_part_2/imgs_part_2/*.png
  imgs_part_3/imgs_part_3/*.png
```

## Step 2 — Put the project in Google Drive
Upload this whole folder to `MyDrive/skin-cancer-phase1/` (or your chosen path),
with the `data/` contents from Step 1 inside it. The notebook defaults to
`/content/drive/MyDrive/skin-cancer-phase1` — edit `PROJECT_IN_DRIVE` if different.

Keeping `data/`, `splits/`, `results/` on Drive means nothing is lost when the
Colab runtime disconnects.

## Step 3 — Connect Drive in Colab
Open `notebooks/colab_phase2.ipynb` in Colab, set Runtime → Change runtime type →
**GPU (T4 is fine)**, then run cells top to bottom. Cell 2 calls
`drive.mount('/content/drive')` and prompts you to authorise your Google account.

## Step 4 — Run the three scripts (the notebook does this for you)
```
python validate_dataset.py   # -> results/validation_report.{txt,json}
python make_split.py          # -> splits/padufes20_patient_split.csv + results/split_report.{txt,json}
python smoke_test.py          # -> results/smoke_report.{txt,json}
```

## Reproducibility notes
- The split is generated once by `make_split.py` and **frozen** to
  `splits/padufes20_patient_split.csv`. Everything downstream loads that file;
  nothing regenerates. Re-running `make_split.py` without `--force` re-derives
  the split in memory and confirms it still matches the frozen file.
- The split faithfully reproduces `train_cnn_final.py:590-603`. `patient_id`
  uniques are coerced to a numpy object array before shuffling — this matches
  what the original produced on older pandas and removes a nondeterminism
  warning that appears on recent pandas (the legacy-`RandomState` permutation
  depends only on array length + seed, so the result is identical).
- `env/env_snapshot.txt` records the exact torch/numpy/etc. versions the Colab
  runtime gave you — attach it to any result you report.

## Guardrails (unchanged from the brief)
- No simulated numbers. Validation/split/smoke reports contain only what your
  data produces.
- Test set is never used for selection or tuning.
- No patient across splits (asserted in `make_split.py` and `smoke_test.py`).
- VGG16 / AlexNet / multiclass / mobile app: **not started.**
- Training: **not started** (Phase 3).
