# Phase 1 — Audit & Reproducibility Plan
## Skin Cancer Detection on PAD-UFES-20 — ResNet50 baseline (Dey et al., 2026)

**Scope of this document:** audit only. No training performed. No results invented. Every ResNet50 setting below is tagged with its source: `CODE:file:line`, `PAPER:§`, or flagged as **MISSING** / **MISMATCH**. Confidence tags: **[confirmed]** (read directly), **[strong inference]**, **[assumption]**.

Reference paper: *CNNs, Transformers, Hybrid, and Vision Language Models for Skin Cancer Detection* — Dey, Yan, Hajjdiab (ICPRAI 2026), arXiv:2605.26294v1.
Code audited: `Skin-NeuralNET-main/` (the uploaded zip; mirror of github.com/Durjoy001/Skin-NeuralNET).

---

## 0. Bottom line up front

1. **The ResNet50 baseline is implemented in `src/train_cnn_final.py`** — one script trains all four CNNs (ResNet50, EfficientNetB3, InceptionV3, DenseNet121). It is self-contained and runnable.
2. **The released CNN code contradicts the paper's Methodology in four material ways** (scheduler, mixed precision, gradient accumulation, early-stopping patience). See §4. This means you have **two different reproduction targets**, and you must decide which one you are reproducing.
3. **The data-download script in the repo is broken/wrong** — it points at a dead personal server and a *thyroid* dataset, not PAD-UFES-20. You must download the data manually from Mendeley. See §5 and §8.
4. **There is no saved split, no config file, and no pinned dependency versions.** The split is regenerated at runtime from `SEED=42`. It is deterministic *if* `metadata.csv` row order and the NumPy version are held fixed. This is the single biggest reproducibility risk. See §6.
5. **Exact reproduction of AUC = 0.893131 is not guaranteed** from the released code, for the reasons in §4 and §6. What *is* achievable is a faithful re-run of the public code and a fair, documented comparison. See §8.
6. **VGG16 and AlexNet are not in the paper and not in the code.** They are your extension, and they require code changes (the model registry and transform registry both raise `KeyError` for them today). See §10.

---

## 1. Dataset facts (PAD-UFES-20)

| Fact | Value | Source |
|---|---|---|
| Patients | 1,373 | PAPER:§3.1 **[confirmed]** |
| Lesions | 1,641 | PAPER:§3.1 **[confirmed]** |
| Images (`.png`) | 2,298 | PAPER:§3.1 **[confirmed]** |
| Classes (6) | BCC, SCC, ACK, SEK, MEL, NEV | PAPER:§3.1 **[confirmed]** |
| Cancer (`is_cancer=1`) | BCC, MEL, SCC | PAPER:§3.1 + CODE:`train_cnn_final.py:578` **[confirmed]** |
| Non-cancer (`is_cancer=0`) | ACK, NEV, SEK | PAPER:§3.1 + CODE:`train_cnn_final.py:579` **[confirmed]** |
| Metadata columns used | `img_id`, `patient_id`, `diagnostic` | CODE:`train_cnn_final.py:573,581,590` **[confirmed]** |
| Source | Mendeley `zr7vgbcyr2/1` (Pacheco et al., Brazilian primary-care smartphone images) | PAPER:§3.1 **[confirmed]** |

**Image/patient/lesion relationship:** one patient → one or more lesions → one or more images. A leakage-free split must therefore be keyed on `patient_id` (grouping by lesion alone is insufficient, because one patient can have several lesions). The code does this correctly (§6).

**Expected on-disk layout the code assumes** (CODE:`train_cnn_final.py:13-28, 619-623`):
```
data/
  metadata.csv
  imgs_part_1/imgs_part_1/<img_id>.png
  imgs_part_2/imgs_part_2/<img_id>.png
  imgs_part_3/imgs_part_3/<img_id>.png
```
Note the **doubled nesting** (`imgs_part_1/imgs_part_1/`). The Mendeley download gives three zips; you must extract them into exactly this nested shape or the loader will not find the images (it searches the three dirs by `img_id` + extension, CODE:`116-134`).

---

## 2. Which script does what

| Script | Role | Relevant to Phase 1? |
|---|---|---|
| `src/train_cnn_final.py` | **Dataset prep + patient split + ResNet50/EfficientNetB3/InceptionV3/DenseNet121 training + full evaluation**. The one file for the baseline. | **YES — primary** |
| `src/train_vit_final.py` | ViT family (Swin/ViT/DeiT/CrossViT), timm-based | No (Phase 2+) |
| `src/train_hybrid_final.py` | Hybrids (CoAtNet0, MaxViTTiny), timm-based | No |
| `src/train_vlm_final.py` | CLIP / SigLIP | No |
| `generate_all_overall_metrics.py` | Recomputes overall test metrics from saved `.pth` checkpoints | Useful later for reporting |
| `statistical_tests/statistical_tests.py` | DeLong (AUC) + McNemar tests between models | **Useful for ResNet50 vs VGG16 vs AlexNet later** |
| `Graph/*.py` | Regenerates the paper's AUC / sensitivity figures from `all_models_results.csv` | Reference only |
| `misc/scripts/download_data.py` | **BROKEN** — downloads a *thyroid* `.hdf5` from a dead personal URL (`berrypidrive.duckdns.org`). Not PAD-UFES-20. | **Do not use** |
| `misc/scripts/check_data.py` | Reads the thyroid `.h5`. Leftover. | Ignore |
| `misc/dataset.py` | Empty stub (`#TODO`) | Ignore |
| `misc/main.py`, `misc/train_vit.py`, `misc/notebook.ipynb` | Earlier scratch versions | Ignore |

**Data preparation** = the top of `run_for_model()` (CODE:`571-611`): read `metadata.csv`, uppercase/strip `diagnostic`, map to binary `is_cancer`, patient-level split, compute `pos_weight` from the training set only.
**Training** = `train_one_epoch()` (CODE:`284-325`) driven by the loop at `701-740`.
**Evaluation** = `evaluate_model()` (CODE:`327-388`) + the test block and threshold calibration (CODE:`743-820`).

---

## 3. Exact ResNet50 configuration (as released in code)

Every row is what the code actually does. Where the paper states something different, see §4.

| Setting | Value in code | Source |
|---|---|---|
| Backbone | `torchvision.models.resnet50`, weights `ResNet50_Weights.IMAGENET1K_V1` | CODE:`192` |
| Preprocessing (eval) | `ResNet50_Weights.IMAGENET1K_V1.transforms()` → resize 256, center-crop 224, ImageNet mean/std | CODE:`150-156, 171` |
| Train augmentation | `RandomHorizontalFlip(0.5)`, `ColorJitter(b=.2,c=.2,s=.2,h=.1)`, `RandomAffine(deg=10, translate=.1, scale=.9–1.1)`, then base transforms, then `RandomErasing(p=.2, scale=.02–.1)` | CODE:`163-170` |
| Classifier head | `Dropout(0.6) → Linear(2048,256) → ReLU → Dropout(0.4) → Linear(256,1)` (single logit) | CODE:`194-200` |
| Loss | `BCEWithLogitsLoss(pos_weight=neg/pos)`, `pos_weight` from **train only** | CODE:`608-611, 640` |
| Optimizer | `AdamW`, two param groups | CODE:`658-661` |
| — backbone LR / weight decay | `LR*0.06 = 6e-6` / `0.1` | CODE:`659` |
| — head LR / weight decay | `LR*0.6 = 6e-5` / `0.01` | CODE:`660` |
| Base `LR` | `1e-4` | CODE:`57` |
| Scheduler | `ReduceLROnPlateau(mode='min', patience=3, factor=0.5)` stepped on **val loss** | CODE:`662, 707` |
| Gradient clipping | `clip_grad_norm_(…, 0.5)` | CODE:`306` |
| Batch size | `16` | CODE:`57` |
| Max epochs | `20` | CODE:`57` |
| Early stopping | on **val AUC**, `PATIENCE=7`, `MIN_DELTA=1e-4` | CODE:`71-72, 723-740` |
| Checkpoint selection | best **val AUC** | CODE:`731-734` |
| Seed | `42` (`random`, `numpy`, `torch`, `cuda`); `cudnn.deterministic=True`, `benchmark=False` | CODE:`57, 77-83` |
| `num_workers` | `0` | CODE:`57` |
| Mixed precision (AMP) | **none** | CODE (absent) |
| Gradient accumulation | **none** | CODE (absent) |

---

## 4. Code-vs-paper discrepancies (the critical section)

These are **confirmed mismatches between the paper's prose and the released CNN code**. They are the reason exact reproduction of Table 1 cannot be promised.

| # | Item | Paper says | Code does | Severity |
|---|---|---|---|---|
| D1 | LR scheduler | "AdamW and a **cosine schedule with warmup**" (§3.4) | `ReduceLROnPlateau` on val loss (CODE:`662`) | **High** — different optimisation trajectory |
| D2 | Mixed precision | "mixed precision (`torch.amp`) with gradient scaling" (§3.4) | none | Medium — can shift numerics/convergence |
| D3 | Gradient accumulation | "we also use gradient accumulation to increase effective batch size" (§3.4) | none; effective batch = 16 | Medium |
| D4 | Early-stopping patience | "patience of **10** epochs" (§3.4, §3.5) | `PATIENCE = 7` (CODE:`71`) | Medium — changes stopping point |
| D5 | CNN preprocessing | "official pretrained data configuration via timm `resolve_data_config` / `create_transform`" (§3.2) | torchvision `Weights.transforms()` (CODE:`150-156`) | Low — numerically ~equivalent for ResNet50 V1, but not literally the same code path |
| D6 | Max epochs | not stated (relies on early stopping) | hard-capped at 20 (CODE:`57`) | Low |

**Interpretation (honest):** either (a) the public code is a cleaned/re-tuned version that differs from the exact code that produced Table 1, or (b) the paper prose describes an intended protocol not fully reflected in the released CNN script. I cannot determine which from the materials provided — **do not assume** the released code reproduces 0.893131 bit-for-bit. Treat §8 accordingly.

---

## 5. Settings that exist *only* in code or *only* in the paper (do not invent the rest)

**Only in code (paper gives no number):** `SEED=42`, `BATCH_SIZE=16`, `EPOCHS=20`, base `LR=1e-4` and the `×0.06`/`×0.6` multipliers, weight decays `0.1`/`0.01`, dropout `0.6`/`0.4`, `ReduceLROnPlateau(patience=3, factor=0.5)`, grad-clip norm `0.5`, augmentation magnitudes.

**Only in paper (no code realisation in the CNN script):** cosine+warmup schedule, AMP, gradient accumulation, patience=10, timm preprocessing for CNNs.

**Genuinely missing / unspecified anywhere — MISSING, not to be guessed:**
- No pinned library versions (`requirements.txt` has no versions). torch/torchvision/numpy/sklearn all float.
- No `DataLoader` generator seed → train-loader shuffle order per epoch is governed only by the global torch RNG; reproducibility across machines is plausible but not guaranteed.
- No saved train/val/test split artifact.
- No config/JSON capturing the run.
- Exact `metadata.csv` row order from the Mendeley release is assumed, not verified here (I did not download the data).

---

## 6. The split — leakage analysis and plan

**What the code does** (CODE:`590-603`):
```
unique_patients = meta["patient_id"].unique()         # order = first appearance after filtering
rng = np.random.RandomState(SEED)                      # SEED = 42 (legacy RandomState)
rng.shuffle(unique_patients)
n_train = int(0.70*n); n_val = int(0.15*n); test = remainder
```
- **Leakage: SAFE at patient level.** Split keys on `patient_id`, so all lesions and all images of a patient land in exactly one split. **[confirmed]** This satisfies "no patient across splits."
- **Model selection: SAFE.** Checkpoint and early stopping use **val** AUC; the test set is only touched in the final block. The screening threshold is chosen on **val** and applied to test (§7). The one test-set threshold computed in code is explicitly labelled post-hoc (CODE:`810-812`). **[confirmed]**
- **Determinism: conditional.** `np.random.RandomState(42).shuffle` is frozen across NumPy versions, so the shuffle is stable. The *input order* to the shuffle is `meta["patient_id"].unique()`, which depends on `metadata.csv` row order after filtering to the 6 valid diagnostics. Since all six classes are exactly the dataset's classes, filtering removes no patients → expected ~**961 train / 205 val / 207 test patients** (`int(0.7·1373)`, `int(0.15·1373)`, remainder). **[strong inference]** — confirm after download.

**Plan (do this once, then freeze):**
1. Run the code's split logic exactly as written, with the official Mendeley `metadata.csv` and a pinned NumPy.
2. **Immediately serialise it** to `splits/padufes20_patient_split.csv` with columns `patient_id, split`. From then on, load splits from this file rather than regenerating — this removes the NumPy/row-order dependency for every downstream run (ResNet50, VGG16, AlexNet) and guarantees all three models share one split.
3. Assert disjointness at load time: `set(train_patients) ∩ set(val) ∩ set(test) == ∅`, and verify image counts sum to 2,298.
4. "Reuse the original split if available": there is **no split file in the repo**, so the "original" split only exists as this deterministic recipe. Reproduce it once (step 1), freeze it (step 2), and treat the frozen file as canonical.

---

## 7. Evaluation metrics — definitions and reproduction targets

**Reproduction target = Table 1 (unambiguous):**

| Metric | ResNet50 target | Source |
|---|---|---|
| AUC | **0.893131** | PAPER Table 1 + `all_models_comprehensive_table.csv` **[confirmed]** |
| F1max | 0.816568 | PAPER Table 1 **[confirmed]** |
| Precision (@F1max) | 0.851852 | PAPER Table 1 **[confirmed]** |
| Recall (@F1max) | 0.784091 | PAPER Table 1 **[confirmed]** |
| Threshold (@F1max) | 0.714025 | `all_models_comprehensive_table.csv` **[confirmed]** |

**How each is computed in code:**
- **AUC** = `roc_auc_score(y_true, continuous_scores)` on the test set (CODE:`739`). Threshold-free. This is the clean primary target.
- **F1max** = sweep PR-curve thresholds, `F1 = 2PR/(P+R)`, take the max; report its precision/recall/threshold (CODE:`539-544`). Paper §3.6 describes the same, phrased as a sweep over [0,1].
- **Sensitivity @ 80% specificity** — **ambiguous, handle with care.** Code produces two numbers:
  1. *Official* (per docstring): choose the threshold on **val** that yields ~0.80 specificity, apply to **test**; report test sensitivity/specificity at that fixed threshold (CODE:`533-537, 790-806`).
  2. *Post-hoc*: choose the threshold on **test** to force 0.80 specificity (CODE:`810-812`, explicitly commented as post-hoc).
  The released `all_models_results.csv` lists ResNet50 as sensitivity **0.8409** / specificity **0.7592** — i.e. definition (1): on test the specificity drifted to 0.759 because the threshold was fixed from val. But **Fig. 3 reports ResNet50 "sensitivity at 80% specificity" = 81.2%**, which matches neither 0.8409 at 0.759 spec nor a clean 0.80 point. **This is an unresolved inconsistency between the repo CSV and the paper figure; I cannot reconcile it without rerunning.** Recommendation: reproduce and report **both** definitions explicitly, and treat AUC/F1max (Table 1) as the authoritative comparison metrics.

---

## 8. How to compare your AUC = 0.893131 *fairly*

The number 0.893131 is a **point estimate from one patient-level split with no confidence interval** (PAPER Table 1; no CI reported). Fair comparison therefore requires controlling what you can and quantifying what you can't:

1. **Same split.** Use the frozen split from §6 for your run. If your split differs from theirs (possible, per §6), AUC differences of ±0.01–0.02 are expected from split variance alone and are **not** evidence of a reproduction failure.
2. **Same metric definition.** Compare your *continuous test AUC* to their 0.893131 — same estimator (`roc_auc_score`), test set, lesion-level (PAPER §3.1: "Metrics are reported on the held-out test set at the lesion level").
3. **Quantify uncertainty they omitted.** Report a **bootstrap 95% CI** for your test AUC (resample test lesions, e.g. 2,000 iterations). If 0.893131 falls inside your CI, that is a successful reproduction in the only statistically honest sense available. A bare "mine = 0.8xx vs theirs = 0.893" is not interpretable without this.
4. **Decide the target explicitly (because of §4).** Pick one and state it in your report:
   - **Target A — reproduce the public code**: run `train_cnn_final.py` as-is. Achievable and deterministic. Expect to land *near* 0.893 but not necessarily on it (split + unpinned libs + the D1–D4 mismatches mean the released code may not be the exact Table-1 code).
   - **Target B — reproduce the paper's described protocol**: modify the code to add cosine+warmup, AMP, grad accumulation, patience=10 (fix D1–D4). This tests the paper as written, not the released script.
   Do **not** silently mix them. Target A is the right first step; your brief says "reproduce the ResNet50 baseline from the reference paper," and the only runnable baseline is the released code.
5. **Seed sensitivity.** A single seed gives one draw. If feasible, run 3–5 seeds and report mean ± std; this separates "reproduced" from "got lucky/unlucky on seed 42."

---

## 9. Recommended reproducible project structure (PyTorch / Colab)

Keep the original repo and data **read-only** as reference; do all work in a parallel tree.

```
skin-cancer-phase1/
├── reference/                      # READ-ONLY copies — never edit
│   ├── Skin-NeuralNET-main/        # original repo
│   └── paper/2605.26294.pdf
├── data/                           # PAD-UFES-20 (not in git; download per §5)
│   ├── metadata.csv
│   ├── imgs_part_1/imgs_part_1/*.png
│   ├── imgs_part_2/imgs_part_2/*.png
│   └── imgs_part_3/imgs_part_3/*.png
├── splits/
│   └── padufes20_patient_split.csv # frozen once (§6), reused by all models
├── configs/
│   └── resnet50_baseline.yaml      # every hyperparameter from §3, version-controlled
├── src/
│   ├── data.py                     # dataset + split loader (loads frozen split)
│   ├── train.py                    # adapted from train_cnn_final.py, config-driven
│   ├── evaluate.py                 # AUC + F1max + sens@80 (both defs) + bootstrap CI
│   └── repro.py                    # seeding, env capture
├── results/
│   └── resnet50/                   # checkpoints, metrics.csv, curves, test_results.json
├── logs/
│   └── resnet50_<timestamp>.log
├── env/
│   ├── requirements.txt            # PINNED versions
│   └── env_snapshot.txt            # pip freeze + torch.__version__ + CUDA at run time
└── notebooks/
    └── colab_resnet50.ipynb        # mounts Drive, runs train.py, saves to results/
```

**Colab notes:** T4/L4 GPU is sufficient for ResNet50 at batch 16. Mount Google Drive for `data/`, `splits/`, `results/` so nothing is lost on disconnect. Set `num_workers=2` on Colab (the repo's `0` is a macOS workaround). Pin versions in the first cell and write `pip freeze` into `env/env_snapshot.txt` for every run.

---

## 10. ResNet50 (exact reproduction) vs VGG16/AlexNet (your extension)

**Exact reproduction (Dey et al.):** only ResNet50, EfficientNetB3, InceptionV3, DenseNet121 exist in the paper and in `build_model()` (CODE:`176-230`). Your in-scope target is **ResNet50 only** for Phase 1.

**VGG16 and AlexNet are NOT in the paper and NOT in the code.** They are a novel extension. Minimum code changes required (flag now, implement in Phase 2 — not now):
- `get_transforms()` (CODE:`150-155`) and `build_model()` (CODE:`176-228`) both key off a dict and will raise `KeyError` for `vgg16`/`alexnet`. Add entries.
- **Head attachment differs.** VGG16 and AlexNet both expose `.features` + `.classifier`; the final layer is `model.classifier[6]` (a `Linear(4096,1000)`), not `.fc`. To keep the paper's *shared head* (dropout→FC256→ReLU→dropout→FC1) comparable, decide and document: replace only `classifier[6]` while keeping the 4096-unit FC stack, **or** replace the whole `.classifier` with the shared head on top of the flattened `.features` output (25088 for VGG16, 9216 for AlexNet). These are different models — pick one and justify it.
- **Param-group logic caveat** (CODE:`643-656`): it branches on `hasattr(model,"features")` first. VGG16/AlexNet have `.features`, so they'd be routed through the DenseNet branch (backbone=`features`, head=`classifier`). That happens to be correct for them, but verify after you edit the head.
- Keep **everything else identical** to the ResNet50 config (same split, loss, optimizer scheme, metrics, seed) so the comparison isolates the backbone — this is exactly the paper's stated design (PAPER §3.3: "Using the same head enables a controlled comparison of the feature extractors").

---

## 11. Implementation checklist (Phase 1)

**Prerequisites / environment**
- [ ] Download PAD-UFES-20 manually from Mendeley `zr7vgbcyr2/1` (the repo's `download_data.py` is the wrong, broken script — §2).
- [ ] Arrange images into the doubled-nested layout `data/imgs_part_N/imgs_part_N/*.png` and place `metadata.csv` at `data/` (§1).
- [ ] Verify integrity: 2,298 `.png`, 1,641 lesions, 1,373 patients, 6 diagnostic values present.
- [ ] Pin dependencies (torch, torchvision, numpy, scikit-learn, pandas) and record `pip freeze` (§5).

**Split**
- [ ] Generate the patient-level 70/15/15 split with `SEED=42` using the code's recipe (§6).
- [ ] Freeze to `splits/padufes20_patient_split.csv`; assert train/val/test patient sets are disjoint; assert image counts sum to 2,298.
- [ ] Confirm the ~961/205/207 patient counts (adjust the doc if the real data differs).

**Baseline run (Target A — public code)**
- [ ] Run ResNet50 via `train_cnn_final.py` logic, loading the frozen split, on Colab GPU.
- [ ] Capture per-epoch `metrics.csv`, best-val-AUC checkpoint, and the training curves.
- [ ] Compute test **AUC (continuous)**, **F1max** + its precision/recall/threshold.
- [ ] Compute sensitivity @ 80% specificity under **both** definitions (§7) and label them.
- [ ] Compute a **bootstrap 95% CI** for test AUC (§8).

**Fair-comparison reporting**
- [ ] Table: your ResNet50 vs Dey et al. Table 1 (AUC 0.893131, F1max 0.816568, P 0.851852, R 0.784091, thr 0.714025).
- [ ] State explicitly whether your split equals theirs or is a re-derivation (and the variance implication).
- [ ] State that you are reproducing Target A (public code), and list the D1–D4 mismatches as known deviations from the paper prose.
- [ ] (Optional but recommended) 3–5 seeds → mean ± std.

**Guardrails (do not violate)**
- [ ] No simulated/placeholder numbers anywhere.
- [ ] Test set never used for threshold selection or tuning (val only).
- [ ] No patient across splits.
- [ ] Original repo + data kept as read-only reference copies.
- [ ] VGG16 / AlexNet / multiclass / mobile app **not started** — Phase 2.

---

## 12. Missing prerequisites — summary (nothing invented)

1. **The dataset itself** — must be downloaded manually; the repo cannot fetch it.
2. **Pinned library versions** — absent; required for split and weight-enum reproducibility.
3. **A frozen split file** — absent; must be generated and saved.
4. **A config file capturing the run** — absent; recommended.
5. **Resolution of D1–D4** (§4) — you must *decide* Target A vs Target B; the materials don't tell you which produced 0.893131.
6. **Resolution of the sensitivity@80 inconsistency** (§7) — unresolvable without rerunning; report both definitions.

**Stop here for review before Phase 2 (implementation), per the brief.**
