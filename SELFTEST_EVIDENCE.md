# Self-test evidence (code correctness, NOT dataset results)

These checks ran in my environment against **synthetic fixtures** (120 fake
patients, 354 fake images with PAD-UFES-20-style columns). They prove the
scripts run and the split logic is sound. They are **not** PAD-UFES-20 results —
real numbers appear only when you run on your data in Colab.

| Check | Where | Result |
|---|---|---|
| All 6 scripts byte-compile | here | PASS |
| `validate_dataset.py` runs, resolves every image, reports counts/balance | here (synthetic) | PASS (exit 0) |
| `make_split.py` freezes split, asserts disjointness | here (synthetic) | PASS, leakage check PASS |
| Split determinism across two fresh runs (file deleted between) | here (synthetic) | **IDENTICAL** |
| Idempotency guard (re-run matches frozen file) | here (synthetic) | MATCHES = True |
| `smoke_test.py` (torch) | **Colab only** | not run here — torch unavailable in my env |

## Bug caught and fixed before Colab
On recent pandas, `meta["patient_id"].unique()` returns a `StringArray`, and
`np.random.RandomState.shuffle` on a non-ndarray raises *"shuffle is not
guaranteed to behave correctly … may contain duplicates"* — a real
nondeterminism/leakage hazard. Fixed in `splits.py` by coercing to a numpy
object array before shuffling. This reproduces exactly what the original code
produced on its (older) pandas and removes the warning. Verified: warning gone,
split identical across fresh runs.

## What still must run on your data (Colab)
- The real validation report (true 1373/1641/2298 counts, real class balance,
  real image-resolution check).
- The real frozen split numbers (≈961/205/207 patients expected).
- The smoke test's forward pass on a real batch.
