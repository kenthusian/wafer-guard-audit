# WAFER-GUARD: audited and repaired wafer anomaly detector

This repo contains a corrected PyTorch supervised-autoencoder pipeline for detecting defective wafers in Fab-7 in-line telemetry (a SECOM-derived dataset).
The original notebook reported F1 ≥ 0.90. That score came from data leakage, a silent tensor-broadcasting bug in the loss, and mishandled legacy probe-station data.
**See [`AUDIT.md`](AUDIT.md) for the findings, the evidence and the honest metrics.**

## Repository layout
| path | purpose |
|---|---|
| `wafer_guard_fixed.ipynb` | repaired, executed pipeline (outputs saved) |
| `AUDIT.md` | findings, repairs, measured metrics, go/no-go |
| `evidence/reproduce_bugs.py` | standalone script reproducing each original bug with numbers |
| `evidence/evidence_output.txt` | output of the script above |
| `bnypi.ipynb` | original notebook, unchanged, for reference |
| `secom_telemetry_modified.csv` | input dataset (1567 wafers × 590 sensors, `label`, `timestamp`) |
| `requirements.txt` | Python dependencies |

`secom_legacy_mask_EVALUATOR_ONLY.csv` is **not** used by the pipeline. Legacy wafers are detected by rule (`sensor_0 == 0.0`). The evidence script reads the mask, if present, only to verify that the rule matches it.

## Prerequisites
- Python 3.10+ (developed on 3.13, Windows 11, CPU only; no GPU needed)
- About 2 GB of disk space for PyTorch

## Installation
```bash
git clone <this-repo-url> && cd <repo>
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
```

## Configuration
All hyperparameters are in the `HP` dict in notebook section 0. They are the same model and optimiser settings as the original, so the comparison isolates the bug fixes. Other settings in that dict:
- `split`: chronological train/val/test fractions
- `max_missing`, `indicator_min_missing`: missing-data policy
- `z_clip`: robust clipping of scaled values

`LEGACY_SENTINEL` and `LEGACY_CODES` (section 3) define the PS-L tool-matching rule.
The input CSV must sit next to the notebook.

## Execution
Interactive: `jupyter notebook wafer_guard_fixed.ipynb`, then *Run All*.

Headless (re-executes and overwrites the saved outputs):
```bash
jupyter nbconvert --to notebook --execute --inplace wafer_guard_fixed.ipynb
```
A full run takes a few minutes on a laptop CPU. It writes `artifacts/` (model weights, fitted preprocessor, `metrics.json`).

## Testing / verification
- **Built-in checks:** the notebook includes assertions that fail the run if any of these break:
  - chronological ordering (no training timestamp at or after a validation/test timestamp)
  - finite features after preprocessing
  - `[B,F]` tensor shapes in `forward()` and the loss
  - a regression cell proving the original `[B,1,F]` broadcast is now rejected
- **Bug reproduction:** `python evidence/reproduce_bugs.py` re-creates the original defects and the leakage ablation (about 2 minutes).
- **Determinism:** seeds are fixed, `torch.use_deterministic_algorithms(True)` is set, and everything runs on CPU, so re-runs reproduce the numbers in `AUDIT.md`.

## Architecture and key decisions
1. **Chronological split, 60/20/20.** The fab drifts, so a random split lets the future train the past.
2. **`CausalPreprocessor`.** Fit on train only and frozen. It drops constant or mostly missing channels, z-scores with train statistics, clips to ±10σ, imputes the train median, and adds missing-indicators.
3. **Legacy PS-L handling.**
   - Stateless and row-wise: `is_legacy` flag plus one-hot firmware codes (100/200/300/unknown).
   - Sentinel and code slots are excluded from all statistics and masked out of the reconstruction loss.
4. **Supervised autoencoder.** Same encoder, decoder and head as the original. The loss is a masked reconstruction MSE plus `BCEWithLogits(pos_weight)`, with explicit shape assertions. SMOTE is removed.
5. **Threshold and evaluation.**
   - Early stopping and the decision threshold use the validation window only.
   - The test window is scored once at its natural prevalence. PR-AUC is primary, with bootstrap CIs.
   - Robustness: 5 seeds, a logistic-regression baseline, and a 5-fold walk-forward backtest.

## Limitations
- Only about 104 defects in total; validation and test windows hold 11 and 17. Metrics have wide confidence intervals.
- The defect rate drifts from 1 % to 15 % across time, so no causal test window sits at exactly 6.6 %. `AUDIT.md` also re-expresses precision and F1 at 6.64 % analytically.
- The repaired model is close to random and does not beat a logistic-regression baseline. **It should not be deployed** (see `AUDIT.md`).
- `artifacts/preprocessor.joblib` pickles a class defined in the notebook. Loading it elsewhere requires that class definition.

## External libraries
PyTorch, scikit-learn, pandas, NumPy, joblib. imbalanced-learn is used only to reproduce the original SMOTE bug. Jupyter and nbclient are used to execute the notebook. No pretrained models or external assets.
