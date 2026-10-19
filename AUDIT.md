# AUDIT: WAFER-GUARD excursion detector

**Recommendation: NO-GO.** The F1 of 0.99 reported by the original notebook is caused by 3 silent bugs. After fixing them, the model scores **F1 = 0.136** at the natural defect rate. My best honest attempt at improving it gets **F1 = 0.19**. Both are far below the 0.90 release gate.

Numbers come from `wafer_guard_fixed.ipynb` and `evidence/reproduce_bugs.py` (output saved in `evidence/evidence_output.txt`).

## Q1. The three silent bugs and how I fixed them

### Bug 1: Data leakage (original cell 8)
**What was wrong**
- `SimpleImputer` and `StandardScaler` were fit on all 1567 wafers, including the test wafers and wafers from the future.
- `SMOTE` was run **before** `train_test_split`. It created 1359 fake defects by blending real ones, and those ended up on both sides of the split.
- The split was random, even though the data is a time series and the fab drifts.
- So the test set was 50% defects (reality is ~6.6%), and full of near-copies of training wafers.

**Evidence** (same model and settings each time, only the data handling changes):

| Setup | Test defect rate | F1 @ 0.5 | PR-AUC |
|---|---|---|---|
| Original pipeline | 50% | **0.992** | 1.000 |
| Original leakage, broadcast bug fixed | 50% | 0.987 | 0.999 |
| Split first (random), fit on train only | 6.7% | 0.071 | 0.095 |
| Split by time, fit on train only | 5.4% | 0.100 | 0.194 |

Fixing the leakage alone drops F1 from 0.99 to about 0.1, so the leak basically *is* the reported score. Also, in the original test set, defect wafers were a median distance of 2.3 from the nearest training wafer, while passing wafers were 17.8 away. The model was being graded on copies of its training data.

**Fix**
- Sort by timestamp. Oldest 60% = train, next 20% = validation, newest 20% = test. Asserts check that no training wafer is newer than a val/test wafer.
- `Preprocessor.fit()` only sees training rows: dropped columns, means, stds and medians are all learned there.
- No SMOTE before the split. I use `pos_weight` in the BCE loss instead. (I also tried SMOTE on training rows only inside CV: PR-AUC 0.111 vs 0.116 with class weights, so not better.)
- The threshold is picked on validation (not a fixed 0.5), and the test set is only used once.

### Bug 2: Corrupted loss gradients from broadcasting (original cells 10 and 14)
**What was wrong.** `make_loader` added an extra dimension, so `batch_x` was `[B, 1, F]`. The model flattened it internally, so `reconstructed` was `[B, F]`. Then `(reconstructed - batch_x)` broadcast to **`[64, 64, 590]`**, comparing every reconstruction with every input in the batch. PyTorch gives no error for this.

**Evidence**
- The loss *value* barely changes (1.0144 buggy vs 1.0145 correct), which is why nobody noticed.
- But the gradient points in a completely different direction (cosine similarity **0.13**).
- The buggy gradient is exactly the gradient of "match the batch average" (`torch.allclose` = True). So the decoder was learning to output the average wafer instead of reconstructing the input.

**Fix**
- Removed `unsqueeze(1)`, so data stays `[B, F]`.
- `WaferSAE.forward()` asserts the input is 2D instead of flattening it.
- `sae_loss()` asserts `recon`, `x` and `mask` are all `[B, F]`, and that logits and labels are both `[B]`, before doing any math.
- Tests check that the old shape now fails.

### Bug 3: Invariant blindness, legacy probe station PS-L (original cell 8)
**What was wrong.** 146 wafers (9.3%) have `sensor_0 == 0.0`, which is the PS-L marker. On exactly those wafers, `sensor_1` contains error codes (100: 61 wafers, 200: 35, 300: 50) that never appear anywhere else. The original scaled and mean-imputed both columns like normal readings. (My rule matches the evaluator's mask file on all 146 wafers.)

**Evidence**
- `sensor_1` mean/std: **2495.7 / 79.8** on real readings vs **2280.5 / 675.3** with the codes mixed in.
- `sensor_0`: 3014 / 74 vs 2732 / 881.
- Missing `sensor_1` values were filled with 2280.5, a value no real station would produce.
- After scaling, the real spread of `sensor_1` was squashed about **8.5x**.
- The codes were turned into fake magnitudes (z = -3.23, -3.09, -2.94), as if code 300 meant "3x more" than 100.

**Fix**
- `legacy_view()` adds an `is_legacy` column and one-hot columns for code 100 / 200 / 300 / unknown. It works row by row, so it can't leak.
- `sensor_0` / `sensor_1` on legacy wafers are left out of every mean, std and median.
- Those slots get a 0 placeholder, but the reconstruction mask is 0 there, so the model is never trained to rebuild a fake value.

### Smaller issues I also fixed
- A fixed 0.5 threshold only made sense on the fake 50/50 test set.
- `drop_last=True` was throwing away training wafers.
- The model was saved behind a "PASSED" gate computed on leaked data.
- 146 constant or mostly-missing sensors (decided on train) are dropped, and "was missing" flags are added for 38 sensors.

## Q2. True metrics at natural prevalence

Test window: newest 20% of wafers (2008-10-02 to 10-17), **314 wafers, 17 defects, 5.41% defect rate**, no resampling.

| Model | F1 | Precision | Recall | PR-AUC | ROC-AUC | TN / FP / FN / TP |
|---|---|---|---|---|---|---|
| **Repaired SAE** (original architecture and settings) | **0.136** | 0.085 | 0.353 | 0.070 | 0.539 | 232 / 65 / 11 / 6 |
| Logistic regression baseline (same features) | 0.141 | 0.078 | 0.706 | 0.089 | 0.629 | 156 / 141 / 5 / 12 |
| **Tuned, chosen by CV** (logistic regression, top 60 features) | **0.190** | 0.160 | 0.235 | 0.219 | 0.556 | 276 / 21 / 13 / 4 |
| Random guessing | - | - | - | 0.054 | 0.5 | - |

- **At the dataset-wide 6.64% rate:** the defect rate drifts over time (13-15% at the start, 1-8% later), so no honest time-based test window lands exactly on 6.6%. Using each model's true and false positive rates, I converted the numbers to 6.64%: F1 = **0.159** (repaired SAE) and **0.211** (tuned).
- **Repaired SAE classification report** (defect class): precision 0.0845, recall 0.3529, F1 0.1364, support 17.
- **Seeds:** across 5 seeds the repaired SAE gets PR-AUC 0.085 ± 0.017. The validation threshold (based on only 11 validation defects) didn't carry over in 4 of the 5 seeds (F1 = 0).

**Why PR-AUC.** With ~5% defects, accuracy and ROC-AUC look fine even for weak models. For PR-AUC a random model scores the defect rate, so it's easy to see how much better than random the model really is.

## Tuning round (Part B of the notebook)
The original-size network overfits: the training classification loss drops to 0.002 while validation PR-AUC stays flat. So I tried 14 settings: a smaller SAE with more dropout, logistic regression, and using only the 60 most label-correlated features.

To keep it fair:
- I picked the winner with a 4-fold walk-forward CV using only data from **before** the test window. Each fold refits everything on past data and scores the next block.
- The threshold came from the CV predictions.
- The winner was refit on the whole history and scored on the test set **once**.

CV picked logistic regression (C = 0.1, top 60 features), with CV PR-AUC 0.116. Its test results are in the table above.

The best SAE setting (CV PR-AUC 0.073) got test F1 0.263. It was **not** picked, since choosing it now based on test results would be cheating. With 17 test defects, the difference is 1 wafer anyway.

## Legacy check: did the model learn "legacy = pass"?
All 146 legacy wafers in this data passed, which is probably a coincidence.

- **Yes, it learned it.** The `is_legacy` weight (-1.0) is the **largest weight of any feature** in the chosen model, and 0% of the 27 legacy wafers in the test window were flagged.
- **It doesn't need it.** Removing the legacy columns barely changes the score: CV PR-AUC 0.116 → 0.111, test PR-AUC 0.219 → 0.215.
- **Recommendation:** in production, either drop `is_legacy` from the classifier or monitor PS-L wafers separately. Otherwise a real defect on PS-L would almost never be flagged.

## Risks and recommendation
- **Very few defects:** 104 total, 76 in training, 17 in the test window. All of these numbers are noisy.
- **Strong drift:** the defect rate ranges from 1% to 15% across the period. The model would need regular retraining and monitoring.
- **The "legacy = pass" shortcut** described above.
- **Decision: NO-GO for Friday.** Even the best honest model only catches about 1 in 4 defects at 16% precision. That might be OK as an extra hint for engineers, but not as an automatic release gate.
