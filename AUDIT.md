# WAFER-GUARD pre-release audit

**Recommendation: NO-GO.** The reported F1 ≥ 0.90 is an artifact of three silent defects.
Measured honestly, on a later time window at natural defect prevalence, the repaired supervised autoencoder scores **F1 = 0.136** (precision 0.085, recall 0.353) and **PR-AUC = 0.070**. A random scorer gets PR-AUC 0.054 on that window.
A leakage-safe tuning round raised the best result to **F1 = 0.191, PR-AUC = 0.219**: cross-validation picked logistic regression on 60 train-selected features. That is still far below the 0.90 gate.

All numbers below come from `wafer_guard_fixed.ipynb` (repaired pipeline) and `evidence/reproduce_bugs.py` (bug reproductions; output in `evidence/evidence_output.txt`). Seed is 1337 unless stated, on CPU.

## Answers to the required audit questions
**Q1. What were the three silent bugs, and how were they fixed?**
1. **Data leakage.**
   - *Bug:* the imputer and scaler were fit on all rows, SMOTE ran before the split, and the split was random in time. The model was then graded on a 50 % synthetic test set.
   - *Fix:* a chronological split, with every statistic fit on train only, no pre-split SMOTE, and a test set left at natural prevalence. (Finding 1)
2. **Corrupted loss gradients.**
   - *Bug:* `[B,F] − [B,1,F]` silently broadcast to `[B,B,F]`, so the decoder learned the batch mean.
   - *Fix:* `[B,F]` tensors, shape asserts, and a masked MSE. (Finding 2)
3. **Invariant blindness.**
   - *Bug:* the PS-L sentinel `sensor_0 = 0.0` and the firmware codes in `sensor_1` were scaled and imputed as real readings.
   - *Fix:* an `is_legacy` flag and one-hot codes, with those slots kept out of every statistic and out of the loss. (Finding 3)

**Q2. What are the true metrics at natural prevalence?**

These are on the untouched chronological test window: 314 wafers, 17 defects, 5.41 % prevalence.

| model | F1 | precision | recall | PR-AUC (random = 0.054) |
|---|---|---|---|---|
| Repaired supervised autoencoder (original architecture and HP) | 0.136 | 0.085 | 0.353 | 0.070 |
| After the leakage-safe tuning round (CV-selected logistic regression) | 0.191 | 0.160 | 0.235 | 0.219 |

Re-expressed at the dataset-wide 6.64 % prevalence, F1 is 0.159 and 0.211. **Both fail the 0.90 gate: NO-GO.**

## Findings, ranked by severity

### 1. Data leakage: future and evaluation data inside training (original cell 8)
**Mechanism.**
- `SimpleImputer` and `StandardScaler` were `fit_transform`ed on **all 1567 rows**, so test-set and future wafers set the means and stds.
- `SMOTE` ran **before** `train_test_split`. Its 1359 synthetic defects are interpolations between real defects, and they landed on both sides of the split.
- The split was random, so wafers from later dates trained a model that was tested on earlier dates.
- The test set was **50 % defective**, not the natural ~6.6 %.

**Evidence** (`reproduce_bugs.py`, same model and HP throughout):

| scenario | test prevalence | F1 @ 0.5 | PR-AUC |
|---|---|---|---|
| A. original pipeline as shipped | 50.0 % | **0.992** | 1.000 |
| B. original leakage, broadcast bug fixed | 50.0 % | 0.987 | 0.999 |
| C. random split first, fit on train only, natural test | 6.7 % | 0.071 | 0.095 |
| D. chronological split, fit on train only, natural test | 5.4 % | 0.100 | 0.194 |

- Fixing the leakage alone drops F1 from 0.99 to about 0.1. The leak *is* the reported score.
- Test defects in scenario A sit a median of 2.27 (z-space L2 distance) from a training point; test passes sit 17.8 away. The model was graded on near-copies of its own training data.

**Fix (`FIX[1]`).**
- Sort by `timestamp` and split chronologically: 60 % train, 20 % validation, 20 % test. Asserts check that no training timestamp is later than any validation or test timestamp.
- `CausalPreprocessor.fit()` sees training rows only. Column drops, means, stds, medians and indicator selection are all learned there and then frozen.
- SMOTE is removed. Imbalance is handled with `BCEWithLogits(pos_weight = neg/pos of train)`.
- The threshold is chosen on validation, not fixed at 0.5. The test window is scored once and never resampled.

### 2. Corrupted loss gradients: silent broadcasting (original cells 10 and 14)
**Mechanism.**
- `make_loader` adds a channel axis, so `batch_x` is `[B,1,F]`. The model flattens internally, so `reconstructed` is `[B,F]`.
- `(reconstructed - batch_x)` broadcasts to **`[64, 64, 590]`**: every reconstruction is compared against every input in the batch. No error is raised.

**Evidence.**
- The buggy and correct losses have nearly identical values (1.0144 vs 1.0145), so the loss curve looks normal.
- The gradients differ completely: cosine similarity is **0.133**.
- The buggy gradient is analytically identical to the MSE gradient toward the **batch mean** (`torch.allclose → True`). The decoder was trained to output the average wafer, which destroys the autoencoder's reconstruction signal.

**Fix (`FIX[2]`).**
- Tensors are built as `[N,F]`, and `forward()` rejects any input that is not rank 2.
- `sae_loss()` asserts `recon.shape == x.shape == mask.shape` and `logits.shape == y.shape == (B,)`.
- Reconstruction is a **masked** MSE over genuinely observed cells only.
- A regression cell in the notebook proves that `[B,1,F]` inputs now fail loudly.

### 3. Invariant blindness: legacy Probe Station PS-L (original cell 8)
**Mechanism.**
- 146 wafers (9.3 %) carry the PS-L sentinel `sensor_0 == 0.0`.
- On exactly those rows, `sensor_1` holds firmware codes {100: 61, 200: 35, 300: 50}. The codes never appear elsewhere, and the rule-based detection matches the evaluator mask exactly.
- The original treated both slots as continuous measurements.

**Evidence.**
- `sensor_1` mean and std went from 2495.7 / 79.8 (genuine) to **2280.5 / 675.3**, and `sensor_0` from 3014 / 74 to 2732 / 881.
- Missing `sensor_1` values were mean-imputed with **2280.5**, a reading no station produces.
- After scaling, the genuine 5th–95th percentile range of `sensor_1` spanned only **0.37σ** instead of about 3.3σ. The real signal was compressed **8.5×**.
- The codes became fake ordinal magnitudes (z = −3.23, −3.09, −2.94).

**Fix (`FIX[3]`).**
- A stateless, row-wise `legacy_view()` produces an `is_legacy` flag and one-hot codes (`s1_code_100/200/300/unknown`). An unseen future code goes to `unknown`; it is never treated as a number.
- Sentinel and code slots are marked *not applicable*. They are excluded from every scaling and imputation statistic, which are fit on genuine train readings only (`sensor_1` scaling uses mean 2494.9, std 85.7).
- Their continuous slot gets a fixed placeholder that is masked out of the reconstruction loss. No fabricated measurement is ever used as a target.

### Minor issues, also fixed
- The fixed 0.5 threshold was calibrated to a synthetic 50/50 world. It is now chosen on validation.
- `drop_last=True` silently discarded training wafers.
- Artifacts were saved behind a "PASSED" gate computed on leaked data.
- 146 channels that are constant or more than 50 % missing **in train** are dropped (444 continuous channels remain). Missing-value indicators are added for 38 channels.

## True metrics of the repaired model (natural prevalence, no resampling)
Primary result: **chronological test window** (2008-10-02 → 10-17, n = 314, 17 defects, natural prevalence **5.41 %**). The threshold (0.0194) was frozen from validation. CIs are 95 % bootstrap with 2 000 resamples.

| metric | WaferSAE (repaired) | Logistic-regression baseline |
|---|---|---|
| PR-AUC (primary) | **0.070** [0.039, 0.155] | 0.089 [0.048, 0.175] |
| ROC-AUC | 0.539 [0.392, 0.687] | 0.629 [0.482, 0.767] |
| F1 | **0.136** [0.047, 0.239] | 0.141 [0.075, 0.212] |
| precision / recall | 0.085 / 0.353 | 0.078 / 0.706 |
| confusion (TN/FP/FN/TP) | 232 / 65 / 11 / 6 | 156 / 141 / 5 / 12 |
| precision / F1 re-expressed at π = 6.64 % | 0.103 / 0.159 | 0.096 / 0.168 |

The defect rate drifts from 14 % in the first fifth of the data to 3–5 % later. So no causal test window sits at exactly 6.64 %. The last row converts the measured TPR and FPR to the dataset-wide prevalence using Bayes' rule, with no resampling.

**Robustness checks.**
- **5 seeds:** PR-AUC 0.085 ± 0.017, ROC-AUC 0.590 ± 0.059, F1 0.032 ± 0.071. In 4 of 5 seeds the validation threshold (picked from only 11 validation defects) did not transfer, and F1 = 0.
- **Walk-forward backtest:** 5 expanding-window folds, refitting on the past each time; 784 wafers, 37 defects, prevalence 4.72 %. Results: PR-AUC 0.059, ROC-AUC 0.581, F1 0.082, precision 0.048, recall 0.270.

`classification_report` for the repaired SAE on the test window (threshold 0.0194):

```
              precision    recall  f1-score   support
        pass     0.9547    0.7811    0.8593       297
      defect     0.0845    0.3529    0.1364        17
    accuracy                         0.7580       314
```

**Why PR-AUC.** At about 5 % prevalence, ROC-AUC and accuracy flatter a classifier. PR-AUC has a clear reference point (random = prevalence) and tracks how useful an alarm is to a fab engineer. F1 is reported at a threshold chosen without touching the test data.

## Stage B: leakage-safe tuning (notebook sections 12–14b)
The repaired SAE overfits: training BCE falls to about 0.002 while validation PR-AUC stays flat. So one bounded tuning round was run.

**Protocol.** The grid was fixed in advance, and the test window was not used for selection.
- Selection used a 4-fold expanding-window CV inside the train+val history only (627 scored wafers, 22 defects). Each fold refits the preprocessor, a top-k correlation feature screen and the model on past wafers only.
- The grid had 38 configurations: SAE hidden {64, 128} × latent {8, 16} × weight decay {1e-3, 1e-2} × epochs {10, 25} × features {all, top-60}, with dropout 0.3 and a 3-seed ensemble in CV; plus class-weighted logistic regression with C {1e-3, 1e-2, 1e-1} × features {all, top-60}.
- The threshold comes from the pooled CV predictions. The winner is refit on the full history (with a 5-seed ensemble for an SAE) and scored **once** on the test window.

| Stage B result (test window, 5.41 % prevalence) | PR-AUC | ROC-AUC | F1 | precision / recall | TN/FP/FN/TP |
|---|---|---|---|---|---|
| **CV-selected: logistic regression, C = 0.1, top-60 features** (CV PR-AUC 0.116) | **0.219** [0.057, 0.432] | 0.556 | **0.191** [0.047, 0.348] | 0.160 / 0.235 | 276 / 21 / 13 / 4 |
| Best SAE by CV (64-16, wd 1e-2, 25 epochs, top-60; CV PR-AUC 0.073), reference only | 0.129 [0.057, 0.326] | 0.620 | 0.263 [0.074, 0.444] | 0.238 / 0.294 | 281 / 16 / 12 / 5 |

- Re-expressed at π = 6.64 %, the selected model reaches precision 0.191 and F1 0.211.
- The SAE row is reported for transparency only. It had the higher test F1, but it was **not** selected, and choosing it now would mean selecting on the test set. With 17 test defects, the F1 difference is 1 extra true positive, well inside the CIs.
- **SMOTE sensitivity (grid v2).** A reviewer asked whether SMOTE helps when applied to training rows only, inside each fold. All 38 configurations were re-run with that SMOTE. *These were added after the v1 test result had been seen, so they are a sensitivity check, not the primary result.* The best SMOTE configuration had CV PR-AUC 0.111 (logistic regression) and 0.091 (SAE), against 0.116 without SMOTE. CV selection was **unchanged**, so SMOTE gave no benefit here.

## Interpretation, residual risks, recommendation
- With leakage removed, the repaired SAE is only marginally better than random. A regularised linear model on a small, train-selected feature set is the best honest option found, at about 4× the random PR-AUC. That is still far too weak for automatic dispositioning. The 940 training wafers contain only 76 defects.
- **Residual risks:**
  - Strong label and covariate drift over time (prevalence 1–15 % per decile).
  - Tiny validation and test defect counts (11 and 17), which make threshold choice unstable and CIs wide.
  - Legacy PS-L wafers show 0 defects in this sample. The model can learn "legacy ⇒ pass", which is a coincidence of this dataset, not physics.
  - The preprocessor pickle depends on the notebook's class definition.
  - Weight decay had almost no effect in the grid (pairs differing only in weight decay score identically to 3 decimals). That dimension effectively wasn't searched.
- **Decision: NO-GO for Friday.** Possible next steps: more labelled history; per-tool monitoring; feature selection or a simpler model with proper time-series CV; calibrated alerting at an agreed precision; and a drift monitor on prevalence and on the legacy-code mix.
