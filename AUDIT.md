# WAFER-GUARD pre-release audit

**Recommendation: NO-GO.** The reported F1 ≥ 0.90 is an artifact of three silent defects.
Measured honestly, on a later time window at natural defect prevalence, the repaired model scores **F1 = 0.136 and PR-AUC = 0.070**. A random scorer gets PR-AUC 0.054 on that window.

All numbers below come from `wafer_guard_fixed.ipynb` (repaired pipeline) and `evidence/reproduce_bugs.py` (bug reproductions; output in `evidence/evidence_output.txt`). Seed is 1337 unless stated, on CPU.

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

**Why PR-AUC.** At about 5 % prevalence, ROC-AUC and accuracy flatter a classifier. PR-AUC has a clear reference point (random = prevalence) and tracks how useful an alarm is to a fab engineer. F1 is reported at a threshold chosen without touching the test data.

## Interpretation, residual risks, recommendation
- Once leakage is removed, the model is only marginally better than random. It does **not** beat a regularised logistic regression. The 940 training wafers contain only 76 defects, and the training classification loss collapses to about 0.002 by epoch 10 while validation PR-AUC stays flat. The network is overfitting badly.
- **Residual risks:**
  - Strong label and covariate drift over time (prevalence 1–15 % per decile).
  - Tiny validation and test defect counts (11 and 17), which make threshold choice unstable and CIs wide.
  - Legacy PS-L wafers show 0 defects in this sample. The model can learn "legacy ⇒ pass", which is a coincidence of this dataset, not physics.
  - The preprocessor pickle depends on the notebook's class definition.
- **Decision: NO-GO for Friday.** Possible next steps: more labelled history; per-tool monitoring; feature selection or a simpler model with proper time-series CV; calibrated alerting at an agreed precision; and a drift monitor on prevalence and on the legacy-code mix.
