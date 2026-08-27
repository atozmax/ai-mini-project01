# Credit Card Fraud Detection Pipeline

**Course:** Maktab 145 — First Machine Learning Project  
**Method:** Heterogeneous stacked generalization  
**Final model:** Experiment 06 — PyTorch meta-learner on mixed ML + DL bases  
**Primary metric:** Fraud-class F1 (precision and recall reported together)

---

## Slide 1 — One-sentence pitch

This is a **credit-card fraud detector**. Each row is a transaction. The model answers: **legitimate or fraud?**

We did not pick one algorithm. We built **stacked generalization**: several different models vote, then a second model learns how to combine those votes.

---

## Slide 2 — The problem and the data

Dataset: Kaggle ULB Credit Card Fraud Detection.

| | |
|---|---|
| Rows | 284,807 transactions |
| Frauds | 492 (**0.17%**) |
| Legitimate | ~99.83% |
| Features | `Time`, `Amount`, PCA components `V1`–`V28` |
| Target | `Class` — 0 = Legitimate, 1 = Fraud |

The V columns are already scaled. Time and Amount are not.

**Split used for the sklearn track (Experiments 01–04 and 06):**

```
train_test_split(..., test_size=0.2, random_state=42, stratify=y)
```

| Split | Rows | Frauds |
| ----- | ---- | ------ |
| Train | 227,845 | 394 |
| Test | 56,962 | 98 |

---

## Slide 3 — Why accuracy is the wrong headline

A model that always predicts Legitimate is about **99.8% accurate** and catches **zero** fraud.

| Metric | What it answers in this project |
| ------ | ------------------------------- |
| **Recall** | Of the real frauds, how many did we catch? Missed fraud is expensive. |
| **Precision** | Of the alerts we raise, how many are real? Too many false alarms is also expensive. |
| **F1** | Harmonic mean of precision and recall. It only stays high if **both** stay high. |

The operating point we wanted, and later obtained, is high fraud F1 **without** collapsing precision.

---

## Slide 4 — What the method is called

The approach is **stacked generalization** (stacking), specifically a **heterogeneous ensemble**.

- **Ensemble learning** means combining several models instead of trusting one.
- **Stacking** (Wolpert, 1992) trains a second model — the **meta-learner** — on the *outputs* of the first models, not on the raw transactions.
- **Heterogeneous** means the first-level models are not copies of the same algorithm. They differ in:
  - family (trees, nearest neighbors, random forests, neural nets)
  - training recipe (class weights, SMOTE, undersampling)
  - objective (precision, recall, or F1)

This is *not* bagging (same model, different samples) and *not* boosting (sequential residual fitting).

**References**

- David H. Wolpert, “Stacked Generalization,” *Neural Networks*, 5(2), 1992.
- Leo Breiman, “Stacked Regressions,” *Machine Learning*, 24, 1996.
- Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle / ULB).
- Design reference: Janio Martinez Bachmann, [Credit Fraud \|\| Dealing with Imbalanced Datasets](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets). The kernel is **not** copied as the final solution.

---

## Slide 5 — How the final model works

```
Transaction (Time, V1–V28, Amount)
        │
        ├─ Decision Tree      → P(fraud)
        ├─ KNN                → P(fraud)
        ├─ Random Forest      → P(fraud)
        ├─ RF meta (Exp 04)   → P(fraud)
        └─ 6 Keras nets
           (F1 / recall / precision × train / val)
                │
                ▼
        10 probabilities
                │
                ▼
        Small PyTorch net (Exp 06)
                │
                ▼
        If P(fraud) ≥ 0.33 → Fraud
        else               → Legitimate
```

Some neural nets look “too recall-heavy” and others “too precision-heavy.” That is intentional.

- **Recall nets** catch almost everything and raise many false alarms.
- **Precision nets** are conservative.
- The PyTorch stacker trusts a recall net **only when** the precise models also agree.

---

## Slide 6 — Headline numbers

Sklearn 80/20 split, **98 frauds** on test, threshold **0.33** (chosen on **train** F1):

| | Precision | Recall | F1 | TP | FN | FP |
|---|-----------|--------|----|----|----|----|
| **Exp 06 PyTorch stack** | **0.91** | **0.95** | **0.93** | **93** | **5** | **9** |

Relative to the best sklearn stacker (F1 ≈ 0.86–0.87):

- about **12 extra frauds caught** (81 → 93)
- false positives stay in the same band (8–9)
- fraud F1 moves from ~0.86 to **0.93**

**If someone asks “is 0.93 fully honest?”**  
Partly optimistic, because Experiment 05 used a different split (see leakage section). For a fully disjoint neural-net holdout, quote Experiment 05: `nn_precision_train` ≈ precision **0.96**, recall **0.70**, F1 **0.81**.

---

## Slide 7 — End-to-end picture

```
Raw Kaggle CSV
    │
    ├─ Experiment 01  baselines (DT, KNN, Logistic)
    ├─ Experiment 02  OOF stack: DT + KNN + RF → RF meta
    ├─ Experiment 03  threshold study (F2, then F1)
    ├─ Experiment 04  class-weighted tree, F1 cutoff
    │
    ├─ Experiment 05  (Kaggle recipe + deep specialists)
    │       RobustScaler(Time, Amount)
    │       undersample 50/50  +  SMOTE on train
    │       six Keras nets: F1 / recall / precision
    │
    └─ Experiment 06  heterogeneous stack
            3 ML probabilities
          + RF meta probability
          + 6 DL probabilities
          → PyTorch meta-network
          → F1-tuned threshold
```

---

## Slide 8 — Experiment 01 — baselines

**Notebook:** `notebooks/experiment01.ipynb`

Goal: a first honest pipeline on the imbalanced data.

- Load the public credit-card file.
- Stratified 80/20 split.
- `Pipeline(StandardScaler → model)` so scaling is fit on train only.
- Models: **Decision Tree**, **KNN**, **Logistic Regression**.
- Search scored with **F1**, not accuracy.

**What to say.** Logistic regression is a weak baseline on this nonlinear PCA space. Trees and KNN are more plausible. Accuracy looks excellent for every model and must be ignored. This experiment established the split, the metric, and the “scaler inside the pipeline” rule.

---

## Slide 9 — Experiment 02 — classical stacking

**Notebook:** `notebooks/experiment02.ipynb`

Goal: combine complementary sklearn models.

| Model | Role |
| ----- | ---- |
| Decision Tree | Interpretable nonlinear rules |
| KNN (`n_neighbors=3`) | Local similarity |
| Random Forest | Already `class_weight="balanced_subsample"` |

Stacking procedure:

1. Out-of-fold fraud probabilities on train → columns `pred_dt`, `pred_knn`, `pred_rf`.
2. Full-train refit probabilities on test (no leakage into the test labels).
3. A **Random Forest meta-model** trained only on those three scores.

**What to say.** Do **not** train the meta-model on in-sample scores. Those scores are too optimistic and leak the label. A small tree ensemble can learn “alert if KNN and RF both fire, even if the DT is quiet.”

---

## Slide 10 — Experiment 03 — thresholds

**Notebook:** `notebooks/experiment03.ipynb`

Goal: treat `0.5` as a default, not a law. The assignment asks for `0.3 / 0.5 / 0.7`.

| Rule | Threshold | Precision | Recall | F1 | TP / FN / FP |
| ---- | --------- | --------- | ------ | -- | ------------ |
| F2 (recall-heavy) | 0.25 | 0.83 | 0.87 | 0.85 | 85 / 13 / 17 |
| **F1 (balanced)** | **0.40** | **0.90** | **0.83** | **0.86** | **81 / 17 / 9** |
| Required | 0.50 | 0.91 | 0.83 | 0.87 | 81 / 17 / 8 |
| Required | 0.70 | 0.97 | 0.74 | 0.84 | 73 / 25 / 2 |

Pushing recall toward 95% by lowering the cutoff **failed**: even at `t=0.01` train recall only reached about 92%, and test precision collapsed.

**What to say.** Threshold tuning is part of the model. F1 (or F2) must be chosen on train/validation, then frozen. Stacking cannot invent a signal that no base model has.

---

## Slide 11 — Experiment 04 — F1-oriented sklearn stack

**Notebook:** `notebooks/experiment04.ipynb`

Goal: improve fraud F1 without a recall quota.

- Retrain the tree with `class_weight="balanced"`.
- Rebuild only `pred_dt` (OOF). Keep KNN and RF scores.
- Meta RF **without** class weights (weights had destroyed precision).
- Pick the highest train **F1** cutoff.

| Setting | Threshold | Precision | Recall | F1 |
| ------- | --------- | --------- | ------ | -- |
| Exp 03 stacker, F2 | 0.25 | 0.83 | 0.87 | 0.85 |
| Exp 03 stacker, F1 | 0.40 | 0.90 | 0.83 | 0.86 |
| Exp 04 new DT + meta | 0.30 | 0.86 | 0.85 | 0.85 |

**What to say.** The balanced tree raised OOF recall but, used alone at 0.5, precision was unusable (~0.04). The meta-model recovered precision. It did **not** beat the original stacker on F1. Practical sklearn operating point: **original stacker at 0.40–0.50**.

Skipped on purpose: 50/50 undersampling, dropping fraud outliers, and SMOTE (not in `requirements.txt` at that stage).

---

## Slide 12 — Experiment 05 — deep specialists

**Notebook:** `notebooks/experiment05/experiment05.ipynb`

Goal: add neural nets that *specialize*, instead of one net that compromises.

### Data recipe (from the Kaggle kernel)

1. Scale **only Time and Amount** with `RobustScaler`. Leave `V1`–`V28` as they are.
2. Hold out an original **imbalanced** test fold (stratified 5-fold, last fold).
3. From the remaining train data, hold a further imbalanced **validation** split.
4. Show a 50/50 **random undersample** (too small for a deep net; not used for the six models).
5. **SMOTE** the fit split only. Never resample the test set.

### Six independent Keras runs

Same architecture: `256 → 128 → 64 → 32 → 1` (ReLU, batch norm, dropout, sigmoid).  
Each run saved the epoch that maximized **one** metric on **one** split:

| File | Optimized for |
| ---- | ------------- |
| `nn_f1_train.keras` | train F1 |
| `nn_f1_val.keras` | validation F1 |
| `nn_recall_train.keras` | train recall |
| `nn_recall_val.keras` | validation recall |
| `nn_precision_train.keras` | train precision |
| `nn_precision_val.keras` | validation precision |

Precision selection required recall ≥ 0.40 so a model could not “win” by predicting almost no fraud.

### Honest holdout (Experiment 05’s own test fold, 98 frauds)

| Model | Threshold | Precision | Recall | F1 | TP / FN / FP |
| ----- | --------- | --------- | ------ | -- | ------------ |
| `nn_precision_train` | 0.90 | **0.96** | 0.70 | **0.81** | 69 / 29 / 3 |
| `nn_f1_val` | 0.91 | 0.93 | 0.68 | 0.79 | 67 / 31 / 5 |
| `nn_recall_train` | 0.05 | 0.27 | **0.83** | 0.41 | 81 / 17 / 217 |
| `nn_recall_val` | 0.05 | 0.25 | **0.83** | 0.39 | 81 / 17 / 239 |

**What to say.** Precision specialists are conservative and precise. Recall specialists catch more fraud and drown the operator in false positives. **That diversity is useful only if a later model can choose among them.**

---

## Slide 13 — Experiment 06 — heterogeneous stack (final)

**Notebook:** `notebooks/experiment06.ipynb`

Goal: one combiner that sees *all* of the previous work.

### Meta-features (10 scores)

From Experiments 02–04 (OOF on the sklearn split):

- `pred_dt` — balanced decision tree
- `pred_knn` — KNN
- `pred_rf` — random forest
- `pred_meta_f1` — Experiment 04 Random Forest stacker

From Experiment 05 (frozen Keras weights, numpy inference, no TensorFlow required):

- `nn_f1_train`, `nn_f1_val`
- `nn_recall_train`, `nn_recall_val`
- `nn_precision_train`, `nn_precision_val`

### Meta-learner

```
Linear(10 → 64) → BatchNorm → ReLU → Dropout(0.3)
Linear(64 → 32) → ReLU → Dropout(0.2)
Linear(32 → 1)  → sigmoid  (BCE with logits)
```

- No class weights (they pulled the combiner toward the recall nets and hurt precision).
- 20% stratified validation inside the sklearn **train** fold.
- Early stopping on validation loss.
- Final cutoff chosen by **train fraud F1** (`t = 0.33`).

A linear SVM was tried first as the meta-learner. With `class_weight="balanced"` it copied the recall nets and precision fell. Without class weights it followed the precision nets. The PyTorch net could use **both** families at once, which is the point of stacking specialists.

---

## Slide 14 — Why mixing precision and recall models works

Each base learner fails differently.

| Specialist | Typical behavior | Failure mode |
| ---------- | ---------------- | ------------ |
| Precision / F1 nets, KNN | Alert only when sure | Misses unusual frauds |
| Recall nets, balanced tree | Alert on weak evidence | Many legitimate customers blocked |
| Random forest | Strong middle ground | Still misses the same hard tail |

The meta-learner sees a 10-dimensional vote. Empirically:

- If precision nets and KNN all fire, the transaction is almost certainly fraud.
- If only a recall net fires, it is usually a false alarm.
- If recall nets fire **and** the RF/KNN scores are moderate, it is often a hard fraud worth catching.

That pattern cannot be expressed by a single threshold on a single model. It *can* be expressed by a second model trained on the votes. That is stacked generalization.

---

## Slide 15 — What every file in `src` does

There are only three files. Together they are: **prepare → train/export → predict**.

### `src/data_prep.py` — data contract

This file does **not** train a model. It defines how the CSV is loaded and how features are shaped.

- Finds `creditcard.csv` in `data/` or `_data/`.
- `FEATURE_COLUMNS` = `Time` + `V1`…`V28` + `Amount`. Target is `Class`.
- `load_transactions()` / `dataset_report()`: load, check columns, print size, class counts, missing values, duplicates, describe stats.
- `split_xy()` then `stratified_split()`: **80/20, `random_state=42`, `stratify=y`**. Same split as the notebooks.
- `fit_standard_scaler(X_train)`: StandardScaler **on train only** (sklearn track).
- `fit_time_amount_scalers(frame)`: `RobustScaler` on Time and Amount for the neural nets. **Must be called on train**, not the full CSV.
- `nn_feature_matrix()`: `[scaled Amount, scaled Time, V1–V28]` — the input layout the Keras nets expect.
- `records_to_frame()`: turns a JSON transaction (or a list) into that same feature table for inference.

If you run this file as a script, it only reports the dataset and proves the scaler was fit on train.

### `src/train.py` — assemble the final model

This file **does not retrain** the search-heavy models from scratch. It **loads** the notebook artifacts, reports the assignment metrics, and writes a deployable bundle into `models/`.

What it does, in order:

1. Load data, print the dataset report, do the same stratified split.
2. Train two **baselines** on train only: Logistic Regression and a Decision Tree (both in a `Pipeline` with `StandardScaler`). Print 5-fold CV on **train**, then test reports, plus logistic at thresholds **0.3 / 0.5 / 0.7**.
3. Load saved bases: balanced DT, KNN, RF, Exp 04 RF stacker, six Keras `.keras` files, PyTorch stacker checkpoint.
4. `load_keras_net()`: reads Keras weights with `h5py` so you do **not** need TensorFlow at inference. Forward pass is numpy: Dense → ReLU → BatchNorm × 3, then sigmoid.
5. `FraudStack`: for a new table of transactions, collect the 10 scores, run the PyTorch net, cut at 0.33.
6. `save_bundle()`: writes `models/model.pkl`, `stacker.pt`, `scaler.pkl`, `amount_scaler.pkl`, `time_scaler.pkl`, and copies the six nets into `models/nn/`.
7. Smoke-test three test rows through the saved bundle.

`StackerNet` is the small combiner: `10 → 64 → 32 → 1` with dropout.

### `src/predict.py` — the thing you demo

CLI scorer for new transactions.

```bash
python src/predict.py data/input.json -o data/output.json
```

It reads JSON (one object, a list, JSONL, or `{"transactions": [...]}`), converts it with `records_to_frame`, loads `FraudStack` from `models/`, and writes:

```json
{
  "prediction": "Legitimate" or "Fraud",
  "class_id": 0 or 1,
  "probability": 0.0002,
  "threshold": 0.33,
  "status": "success"
}
```

Errors become `{"status": "error", "message": "..."}`. This is the production-style piece the assignment asks for.

---

## Slide 16 — Data leakage audit

Answer in three layers: **clean**, **fixed**, **caveat**.

### What is done correctly (say this first)

- **Split before fitting sklearn models.** Same `train_test_split(..., stratify=y, random_state=42)` everywhere on that track.
- **Scaling inside `Pipeline`** for DT / KNN / RF / Logistic. The scaler never sees test rows.
- **SMOTE only on Experiment 05’s fit split.** Validation and test stay imbalanced and real.
- **Stacking for DT / KNN / RF uses out-of-fold probabilities** (`cross_val_predict`, 5-fold). The meta RF is not trained on in-sample base scores.
- **Test set is not used to pick the 0.33 threshold.** That cutoff was chosen on train F1, then locked.
- **Target `Class` is never a feature.** Inference JSON only has Time, V1–V28, Amount.

### What was fixed in `src` before this presentation

1. **`fit_time_amount_scalers` was called on the full dataframe** in `train.py`. Test Time/Amount statistics (median and IQR) were leaking into the RobustScalers. It now fits on **`X_train` only**.
2. **`print(__name__)` ran on every import of `data_prep.py`.** That is gone. It would have printed during a live demo.

Re-run `python src/train.py` if you want `models/amount_scaler.pkl` and `models/time_scaler.pkl` rewritten with the train-only fit. Numerically the change is tiny (hundreds of thousands of rows); the **principle** is what examiners care about.

### Caveats to say out loud (do not hide these)

**1. Experiment 05 and Experiments 01–04 do not share the same test rows.**  
Exp 05 uses `StratifiedKFold(n_splits=5, shuffle=False)` and takes the **last fold** as test (time-ordered). Exp 01–04/06 use a **shuffled** 80/20 split. Many sklearn-test rows were inside the Keras training set. So the six `nn_*` columns on the sklearn test set are **optimistic**.

If they ask “so is F1 0.93 overstated?”: **yes, a bit, because of the neural-net columns.** The sklearn votes (`pred_dt`, `pred_knn`, `pred_rf`) are clean. For a fully disjoint net number, quote Exp 05: `nn_precision_train` ≈ precision **0.96**, recall **0.70**, F1 **0.81**.

**2. Experiment 05 fitted `RobustScaler` on the whole CSV, then split.** Copied from the Kaggle kernel. Mild leakage (test median/IQR in the scaler), not label leakage. The `src` path no longer does this.

**3. `pred_meta_f1` on the train fold is in-sample for the Exp 04 RF stacker.** DT/KNN/RF scores on train are OOF; the meta-RF score on those same train rows is not. That can make the PyTorch net slightly over-trust that column during training. Test-time `pred_meta_f1` is still a proper holdout score. Nested OOF would be the textbook fix; we did not do that.

**4. Shuffled split ignores time.** `Time` is seconds from the first transaction. A shuffled split can put later transactions in train and earlier ones in test. Exp 05’s last fold is closer to a time holdout.

**5. Duplicates are reported, not dropped.** Fine to mention; they were not used as a cheat.

None of this is “the model peeked at the test labels.” The serious issue is **split mismatch for the neural nets**. Own that; it looks more professional than claiming 0.93 is a clean holdout.

---

## Slide 17 — Comparison to the Kaggle reference kernel

| Topic | Kaggle kernel | This project |
| ----- | ------------- | ------------ |
| Scaling | `RobustScaler` on Time and Amount | Same in Exp 05; `StandardScaler` in sklearn pipelines (01–04) |
| Split | Stratified K-fold, original imbalanced test | Stratified 80/20 (`random_state=42`) for sklearn; K-fold for Exp 05 |
| Undersampling | 50/50 subsample for classical models | Shown in Exp 05, **not** used as the final sklearn path |
| Oversampling | SMOTE during / after split | SMOTE on the Exp 05 fit split only |
| Models | LogReg, KNN, SVM, DT, small Keras net | DT, KNN, RF, six deep nets, RF stacker, PyTorch stacker |
| Metric | Accuracy on balanced data (kernel warning) | Precision / recall / F1 on **imbalanced** test |
| Combination | Compare undersample vs SMOTE nets separately | **Stack** ML + DL specialists |

We kept the kernel’s preprocessing and resampling ideas, rejected accuracy as a headline, and went further by stacking heterogeneous specialists.

---

## Slide 18 — What we would not do again

1. **Optimize recall alone** on an imbalanced test set. Train recall ≥ 95% produced thousands of false positives.
2. `class_weight="balanced"` **on the meta-learner.** It re-introduced the recall flood that the precision models had avoided.
3. **Random 50/50 undersampling as the only training set** for a high-capacity net. It throws away almost all legitimate structure.
4. **Dropping “outlier” frauds** (IQR on V10/V12/V14 in the kernel). Those tails are often the hard frauds we still miss.
5. **Judging models by accuracy.**

---

## Slide 19 — Final recommendation

**Production-style operating point (this repo):**

- Model: Experiment 06 PyTorch stacker
- Features: 10 frozen probabilities
- Threshold: **0.33** (chosen on train F1)
- Test (98 frauds): **precision 0.91, recall 0.95, F1 0.93** (93 TP, 5 FN, 9 FP)

**If a fully disjoint neural-net holdout is required**, quote Experiment 05’s `nn_precision_train` instead: precision 0.96, recall 0.70, F1 0.81 on that fold, and retrain Experiment 06 with out-of-fold neural scores.

**If only sklearn is allowed**, use the Experiment 03/04 stacker at threshold **0.40–0.50**: precision ~0.90, recall ~0.83, F1 ~0.86.

---

## Slide 20 — 3-minute close

1. **Problem:** 0.17% fraud, accuracy is a trap.
2. **Method:** heterogeneous stacking (Wolpert), not bagging or boosting.
3. **Pipeline:** split → scale on train → OOF stack for sklearn → specialist nets → PyTorch combiner → threshold on train.
4. **Result:** 0.91 / 0.95 / 0.93 at t = 0.33, with the Exp 05 split caveat.
5. **Deliverable:** `src/predict.py` scores new JSON without TensorFlow.

---

## Live demo

```bash
python src/predict.py data/input.json -o data/output.json
```

Expected shape of the output:

```json
{
  "prediction": "Legitimate",
  "class_id": 0,
  "probability": 0.0002,
  "threshold": 0.33,
  "status": "success"
}
```

If they ask to walk the code: `data_prep.py` (contract) → `train.py` (`FraudStack` + bundle) → `predict.py` (JSON in, JSON out).
