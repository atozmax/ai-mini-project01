# Mentor Q&A

Spoken answers for likely questions. Sibling to `presentation.md`. Do not treat this as a slide deck.

---

## 1. RobustScaler vs StandardScaler

**StandardScaler** (z-score):

```
scaled = (x - mean) / std
```

Mean and standard deviation are **sensitive to outliers**. One huge `Amount` (thousands of dollars) pulls the mean up and inflates the std, so ordinary transactions get squashed.

**RobustScaler**:

```
scaled = (x - median) / IQR
IQR    = 75th percentile - 25th percentile
```

Median and interquartile range ignore the tails. A few luxury purchases do not rewrite the scale of every other row.

### Where we used which

| Track | Scaler | What it scales |
| ----- | ------ | -------------- |
| Experiments 01–04 (DT, KNN, RF, Logistic) | `StandardScaler` **inside a `Pipeline`** | All 30 features, **fit on train only** |
| Experiments 05–06 (Keras nets + inference) | `RobustScaler` | **Only `Time` and `Amount`**. `V1`–`V28` are left alone |

### Why not RobustScaler on everything?

`V1`–`V28` are already PCA components. They are already centered and on a comparable scale. Rescaling them again is unnecessary. `Time` (0 to ~172,792 seconds) and `Amount` (many small values, a few very large) are the raw, skewed columns.

### Why RobustScaler for the nets, StandardScaler for sklearn?

- KNN and Logistic **need** features on one scale; trees/RF do not, but putting `StandardScaler` in the pipeline is still correct and does not leak.
- Amount outliers are a known issue on this dataset (same reason as the Kaggle kernel). RobustScaler is the safer choice before a neural net.
- Sklearn pipelines scale **all** columns because KNN uses Euclidean distance on the whole vector. The nets only need Time/Amount fixed.

### One-liner

StandardScaler uses mean and std; RobustScaler uses median and IQR. We used RobustScaler only on Time and Amount for the deep models because Amount has heavy outliers. Sklearn models use StandardScaler in a Pipeline, fit on train only.

---

## 2. Stratified K-fold

**Ordinary K-fold** splits rows into K chunks at random (or in order). On this dataset that is dangerous: fraud is **0.17%**. A random fold can easily get **almost no frauds**, so F1/recall in that fold is noise.

**Stratified K-fold** keeps the **same class ratio** in every fold. Each fold still has about 0.17% fraud, so every validation score is computed on a realistic mix.

In sklearn:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

What that means in words:

1. Split the 394 train frauds into 5 groups (about 79 each).
2. Split the legitimate rows into 5 groups the same way.
3. Each fold = one fraud group + one legit group.
4. Train on 4 folds, validate on the held-out fold. Repeat 5 times so **every train row is validated once**.

### Where we used it

- **Hyperparameter search** (Exp 01, 02): `RandomizedSearchCV(..., cv=StratifiedKFold(...))` so F1 is not luck of one split.
- **Stacking without leakage** (Exp 02, 04): `cross_val_predict(..., method="predict_proba")` with 5 stratified folds. Each train row’s `pred_dt` / `pred_knn` / `pred_rf` comes from a clone that **did not train on that row**. That is out-of-fold stacking.
- **`src/train.py`**: 5-fold stratified CV on the logistic and tree baselines.
- **Experiment 05 holdout:** `StratifiedKFold(n_splits=5, shuffle=False)` — last fold is test. `shuffle=False` keeps time order (the CSV is time-sorted). That is stratified **and** closer to a time holdout.

### Vs the 80/20 split

`train_test_split(..., stratify=y)` is the same idea for a **single** split: test still has ~98 frauds, not 0. Stratified K-fold is that idea repeated K times.

### One-liner

With 0.17% fraud, a normal K-fold can produce a fold with almost no frauds. Stratified K-fold forces every fold to keep that 0.17% ratio. We used it for CV, for out-of-fold stacking scores, and for Exp 05’s test fold.

---

## 3. Neural networks: architecture and parameter counts

There are **two architectures**. Six Keras nets share the first one. The PyTorch stacker is the second.

### A. Six Keras specialists (Experiment 05)

**Same architecture, six independent trainings.** They do not share weights. Each run starts fresh and saves the epoch that maximized **one** metric on **one** split:

| File | Optimized for |
| ---- | ------------- |
| `nn_f1_train` | F1 on the (original, not SMOTE) fit split |
| `nn_f1_val` | F1 on validation |
| `nn_recall_train` / `nn_recall_val` | recall on train / val |
| `nn_precision_train` / `nn_precision_val` | precision on train / val (only if recall ≥ 0.40) |

**Input:** 30 numbers — Robust-scaled Amount, Robust-scaled Time, then `V1`–`V28`.  
Trained on **SMOTE’d** fit data; checkpoints scored on the **real** imbalanced split so “best F1” is not inflated by synthetic rows.

**Architecture** (`256 → 128 → 64 → 32 → 1`):

```
Input (30)
  Dense(256, ReLU) + BatchNorm + Dropout(0.35)
  Dense(128, ReLU) + BatchNorm + Dropout(0.30)
  Dense(64,  ReLU) + BatchNorm + Dropout(0.25)
  Dense(32,  ReLU) + Dropout(0.15)
  Dense(1, sigmoid)
```

- Optimizer: Adam, `lr=1e-3`
- Loss: binary cross-entropy
- Batch size 512, up to 40 epochs, early stopping patience 10 on the target metric
- `ReduceLROnPlateau` on val loss

**Why six copies?**  
Not six different architectures. Six **specialists**: recall nets fire often, precision nets fire rarely. The stacker later chooses when to trust which vote.

**Parameter count (each of the six nets)** — from Keras `summary()`:

| Layer | Params |
| ----- | ------ |
| Dense 30 → 256 | 7,936 |
| BatchNorm (256) | 1,024 |
| Dropout | 0 |
| Dense 256 → 128 | 32,896 |
| BatchNorm (128) | 512 |
| Dropout | 0 |
| Dense 128 → 64 | 8,256 |
| BatchNorm (64) | 256 |
| Dropout | 0 |
| Dense 64 → 32 | 2,080 |
| Dropout | 0 |
| Dense 32 → 1 | 33 |
| **Total** | **52,993** |
| Trainable | **52,097** |
| Non-trainable | **896** (BN moving mean/variance) |

All six together: 6 × 52,993 ≈ 318k parameters, but they are six separate models, not one big net.

How Dense params are computed:

```
params = (in_features × out_features) + out_features   # weights + bias
```

Example: `(30 × 256) + 256 = 7,936`.

---

### B. PyTorch stacker (Experiment 06) — the final combiner

**Input:** 10 probabilities, not raw transactions  
(`pred_dt`, `pred_knn`, `pred_rf`, `pred_meta_f1`, plus the six Keras scores).

```
Linear(10 → 64) → BatchNorm1d(64) → ReLU → Dropout(0.30)
Linear(64 → 32) → ReLU → Dropout(0.20)
Linear(32 → 1)  → logits, then sigmoid at inference
```

- Loss: `BCEWithLogitsLoss` (no class weights — weights pulled it toward the recall nets)
- Adam `lr=1e-3`, batch 512, up to 40 epochs, early stop on val loss (patience 8)
- Threshold **0.33** chosen on **train F1**, then frozen

**Trainable parameter count:**

| Layer | Formula | Params |
| ----- | ------- | ------ |
| Linear 10 → 64 | `(10 × 64) + 64` | 704 |
| BatchNorm1d(64) | scale and shift: `64 + 64` | 128 |
| Linear 64 → 32 | `(64 × 32) + 32` | 2,080 |
| Linear 32 → 1 | `(32 × 1) + 1` | 33 |
| **Total trainable** | | **2,945** |

BN also stores 128 running-mean/variance buffers (not trained by gradient). Dropout has zero parameters.

This net is small **on purpose**: it only mixes 10 scores. The heavy lifting is in the six Keras nets (~53k each) and the sklearn bases.

---

### How many neural nets in the final system?

**Seven:** six Keras specialists (same 30 → 256 → 128 → 64 → 32 → 1 architecture, 52,993 params each) plus one PyTorch stacker (10 → 64 → 32 → 1, 2,945 trainable params). Dropout is on during training and **off** at inference.
