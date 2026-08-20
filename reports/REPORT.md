# Credit Card Fraud Detection: Heterogeneous Stacked Generalization

**Project:** Credit Card Fraud Detection Pipeline  
**Course:** First Machine Learning Project (Maktab 145)  
**Final model:** Experiment 06 — PyTorch meta-learner on mixed ML + DL bases  
**Primary metric:** Fraud-class F1, with precision and recall reported together  

---

## 1. What this method is called

The approach used across these six experiments is **stacked generalization** (stacking), specifically a **heterogeneous ensemble**.

- **Ensemble learning** means combining several models instead of trusting one.
- **Stacking** (Wolpert, 1992) trains a second model — the **meta-learner** — on the *outputs* of the first models, not on the raw transactions.
- **Heterogeneous** means the first-level models are not copies of the same algorithm. They differ in:
  - family (trees, nearest neighbors, random forests, neural nets)
  - training recipe (class weights, SMOTE, undersampling)
  - objective (precision, recall, or F1)

In this project the two-level structure is:

```
Level 0 (base learners)
  classical ML: Decision Tree, KNN, Random Forest
  deep learning: six Keras networks (F1 / recall / precision × train / val)
        ↓  fraud probabilities
Level 1 (meta-learner)
  Experiment 02–04: Random Forest on 3 ML scores
  Experiment 06:   PyTorch network on 10 mixed ML + DL scores
```

A related name is **stacked ensemble** or **super learner**. It is *not* bagging (same model, different samples) and *not* boosting (sequential residual fitting). It is closest to stacking, with an extra idea used in Experiment 05: train specialists for different error types, then let the meta-model decide when to trust each specialist.

That is why some bases look “perfect at recall” and others “perfect at precision.” The stacker is supposed to keep the high-recall votes when they agree with the high-precision votes, and ignore a lone recall model that would flood false alarms.

**References for the method**

- David H. Wolpert, “Stacked Generalization,” *Neural Networks*, 5(2), 1992.
- Leo Breiman, “Stacked Regressions,” *Machine Learning*, 24, 1996.

**Reference for the dataset and resampling recipe**

- Dataset: Machine Learning Group — ULB, [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle).
- Kernel used as a design reference: Janio Martinez Bachmann, [Credit Fraud || Dealing with Imbalanced Datasets](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets).  
  Local copies: `research/credit-fraud-dealing-with-imbalanced-datasets.ipynb` and `refrences/credit-fraud-dealing-with-imbalanced-datasets.ipynb`.

The Kaggle kernel is **not** copied as the final solution. It supplied the *imbalanced-data playbook* used in Experiment 05: `RobustScaler` on Time/Amount, split before resampling, random 50/50 undersampling, SMOTE on train only, and a small neural net. Experiments 02–04 follow a different sklearn stacking path. Experiment 06 combines both paths.

---

## 2. Problem, data, and evaluation rules

### 2.1 Task

Binary classification of credit-card transactions:

| Class | Meaning | Approximate share |
|---|---|---|
| 0 | Legitimate | 99.83% |
| 1 | Fraud | 0.17% (492 frauds in 284,807 rows) |

Features: `Time`, `Amount`, and PCA components `V1`–`V28`. The V features are already scaled; Time and Amount are not.

### 2.2 Why accuracy is the wrong headline

A model that always predicts Legitimate is about 99.8% accurate and catches **zero** fraud. The assignment therefore forbids accuracy as the primary score.

| Metric | What it answers in this project |
|---|---|
| **Recall** | Of the real frauds, how many did we catch? Missed fraud is expensive. |
| **Precision** | Of the alerts we raise, how many are real? Too many false alarms is also expensive. |
| **F1** | Harmonic mean of precision and recall. It only stays high if **both** stay high. |

The operating point we wanted, and later obtained, is high fraud F1 **without** collapsing precision.

### 2.3 Split used for the sklearn track (Experiments 01–04 and 06)

```
train_test_split(..., test_size=0.2, random_state=42, stratify=y)
```

| Split | Rows | Frauds |
|---|---|---|
| Train | 227,845 | 394 |
| Test  | 56,962  | 98 |

Base-model stacking features for trees/KNN/RF were built with **out-of-fold** `predict_proba` (`StratifiedKFold`, 5 folds) so the meta-learner does not train on in-sample scores.

Thresholds `0.3`, `0.5`, and `0.7` were always reported, as required. Extra cutoffs (F2, then F1) were chosen on **train only** and locked for test.

---

## 3. End-to-end picture

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

## 4. Experiment 01 — baselines

**Notebook:** `notebooks/experiment01.ipynb`

Goal: a first honest pipeline on the imbalanced data.

- Load the public credit-card file.
- Stratified 80/20 split.
- `Pipeline(StandardScaler → model)` so scaling is fit on train only.
- Models: **Decision Tree**, **KNN**, **Logistic Regression**.
- Search scored with **F1**, not accuracy.
- Save models and reports.

**Lesson.** Logistic regression is a weak baseline on this nonlinear PCA space. Trees and KNN are more plausible. Accuracy looks excellent for every model and must be ignored. This experiment established the split, the metric, and the “scaler inside the pipeline” rule used later.

---

## 5. Experiment 02 — classical stacking

**Notebook:** `notebooks/experiment02.ipynb`

Goal: combine complementary sklearn models.

Level-0 models (after randomized search):

| Model | Role |
|---|---|
| Decision Tree | Interpretable nonlinear rules; originally `class_weight=None` |
| KNN (`n_neighbors=3`) | Local similarity; no class weight |
| Random Forest | Already `class_weight="balanced_subsample"` |

Stacking procedure:

1. Out-of-fold fraud probabilities on train → columns `pred_dt`, `pred_knn`, `pred_rf`.
2. Full-train refit probabilities on test (no leakage into the test labels).
3. A **Random Forest meta-model** trained only on those three scores.

Saved artifacts: `_data/stacking_train_data.csv`, `_data/stacking_test_data.csv`, `_models/rf_stacking_search_result_2026_08_20_17_30_45.pkl`.

Best meta RF (F1 search): `n_estimators=300`, `max_depth=20`, `min_samples_leaf=46`, `class_weight=None`, entropy splits.

**Lesson.** A linear mix is unnecessary. A small tree ensemble can learn “alert if KNN and RF both fire, even if the DT is quiet.” This is already stacked generalization, but still **homogeneous in methodology** (all sklearn classifiers, one loss).

---

## 6. Experiment 03 — thresholds

**Notebook:** `notebooks/experimnet03.ipynb` (filename typo kept)

Goal: treat `0.5` as a default, not a law.

The assignment asks for `0.3 / 0.5 / 0.7`. We also swept a fine grid.

| Rule | Threshold | Precision | Recall | F1 | TP / FN / FP |
|---|---|---|---|---|---|
| F2 (recall-heavy) | 0.25 | 0.83 | 0.87 | 0.85 | 85 / 13 / 17 |
| **F1 (balanced)** | **0.40** | **0.90** | **0.83** | **0.86** | **81 / 17 / 9** |
| Required | 0.50 | 0.91 | 0.83 | 0.87 | 81 / 17 / 8 |
| Required | 0.70 | 0.97 | 0.74 | 0.84 | 73 / 25 / 2 |

Pushing recall toward 95% by lowering the cutoff **failed**: even at `t=0.01` train recall only reached about 92%, and test precision collapsed (about 210 extra false positives).

The 13 frauds missed at `t=0.25` sat near `(0, 0)` for every base model. Stacking cannot invent a signal that no base model has.

**Lesson.** Threshold tuning is part of the model. F1 (or F2) must be chosen on train/validation, then frozen.

---

## 7. Experiment 04 — F1-oriented sklearn stack

**Notebook:** `notebooks/experiment04.ipynb`

Goal: improve fraud F1 without a recall quota.

Changes:

- Retrain the tree with the same hyperparameters plus `class_weight="balanced"`.
- Rebuild only `pred_dt` (OOF). Keep KNN and RF scores.
- Meta RF **without** class weights (weights had destroyed precision).
- Pick the highest train **F1** cutoff.

| Setting | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| Exp 03 stacker, F2 | 0.25 | 0.83 | 0.87 | 0.85 |
| Exp 03 stacker, F1 | 0.40 | 0.90 | 0.83 | 0.86 |
| Exp 04 new DT + meta | 0.30 | 0.86 | 0.85 | 0.85 |

The balanced tree raised OOF recall but, used alone at 0.5, precision was unusable (~0.04). The meta-model recovered precision. It did **not** beat the original stacker on F1. The practical sklearn operating point remained: **original stacker at 0.40–0.50**.

Skipped on purpose: 50/50 undersampling, dropping fraud outliers, and SMOTE (not in `requirements.txt` at that stage).

---

## 8. Experiment 05 — deep specialists (Kaggle recipe)

**Notebook:** `notebooks/experiment05/experiment05.ipynb`  
**Artifacts:** `notebooks/experiment05_models/`

Goal: add neural nets that *specialize*, instead of one net that compromises.

### 8.1 Data recipe (from the Kaggle kernel)

1. Scale **only Time and Amount** with `RobustScaler` (robust to amount outliers). Leave `V1`–`V28` as they are.
2. Hold out an original **imbalanced** test fold (stratified 5-fold, last fold).
3. From the remaining train data, hold a further imbalanced **validation** split.
4. Show a 50/50 **random undersample** (too small for a deep net; not used for the six models).
5. **SMOTE** the fit split only (`sampling_strategy="minority"`). Never resample the test set.

### 8.2 Six independent Keras runs

Same architecture: `256 → 128 → 64 → 32 → 1` (ReLU, batch norm, dropout, sigmoid).  
Each run saved the epoch that maximized **one** metric on **one** split:

| File | Optimized for |
|---|---|
| `nn_f1_train.keras` | train F1 |
| `nn_f1_val.keras` | validation F1 |
| `nn_recall_train.keras` | train recall |
| `nn_recall_val.keras` | validation recall |
| `nn_precision_train.keras` | train precision |
| `nn_precision_val.keras` | validation precision |

Precision selection required recall ≥ 0.40 so a model could not “win” by predicting almost no fraud.

### 8.3 Honest holdout (Experiment 05’s own test fold, 98 frauds)

| Model | Threshold | Precision | Recall | F1 | TP / FN / FP |
|---|---|---|---|---|---|
| `nn_precision_train` | 0.90 | **0.96** | 0.70 | **0.81** | 69 / 29 / 3 |
| `nn_f1_val` | 0.91 | 0.93 | 0.68 | 0.79 | 67 / 31 / 5 |
| `nn_f1_train` | 0.95 | 0.92 | 0.68 | 0.78 | 67 / 31 / 6 |
| `nn_precision_val` | 0.80 | 0.93 | 0.67 | 0.78 | 66 / 32 / 5 |
| `nn_recall_train` | 0.05 | 0.27 | **0.83** | 0.41 | 81 / 17 / 217 |
| `nn_recall_val` | 0.05 | 0.25 | **0.83** | 0.39 | 81 / 17 / 239 |

This table is the important one for the neural nets. Precision specialists are conservative and precise. Recall specialists catch more fraud and drown the operator in false positives. **That diversity is useful only if a later model can choose among them.**

---

## 9. Experiment 06 — heterogeneous stack (final)

**Notebook:** `notebooks/experiment06.ipynb`

Goal: one combiner that sees *all* of the previous work.

### 9.1 Meta-features (10 scores)

From Experiments 02–04 (OOF on the sklearn split):

- `pred_dt` — balanced decision tree  
- `pred_knn` — KNN  
- `pred_rf` — random forest  
- `pred_meta_f1` — Experiment 04 Random Forest stacker  

From Experiment 05 (frozen Keras weights, numpy inference, no TensorFlow required):

- `nn_f1_train`, `nn_f1_val`  
- `nn_recall_train`, `nn_recall_val`  
- `nn_precision_train`, `nn_precision_val`  

### 9.2 Meta-learner

A small **PyTorch** network, because it only mixes probabilities:

```
Linear(10 → 64) → BatchNorm → ReLU → Dropout(0.3)
Linear(64 → 32) → ReLU → Dropout(0.2)
Linear(32 → 1)  → sigmoid  (BCE with logits)
```

- No class weights (they pulled the combiner toward the recall nets and hurt precision).  
- 20% stratified validation inside the sklearn **train** fold.  
- Early stopping on validation loss.  
- Final cutoff chosen by **train fraud F1** (`t = 0.33`).

### 9.3 Test results (sklearn split, 98 frauds)

This is the result that matches the target operating point:

```
              precision    recall  f1-score   support

  Legitimate       1.00      1.00      1.00     56864
       Fraud       0.91      0.95      0.93        98
```

| Setting | Threshold | Precision | Recall | F1 | TP | FN | FP |
|---|---|---|---|---|---|---|---|
| Exp 03 stacker F2 | 0.25 | 0.83 | 0.87 | 0.85 | 85 | 13 | 17 |
| Exp 03 stacker F1 | 0.40 | 0.90 | 0.83 | 0.86 | 81 | 17 | 9 |
| Exp 04 KNN | 0.50 | 0.91 | 0.83 | 0.87 | 81 | 17 | 8 |
| Exp 04 meta RF | 0.30 | 0.86 | 0.85 | 0.85 | 83 | 15 | 14 |
| **Exp 06 PyTorch stack** | **0.33** | **0.91** | **0.95** | **0.93** | **93** | **5** | **9** |

Relative to the best sklearn stacker (F1 ≈ 0.86–0.87):

- about **12 extra frauds caught** (81 → 93),
- false positives stay in the same band (8–9),
- fraud F1 moves from ~0.86 to **0.93**.

A linear SVM was tried first as the meta-learner. With `class_weight="balanced"` it copied the recall nets and precision fell. Without class weights it followed the precision nets. The PyTorch net could use **both** families at once, which is the point of stacking specialists.

### 9.4 Caveat (must be stated)

Experiment 05 used a **different** fold split than Experiments 01–04. Many rows in the sklearn test set were seen by the Keras nets during their own training. Neural-net columns on that test fold are therefore **optimistic**.

The honest neural-net holdout remains Experiment 05’s own table (best F1 ≈ 0.81, precision ≈ 0.96). Experiment 06 still demonstrates the stacking idea on a shared sklearn test set, and the sklearn bases (`pred_dt` / `pred_knn` / `pred_rf`) are properly out-of-fold.

---

## 10. Why mixing “precision models” and “recall models” works

Each base learner fails differently.

| Specialist | Typical behavior | Failure mode |
|---|---|---|
| Precision / F1 nets, KNN | Alert only when sure | Misses unusual frauds |
| Recall nets, balanced tree | Alert on weak evidence | Many legitimate customers blocked |
| Random forest | Strong middle ground | Still misses the same hard tail |

The meta-learner sees a 10-dimensional vote. Empirically:

- If precision nets and KNN all fire, the transaction is almost certainly fraud.  
- If only a recall net fires, it is usually a false alarm.  
- If recall nets fire **and** the RF/KNN scores are moderate, it is often a hard fraud worth catching.

That pattern cannot be expressed by a single threshold on a single model. It *can* be expressed by a second model trained on the votes. That is stacked generalization.

---

## 11. Comparison to the Kaggle reference kernel

| Topic | Kaggle kernel | This project |
|---|---|---|
| Scaling | `RobustScaler` on Time and Amount | Same in Exp 05; `StandardScaler` in sklearn pipelines (01–04) |
| Split | Stratified K-fold, original imbalanced test | Stratified 80/20 (`random_state=42`) for sklearn; K-fold for Exp 05 |
| Undersampling | 50/50 subsample for classical models | Shown in Exp 05, **not** used as the final sklearn path (information loss) |
| Oversampling | SMOTE during / after split | SMOTE on the Exp 05 fit split only |
| Models | LogReg, KNN, SVM, DT, small Keras net | DT, KNN, RF, six deep nets, RF stacker, PyTorch stacker |
| Metric | Accuracy on balanced data (kernel warning) | Precision / recall / F1 on **imbalanced** test |
| Combination | Compare undersample vs SMOTE nets separately | **Stack** ML + DL specialists |

We kept the kernel’s preprocessing and resampling ideas, rejected accuracy as a headline, and went further by stacking heterogeneous specialists.

---

## 12. What we would not do again

1. **Optimize recall alone** on an imbalanced test set. Train recall ≥ 95% produced thousands of false positives.  
2. **`class_weight="balanced"` on the meta-learner.** It re-introduced the recall flood that the precision models had avoided.  
3. **Random 50/50 undersampling as the only training set** for a high-capacity net. It throws away almost all legitimate structure.  
4. **Dropping “outlier” frauds** (IQR on V10/V12/V14 in the kernel). Those tails are often the hard frauds we still miss.  
5. **Judging models by accuracy.**

---

## 13. Final recommendation

**Production-style operating point (this repo):**

- Model: Experiment 06 PyTorch stacker  
- Features: 10 frozen probabilities listed in §9.1  
- Threshold: **0.33** (chosen on train F1)  
- Test (98 frauds): **precision 0.91, recall 0.95, F1 0.93** (93 TP, 5 FN, 9 FP)

**If a fully disjoint neural-net holdout is required**, quote Experiment 05’s `nn_precision_train` instead: precision 0.96, recall 0.70, F1 0.81 on that fold, and retrain Experiment 06 with out-of-fold neural scores.

**If only sklearn is allowed**, use the Experiment 03/04 stacker at threshold **0.40–0.50**: precision ~0.90, recall ~0.83, F1 ~0.86.

---

## 14. Artifact map

| Path | Content |
|---|---|
| `notebooks/experiment01.ipynb` | DT / KNN / Logistic baselines |
| `notebooks/experiment02.ipynb` | OOF stacking, RF meta-search |
| `notebooks/experimnet03.ipynb` | Thresholds 0.3 / 0.5 / 0.7, F2 vs F1 |
| `notebooks/experiment04.ipynb` | Balanced tree, F1-oriented meta RF |
| `notebooks/experiment05/experiment05.ipynb` | SMOTE + six Keras specialists |
| `notebooks/experiment05_models/` | `.keras` weights, `manifest.json` |
| `notebooks/experiment06.ipynb` | Heterogeneous PyTorch stack |
| `_data/stacking_*_exp04.csv` / `_exp06.csv` | Meta-feature tables |
| `_models/` | Pickled sklearn models and PyTorch checkpoint |
| `research/` and `refrences/` | Kaggle imbalanced-data kernel |

---

## 15. References

1. Machine Learning Group — ULB. *Credit Card Fraud Detection*. Kaggle. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
2. Janio Martinez Bachmann. *Credit Fraud || Dealing with Imbalanced Datasets*. Kaggle Notebook. https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets  
3. David H. Wolpert. “Stacked Generalization.” *Neural Networks* 5(2), 1992, 241–259.  
4. Leo Breiman. “Stacked Regressions.” *Machine Learning* 24, 1996, 49–64.  
5. Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall, and W. Philip Kegelmeyer. “SMOTE: Synthetic Minority Over-sampling Technique.” *JAIR* 16, 2002, 321–357.  
6. Scikit-learn documentation: *Ensemble methods — Stacked generalization*. https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization  
7. Course brief: `notebooks/Project-Definition-v2-fraud.ipynb` (stratified split, F1/precision/recall, thresholds 0.3 / 0.5 / 0.7).
