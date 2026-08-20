import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from torch import nn

import data_prep as dp

NN_NAMES = [
    "nn_f1_train",
    "nn_f1_val",
    "nn_recall_train",
    "nn_recall_val",
    "nn_precision_train",
    "nn_precision_val",
]
META_COLS = ["pred_dt", "pred_knn", "pred_rf", "pred_meta_f1"] + NN_NAMES
THRESHOLD = 0.33


def unwrap(model):
    if hasattr(model, "best_estimator_"):
        return model.best_estimator_
    return model


def latest_artifact(pattern):
    matches = sorted(dp.ARTIFACT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"no files matching {pattern} in {dp.ARTIFACT_DIR}")
    return matches[-1]


class StackerNet(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def _h5_arr(f, path):
    return np.array(f[path][()], dtype=np.float32)


def load_keras_net(path):
    path = Path(path)
    with zipfile.ZipFile(path) as zf, tempfile.TemporaryDirectory() as td:
        zf.extract("model.weights.h5", td)
        with h5py.File(Path(td) / "model.weights.h5", "r") as f:
            dense = []
            for name in ["dense", "dense_1", "dense_2", "dense_3", "dense_4"]:
                dense.append(
                    (
                        _h5_arr(f, f"layers/{name}/vars/0"),
                        _h5_arr(f, f"layers/{name}/vars/1"),
                    )
                )
            bns = []
            for name in [
                "batch_normalization",
                "batch_normalization_1",
                "batch_normalization_2",
            ]:
                bns.append(
                    (
                        _h5_arr(f, f"layers/{name}/vars/0"),
                        _h5_arr(f, f"layers/{name}/vars/1"),
                        _h5_arr(f, f"layers/{name}/vars/2"),
                        _h5_arr(f, f"layers/{name}/vars/3"),
                    )
                )

    def _bn(x, params, eps=1e-3):
        gamma, beta, mean, var = params
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    def predict_proba(X, batch_size=8192):
        X = np.asarray(X, dtype=np.float32)
        out = np.empty(len(X), dtype=np.float32)
        for i in range(0, len(X), batch_size):
            h = X[i : i + batch_size]
            h = np.maximum(h @ dense[0][0] + dense[0][1], 0)
            h = _bn(h, bns[0])
            h = np.maximum(h @ dense[1][0] + dense[1][1], 0)
            h = _bn(h, bns[1])
            h = np.maximum(h @ dense[2][0] + dense[2][1], 0)
            h = _bn(h, bns[2])
            h = np.maximum(h @ dense[3][0] + dense[3][1], 0)
            logits = h @ dense[4][0] + dense[4][1]
            out[i : i + batch_size] = 1.0 / (
                1.0 + np.exp(-np.clip(logits.ravel(), -80, 80))
            )
        return out

    return predict_proba


def predict_proba_torch(model, X, batch_size=2048, dev=None):
    if dev is None:
        dev = torch.device("cpu")
    model = model.to(dev)
    model.eval()
    X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=dev)
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            logits = model(X_t[i : i + batch_size])
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def metrics_at_threshold(y_true, proba, threshold):
    y_true = np.asarray(y_true).ravel()
    y_pred = (np.asarray(proba).ravel() >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
    }


def print_report(y_true, y_pred, title):
    print(f"\n{title}")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )
    print("Confusion matrix [[TN FP] [FN TP]]")
    print(confusion_matrix(y_true, y_pred))


class FraudStack:
    def __init__(
        self,
        dt,
        knn,
        rf,
        meta_rf,
        nn_predictors,
        amount_scaler,
        time_scaler,
        stacker,
        threshold=THRESHOLD,
    ):
        self.dt = dt
        self.knn = knn
        self.rf = rf
        self.meta_rf = meta_rf
        self.nn_predictors = nn_predictors
        self.amount_scaler = amount_scaler
        self.time_scaler = time_scaler
        self.stacker = stacker
        self.threshold = float(threshold)

    def meta_features(self, frame):
        X = frame[dp.FEATURE_COLUMNS]
        pred_dt = self.dt.predict_proba(X)[:, 1]
        pred_knn = self.knn.predict_proba(X)[:, 1]
        pred_rf = self.rf.predict_proba(X)[:, 1]
        ml = pd.DataFrame(
            {"pred_dt": pred_dt, "pred_knn": pred_knn, "pred_rf": pred_rf}
        )
        pred_meta = self.meta_rf.predict_proba(ml)[:, 1]
        nn_x = dp.nn_feature_matrix(X, self.amount_scaler, self.time_scaler)
        nn_scores = [self.nn_predictors[name](nn_x) for name in NN_NAMES]
        return np.column_stack(
            [ml.to_numpy(), pred_meta, *nn_scores]
        ).astype(np.float32)

    def predict_proba(self, frame):
        return predict_proba_torch(self.stacker, self.meta_features(frame))

    def predict(self, frame):
        proba = self.predict_proba(frame)
        return (proba >= self.threshold).astype(int), proba


def load_base_models():
    dt = unwrap(joblib.load(latest_artifact("dt_balanced_f1_*.pkl")))
    knn_search = joblib.load(latest_artifact("knn_random_search_result_*.pkl"))
    knn_cv_f1 = float(getattr(knn_search, "best_score_", float("nan")))
    knn = unwrap(knn_search)
    rf = unwrap(joblib.load(latest_artifact("rf_random_search_result_*.pkl")))
    meta_rf = unwrap(joblib.load(latest_artifact("rf_stacking_f1_*.pkl")))
    return dt, knn, rf, meta_rf, knn_cv_f1


def load_stacker_checkpoint():
    stamped = sorted(dp.ARTIFACT_DIR.glob("pytorch_stacking_exp06_*.pt"))
    stamped = [p for p in stamped if "meta" not in p.name]
    path = stamped[-1] if stamped else dp.ARTIFACT_DIR / "pytorch_stacking_exp06.pt"
    if not path.exists():
        raise FileNotFoundError("pytorch stacker checkpoint not found")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    n_inputs = int(ckpt.get("n_inputs", 10))
    threshold = float(ckpt.get("threshold", THRESHOLD))
    stacker = StackerNet(n_inputs)
    stacker.load_state_dict(ckpt["model"])
    stacker.eval()
    return stacker, threshold, path


def save_bundle(dt, knn, rf, meta_rf, amount_scaler, time_scaler, stacker, threshold):
    models_dir = dp.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    nn_src = dp.find_nn_dir()
    nn_dst = models_dir / "nn"
    nn_dst.mkdir(exist_ok=True)
    for name in NN_NAMES:
        shutil.copy2(nn_src / f"{name}.keras", nn_dst / f"{name}.keras")

    stacker_path = models_dir / "stacker.pt"
    torch.save(
        {
            "model": stacker.state_dict(),
            "n_inputs": 10,
            "threshold": float(threshold),
        },
        stacker_path,
    )

    scaler = None
    if hasattr(knn, "named_steps") and "scaler" in knn.named_steps:
        scaler = knn.named_steps["scaler"]
    else:
        scaler = StandardScaler()

    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(amount_scaler, models_dir / "amount_scaler.pkl")
    joblib.dump(time_scaler, models_dir / "time_scaler.pkl")
    payload = {
        "dt": dt,
        "knn": knn,
        "rf": rf,
        "meta_rf": meta_rf,
        "threshold": float(threshold),
        "feature_columns": dp.FEATURE_COLUMNS,
        "meta_columns": META_COLS,
        "nn_dir": "nn",
        "stacker_file": "stacker.pt",
        "class_names": {0: "Legitimate", 1: "Fraud"},
    }
    joblib.dump(payload, models_dir / "model.pkl")
    print("saved", models_dir / "model.pkl")
    print("saved", models_dir / "scaler.pkl")
    print("saved", stacker_path)
    return models_dir


def load_fraud_stack(models_dir=None):
    models_dir = Path(models_dir) if models_dir else dp.MODELS_DIR
    payload = joblib.load(models_dir / "model.pkl")
    amount_scaler = joblib.load(models_dir / "amount_scaler.pkl")
    time_scaler = joblib.load(models_dir / "time_scaler.pkl")
    nn_dir = models_dir / payload["nn_dir"]
    nn_predictors = {
        name: load_keras_net(nn_dir / f"{name}.keras") for name in NN_NAMES
    }
    ckpt = torch.load(
        models_dir / payload["stacker_file"],
        map_location="cpu",
        weights_only=True,
    )
    stacker = StackerNet(int(ckpt.get("n_inputs", 10)))
    stacker.load_state_dict(ckpt["model"])
    stacker.eval()
    return FraudStack(
        dt=payload["dt"],
        knn=payload["knn"],
        rf=payload["rf"],
        meta_rf=payload["meta_rf"],
        nn_predictors=nn_predictors,
        amount_scaler=amount_scaler,
        time_scaler=time_scaler,
        stacker=stacker,
        threshold=float(payload.get("threshold", ckpt.get("threshold", THRESHOLD))),
    )


def cv_table(model, X, y, name):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["precision", "recall", "f1"],
        n_jobs=-1,
    )
    row = {
        "model": name,
        "mean_precision": float(scores["test_precision"].mean()),
        "mean_recall": float(scores["test_recall"].mean()),
        "mean_f1": float(scores["test_f1"].mean()),
    }
    print(
        f"{name:20s}  P={row['mean_precision']:.4f}  "
        f"R={row['mean_recall']:.4f}  F1={row['mean_f1']:.4f}"
    )
    return row


def main():
    df = dp.load_transactions()
    dp.dataset_report(df)
    X, y = dp.split_xy(df)
    X_train, X_test, y_train, y_test = dp.stratified_split(X, y)
    print()
    print("Train:", X_train.shape, "frauds", int(y_train.sum()))
    print("Test: ", X_test.shape, "frauds", int(y_test.sum()))

    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    tree = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "dt",
                DecisionTreeClassifier(
                    criterion="entropy",
                    max_depth=5,
                    max_features=None,
                    min_samples_leaf=23,
                    min_samples_split=54,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    dt, knn, rf, meta_rf, knn_cv_f1 = load_base_models()

    print("\n5-fold stratified CV (train fold)")
    cv_table(logistic, X_train, y_train, "Logistic Regression")
    cv_table(tree, X_train, y_train, "Decision Tree")
    print(f"{'KNN (search CV F1)':20s}  F1={knn_cv_f1:.4f}")

    logistic.fit(X_train, y_train)
    tree.fit(X_train, y_train)
    print_report(y_test, logistic.predict(X_test), "Logistic Regression - test")
    print_report(y_test, tree.predict(X_test), "Decision Tree - test")

    stack_test_path = dp.ROOT / "_data" / "stacking_test_data_exp04.csv"
    if stack_test_path.exists():
        stack_test = pd.read_csv(stack_test_path)
        for col, label in [
            ("pred_knn", "KNN - test @ 0.50"),
            ("pred_rf", "Random Forest - test @ 0.50"),
        ]:
            pred = (stack_test[col].to_numpy() >= 0.5).astype(int)
            print_report(y_test, pred, label)

    print("\nRequired thresholds on the logistic model")
    log_proba = logistic.predict_proba(X_test)[:, 1]
    for t in (0.3, 0.5, 0.7):
        m = metrics_at_threshold(y_test, log_proba, t)
        print(
            f"  t={t:.1f}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
            f"F1={m['f1']:.4f}  FP={m['fp']} FN={m['fn']}"
        )

    amount_scaler, time_scaler = dp.fit_time_amount_scalers(df)
    stacker, threshold, ckpt_path = load_stacker_checkpoint()
    print("\nLoaded stacker", ckpt_path, "threshold", threshold)

    exp06_test = dp.ROOT / "_data" / "stacking_test_data_exp06.csv"
    if exp06_test.exists():
        meta_test = pd.read_csv(exp06_test)
        proba = predict_proba_torch(stacker, meta_test[META_COLS].to_numpy(dtype=np.float32))
        y_hat = (proba >= threshold).astype(int)
        print_report(meta_test[dp.TARGET], y_hat, "Final model (PyTorch stack) - test")
        m = metrics_at_threshold(meta_test[dp.TARGET], proba, threshold)
        print(
            f"Fraud  P={m['precision']:.2f}  R={m['recall']:.2f}  "
            f"F1={m['f1']:.2f}  TP={m['tp']} FN={m['fn']} FP={m['fp']}"
        )

    save_bundle(dt, knn, rf, meta_rf, amount_scaler, time_scaler, stacker, threshold)

    bundle = load_fraud_stack()
    sample = X_test.iloc[:3]
    pred, proba = bundle.predict(sample)
    print("\nSmoke test on 3 test rows:", pred.tolist(), proba.round(4).tolist())
    print("Final model: heterogeneous stack, threshold", bundle.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
