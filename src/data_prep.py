from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = [
    ROOT / "data" / "creditcard.csv",
    ROOT / "_data" / "creditcard.csv",
]
MODELS_DIR = ROOT / "models"
ARTIFACT_DIR = ROOT / "_models"
NN_DIR_CANDIDATES = [
    ROOT / "notebooks" / "experiment05" / "_models",
    ROOT / "notebooks" / "experiment05_models",
]

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
V_COLUMNS = [f"V{i}" for i in range(1, 29)]
TARGET = "Class"


def find_csv(path=None):
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "creditcard.csv not found. Put it in data/ or _data/."
    )


def find_nn_dir():
    for candidate in NN_DIR_CANDIDATES:
        if (candidate / "nn_f1_train.keras").exists():
            return candidate
    raise FileNotFoundError("experiment 05 keras weights not found")


def load_transactions(path=None):
    csv_path = find_csv(path)
    df = pd.read_csv(csv_path, index_col=0)
    missing = [c for c in FEATURE_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df[TARGET] = df[TARGET].astype(int)
    return df


def dataset_report(df):
    n_features = df.drop(columns=[TARGET]).shape[1]
    class_counts = df[TARGET].value_counts().sort_index()
    n_fraud = int(class_counts.get(1, 0))
    n_legit = int(class_counts.get(0, 0))
    report = {
        "n_samples": int(len(df)),
        "n_features": int(n_features),
        "n_legitimate": n_legit,
        "n_fraud": n_fraud,
        "fraud_rate": float(n_fraud / len(df)),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    print("Number of Samples:", report["n_samples"])
    print("Number of Features:", report["n_features"])
    print("Class Distribution:")
    print(f"  Legitimate (0): {n_legit}")
    print(f"  Fraud (1):      {n_fraud}  ({100 * report['fraud_rate']:.4f}%)")
    print("Missing Values:", report["missing_values"])
    print("Duplicate Rows:", report["duplicate_rows"])
    print()
    print(df[FEATURE_COLUMNS].describe().T[["mean", "std", "min", "50%", "max"]])
    return report


def split_xy(df):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].astype(int)
    return X, y


def stratified_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def fit_standard_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train[FEATURE_COLUMNS])
    return scaler


def transform_features(scaler, X):
    values = scaler.transform(X[FEATURE_COLUMNS])
    return pd.DataFrame(values, index=X.index, columns=FEATURE_COLUMNS)


def fit_time_amount_scalers(frame):
    amount_scaler = RobustScaler().fit(frame[["Amount"]])
    time_scaler = RobustScaler().fit(frame[["Time"]])
    return amount_scaler, time_scaler


def nn_feature_matrix(frame, amount_scaler, time_scaler):
    scaled_amount = amount_scaler.transform(frame[["Amount"]])
    scaled_time = time_scaler.transform(frame[["Time"]])
    v = frame[V_COLUMNS].to_numpy(dtype=np.float32)
    return np.hstack([scaled_amount, scaled_time, v]).astype(np.float32)


def records_to_frame(payload):
    if isinstance(payload, dict) and "transactions" in payload:
        rows = payload["transactions"]
        single = False
    elif isinstance(payload, list):
        rows = payload
        single = False
    elif isinstance(payload, dict):
        rows = [payload]
        single = True
    else:
        raise ValueError("input must be a transaction object or a list")

    frame = pd.DataFrame(rows)
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError("missing features: " + ", ".join(missing))
    return frame[FEATURE_COLUMNS].astype(float), single


def main():
    df = load_transactions()
    dataset_report(df)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    scaler = fit_standard_scaler(X_train)
    X_train_scaled = transform_features(scaler, X_train)
    X_test_scaled = transform_features(scaler, X_test)
    print()
    print("Train:", X_train.shape, "frauds", int(y_train.sum()))
    print("Test: ", X_test.shape, "frauds", int(y_test.sum()))
    print("Scaler fitted on train only.")
    print("Train scaled mean (first 3):", X_train_scaled.iloc[:, :3].mean().round(4).tolist())
    print("Test scaled mean (first 3): ", X_test_scaled.iloc[:, :3].mean().round(4).tolist())


if __name__ == "__main__":
    main()
