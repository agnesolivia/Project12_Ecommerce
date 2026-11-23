"""
prepare_models.py
Melatih ulang dan menyimpan artefak model (Random Forest, KNN, dan StandardScaler)
dari data/processed/ dengan konfigurasi yang identik (RANDOM_STATE=42)
untuk kebutuhan deployment Streamlit.
"""
import os
import json
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from config import RANDOM_STATE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
ARTIFACT_DIR = os.path.join(BASE_DIR, "output")

SCALE_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "OperatingSystems", "Browser", "Region", "TrafficType",
]


def build_and_save_artifacts():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    X_train_scaled = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_scaled.csv"))
    X_test_scaled = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_scaled.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv"))["Revenue"]
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv"))["Revenue"]

    # 1. StandardScaler (fit hanya pada X_train)
    scaler = StandardScaler()
    scaler.fit(X_train[SCALE_COLS])
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))

    # 2. Random Forest (100 trees)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, os.path.join(ARTIFACT_DIR, "rf_model.pkl"))

    # 3. KNN (K = 5)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    joblib.dump(knn_model, os.path.join(ARTIFACT_DIR, "knn_model.pkl"))

    # Verifikasi konsistensi terhadap hasil JSON tim
    for name, model, X_eval, json_file in [
        ("Random Forest", rf_model, X_test, "hasil_rf.json"),
        ("KNN", knn_model, X_test_scaled, "hasil_knn.json"),
    ]:
        y_pred = model.predict(X_eval)
        acc = round(accuracy_score(y_test, y_pred), 4)
        prec = round(precision_score(y_test, y_pred, zero_division=0), 4)
        rec = round(recall_score(y_test, y_pred, zero_division=0), 4)
        f1 = round(f1_score(y_test, y_pred, zero_division=0), 4)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print(f"{name:14s} -> acc={acc} prec={prec} rec={rec} f1={f1} | TN={tn} FP={fp} FN={fn} TP={tp}")

        json_path = os.path.join(ARTIFACT_DIR, json_file)
        if os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as f:
                ref = json.load(f)
            assert acc == ref["accuracy"] and rec == ref["recall"] and int(tp) == ref["TP"], (
                f"Mismatch untuk {name}: ref={ref}"
            )
            print(f"  [OK] Cocok 100% dengan {json_file}")

    return rf_model, knn_model, scaler


if __name__ == "__main__":
    build_and_save_artifacts()
