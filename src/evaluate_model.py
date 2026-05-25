# -*- coding: utf-8 -*-
import os
import sys
import json
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, "data", "processed", "clean_churn.csv")
MODEL_PATH   = os.path.join(BASE_DIR, "models", "churn_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")


def evaluate():
    # Safety checks
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Run data_preprocessing.py first.")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Run train_model.py first.")
        return

    print("--------------------------------------------------")
    print("INITIATING MODEL EVALUATION PROTOCOL...")
    print("--------------------------------------------------")

    df = pd.read_csv(DATA_PATH)
    if "Churn" not in df.columns:
        print("Error: Target column 'Churn' not found.")
        return

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Same split as train_model.py — evaluating on the 20% holdout
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Evaluating strictly on unseen test data ({X_test.shape[0]} rows)...")

    model    = joblib.load(MODEL_PATH)
    y_pred   = model.predict(X_test)
    y_prob   = model.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc_auc = None

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Print report
    print("\n" + "=" * 50)
    print("             MODEL EVALUATION REPORT")
    print("=" * 50)
    print(f"Overall Accuracy : {accuracy * 100:.2f}%")
    if roc_auc is not None:
        print(f"ROC-AUC Score    : {roc_auc:.4f}")
    print(f"Precision        : {precision * 100:.2f}%")
    print(f"Recall           : {recall * 100:.2f}%")
    print(f"F1 Score         : {f1 * 100:.2f}%")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("--- Confusion Matrix ---")
    print("               [Predicted Stay]  [Predicted Churn]")
    print(f"[Actual Stay]       {cm[0][0]}                 {cm[0][1]}")
    print(f"[Actual Churn]      {cm[1][0]}                 {cm[1][1]}")
    print("=" * 50 + "\n")

    # MERGE with existing metrics (don't overwrite, only enrich)
    existing = {}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass

    existing.update({
        "accuracy":         round(accuracy * 100, 2),
        "roc_auc":          round(roc_auc, 4) if roc_auc is not None else "N/A",
        "precision":        round(precision * 100, 2),
        "recall":           round(recall * 100, 2),
        "f1":               round(f1 * 100, 2),
        "true_negatives":   int(cm[0][0]),
        "false_positives":  int(cm[0][1]),
        "false_negatives":  int(cm[1][0]),
        "true_positives":   int(cm[1][1]),
        "confusion_matrix": cm.tolist(),
    })

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4)

    print(f"Metrics exported to: {METRICS_PATH}")


if __name__ == "__main__":
    evaluate()