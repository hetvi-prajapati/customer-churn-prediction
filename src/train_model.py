# -*- coding: utf-8 -*-
import os
import sys
import json
import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score,
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_PATH     = os.path.join(BASE_DIR, "data", "processed", "clean_churn.csv")
MODEL_PATH    = os.path.join(BASE_DIR, "models", "churn_model.pkl")
METRICS_PATH  = os.path.join(BASE_DIR, "models", "metrics.json")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")


def train(n_estimators=200, max_depth=12, class_weight="balanced"):
    if not os.path.exists(PROC_PATH):
        print(f"ERROR: {PROC_PATH} not found -- run data_preprocessing first.")
        return

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    print("=" * 48)
    print("  RANDOM FOREST TRAINING PIPELINE")
    print("=" * 48)

    df = pd.read_csv(PROC_PATH)
    print(f"  Dataset loaded   -> {df.shape[0]:,} rows x {df.shape[1]} cols")

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train size       -> {len(X_train):,}")
    print(f"  Test size        -> {len(X_test):,}")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    print(f"  Fitting model    -> trees={n_estimators}, depth={max_depth}")
    model.fit(X_train, y_train)

    # Evaluation
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc       = round(accuracy_score(y_test, y_pred) * 100, 2)
    roc_auc   = round(roc_auc_score(y_test, y_proba), 4)
    precision = round(precision_score(y_test, y_pred, zero_division=0) * 100, 2)
    recall    = round(recall_score(y_test, y_pred, zero_division=0) * 100, 2)
    f1        = round(f1_score(y_test, y_pred, zero_division=0) * 100, 2)
    cm        = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n  Accuracy         -> {acc}%")
    print(f"  ROC-AUC          -> {roc_auc}")
    print(f"  Precision        -> {precision}%")
    print(f"  Recall           -> {recall}%")
    print(f"  F1 Score         -> {f1}%")
    print(f"\n{classification_report(y_test, y_pred)}")

    joblib.dump(model, MODEL_PATH)
    print(f"  Model saved      -> {MODEL_PATH}")

    metrics = {
        "accuracy":  acc,
        "roc_auc":   roc_auc,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "confusion_matrix": cm,
        "false_positives":  cm[0][1],
        "false_negatives":  cm[1][0],
        "true_positives":   cm[1][1],
        "true_negatives":   cm[0][0],
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"  Metrics saved    -> {METRICS_PATH}")

    metadata = {
        "last_trained":    datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        "data_shape":      f"({df.shape[0]:,}, {df.shape[1]})",
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth":    max_depth,
            "class_weight": class_weight,
            "test_split":   "80/20 stratified",
            "random_state": 42,
        },
        "features_tracked": list(X.columns),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print("=" * 48)
    print("  Pipeline complete. Model armed & operational.")
    print("=" * 48)


if __name__ == "__main__":
    train()