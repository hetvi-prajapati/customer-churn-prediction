import os
import json
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

def evaluate():
    data_path = "data/processed/clean_churn.csv"
    model_path = "models/churn_model.pkl"
    metrics_path = "models/metrics.json"

    # 1. Safety Checks
    if not os.path.exists(data_path):
        print(f"❌ Error: Could not find {data_path}. Please run data_preprocessing.py first.")
        return
        
    if not os.path.exists(model_path):
        print(f"❌ Error: Could not find {model_path}. Please run train_model.py first.")
        return

    print("--------------------------------------------------")
    print("🔍 INITIATING MODEL EVALUATION PROTOCOL...")
    print("--------------------------------------------------")
    
    df = pd.read_csv(data_path)

    if "Churn" not in df.columns:
        print("❌ Error: Target column 'Churn' not found in the dataset.")
        return

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # 2. REAL-WORLD LOGIC: The Test Split
    # We use the exact same random_state=42 and stratify=y as train_model.py
    # This guarantees we are only evaluating on the 20% of data the model has NEVER seen.
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Evaluating strictly on unseen test data ({X_test.shape[0]} rows)...")
    
    # Load the trained model
    model = joblib.load(model_path)

    # Make predictions and get probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 3. Calculate Advanced Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # 4. Print Enterprise Terminal Report
    print("\n" + "="*50)
    print("             MODEL EVALUATION REPORT")
    print("="*50)
    
    print(f"✅ Overall Accuracy : {accuracy * 100:.2f}%")
    print(f"📈 ROC-AUC Score    : {roc_auc:.4f} (Ability to distinguish classes)")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    print("--- Confusion Matrix ---")
    print("               [Predicted Stay]  [Predicted Churn]")
    print(f"[Actual Stay]       {conf_matrix[0][0]}                 {conf_matrix[0][1]}")
    print(f"[Actual Churn]      {conf_matrix[1][0]}                 {conf_matrix[1][1]}")
    print("="*50 + "\n")

    # 5. Connect to Web App: Export Metrics to JSON
    # This allows your Flask app's overview.html to read live stats!
    metrics_data = {
        "accuracy": round(accuracy * 100, 2),
        "roc_auc": round(roc_auc, 4),
        "false_positives": int(conf_matrix[0][1]),
        "false_negatives": int(conf_matrix[1][0])
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"💾 Metrics successfully exported to: {metrics_path}")

if __name__ == "__main__":
    evaluate()