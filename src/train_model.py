import os
import json
import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train(n_estimators=100, max_depth=10, class_weight='balanced'):
    """
    Trains the Random Forest model. 
    Accepts dynamic hyperparameters so it can be controlled via the config.html web UI.
    """
    processed_path = 'data/processed/clean_churn.csv'
    model_path = 'models/churn_model.pkl'
    metadata_path = 'models/model_metadata.json'

    # 1. Safety Check
    if not os.path.exists(processed_path):
        print(f"❌ Error: {processed_path} not found. Run data_preprocessing.py first.")
        return

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print("--------------------------------------------------")
    print("⚙️ INITIATING RANDOM FOREST TRAINING PIPELINE...")
    print("--------------------------------------------------")
    
    df = pd.read_csv(processed_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")

    if 'Churn' not in df.columns:
        print("❌ Error: 'Churn' column missing. Check preprocessing.")
        return

    # 2. Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # 3. Stratified Split
    print("Executing stratified train/test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y 
    )

    # 4. Train with Dynamic Parameters
    print(f"Compiling Model: Trees={n_estimators}, Max Depth={max_depth}, Weight={class_weight}")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,             
        class_weight=class_weight,  
        random_state=42,
        n_jobs=-1 # Uses all CPU cores for faster training
    )
    
    model.fit(X_train, y_train)

    # 5. Quick Evaluation
    print("\n--- Initial Training Report ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 6. Save the Model
    print(f"Saving compiled model to {model_path}...")
    joblib.dump(model, model_path)
    
    # 7. ENTERPRISE UPGRADE: Export Metadata Audit Trail
    # This creates a JSON file that your config.html can read to show LIVE system stats!
    metadata = {
        "last_trained": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        "data_shape": f"({df.shape[0]}, {df.shape[1]})",
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "class_weight": class_weight
        },
        "features_tracked": list(X.columns)
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print("✅ Pipeline execution complete. Model is fully armed and operational.")
    print("--------------------------------------------------")


if __name__ == "__main__":
    # If run from the terminal, it uses the default parameters.
    # Later, your Flask app can call train(n_estimators=200) based on UI input!
    train()