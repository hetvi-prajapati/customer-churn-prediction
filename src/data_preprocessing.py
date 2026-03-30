import os
import pandas as pd
import numpy as np

def prepare_data():
    raw_path = 'data/raw/churn.csv'
    processed_path = 'data/processed/clean_churn.csv'
    
    # 1. Check if the raw data actually exists
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"❌ Error: '{raw_path}' not found! "
            "Please ensure you have placed the raw Telco customer CSV inside the 'data/raw/' folder."
        )

    print("--------------------------------------------------")
    print("🚀 INITIATING DATA PREPROCESSING PIPELINE...")
    print("--------------------------------------------------")
    
    df = pd.read_csv(raw_path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns.")

    # 2. Drop unnecessary columns (IDs don't help prediction)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        print("Dropped 'customerID' column.")

    # 3. Handle the 'TotalCharges' blank space issue
    if 'TotalCharges' in df.columns:
        # Replace empty spaces with NaN, convert to float, fill NaNs with 0
        df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        print("Sanitized 'TotalCharges' column.")

    # 4. Convert the Target Variable strictly to 1 and 0
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        print("Mapped 'Churn' target to binary (1/0).")

    # 5. One-Hot Encoding (The Enterprise Fix)
    # We add dtype=int so pandas outputs 1 and 0 instead of True and False.
    # This ensures perfect compatibility with our Flask app.py inputs!
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_clean = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    
    print("Executed One-Hot Encoding on categorical features.")

    # 6. Ensure the processed directory exists
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    # 7. Save the fully cleaned, numeric-only dataset
    df_clean.to_csv(processed_path, index=False)
    
    print("--------------------------------------------------")
    print(f"✅ SUCCESS! Clean data saved to: {processed_path}")
    print(f"📊 Final pipeline shape ready for ML Training: {df_clean.shape}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    prepare_data()