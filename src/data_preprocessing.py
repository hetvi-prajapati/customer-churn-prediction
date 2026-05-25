# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH  = os.path.join(BASE_DIR, "data", "raw",       "churn.csv")
PROC_PATH = os.path.join(BASE_DIR, "data", "processed", "clean_churn.csv")


def prepare_data():
    os.makedirs(os.path.dirname(PROC_PATH), exist_ok=True)

    print("=" * 48)
    print("  DATA PREPROCESSING PIPELINE")
    print("=" * 48)

    df = pd.read_csv(RAW_PATH)
    print(f"  Loaded raw data  -> shape: {df.shape}")

    # 1. Drop customerID
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # 2. TotalCharges -- has whitespace-only strings in IBM dataset
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip().replace("", np.nan),
        errors="coerce"
    )
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = (
        df.loc[mask, "tenure"] * df.loc[mask, "MonthlyCharges"]
    )

    # 3. Feature engineering
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"].clip(lower=1))

    # 4. Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    # 5. One-hot encode categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # 6. Drop any remaining NaN rows
    df = df.dropna()

    print(f"  Cleaned data     -> shape: {df.shape}")
    print(f"  Churn rate       -> {df['Churn'].mean()*100:.1f}%")
    print(f"  Features         -> {df.shape[1]-1}")

    df.to_csv(PROC_PATH, index=False)
    print(f"  Saved to         -> {PROC_PATH}")
    print("=" * 48)


if __name__ == "__main__":
    prepare_data()