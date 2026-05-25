import os
from src.data_preprocessing import prepare_data
from src.train_model import train
from src.evaluate_model import evaluate

def main():
    print("Starting Machine Learning Pipeline...")

    # 1. Preprocess data
    print("\nStep 1: Preprocessing Data...")
    prepare_data()

    # 2. Train the model
    print("\nStep 2: Training Model...")
    train()

    # 3. Evaluate and export metrics.json
    print("\nStep 3: Evaluating Model & Exporting Metrics...")
    evaluate()

    print("\nPipeline complete! Run: py app/app.py")

if __name__ == "__main__":
    main()