import os
from src.data_preprocessing import prepare_data
from src.train_model import train

def main():
    print("🚀 Starting Machine Learning Pipeline...")
    
    # 1. Preprocess data (will generate dummy data if raw is missing)
    print("\nStep 1: Preprocessing Data...")
    prepare_data()
    
    # 2. Train the model
    print("\nStep 2: Training Model...")
    train()
    
    print("\n✅ Pipeline complete! You can now run the Flask app: python app/app.py")

if __name__ == "__main__":
    main()