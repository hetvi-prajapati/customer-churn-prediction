import pandas as pd
import joblib
import os

def predict_churn(customer_data, model_path='models/churn_model.pkl'):
    """
    Takes a dictionary of customer data, aligns it with the model's exact expected features,
    and returns the predicted churn outcome and risk probability.
    """
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}. Please train the model first.")
        return None, None

    try:
        # 1. Load the model using joblib
        model = joblib.load(model_path)
        
        # 2. Extract the exact columns the model requires
        try:
            model_columns = model.feature_names_in_
        except AttributeError:
            print("❌ Error: Model is not a standard scikit-learn model with feature names.")
            return None, None

        # 3. Convert the input dictionary to a single-row Pandas DataFrame
        df = pd.DataFrame([customer_data])

        # 4. REAL-WORLD LOGIC: The Feature Alignment Trick
        # Matches dictionary to the model's required columns, filling missing ones with 0.
        df = df.reindex(columns=model_columns, fill_value=0)

        # 5. Make the Prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return prediction, probability

    except Exception as e:
        print(f"❌ Prediction Pipeline Error: {e}")
        return None, None


if __name__ == "__main__":
    # ---------------------------------------------------------
    # CLI Testing: Simulating a real enterprise customer payload
    # ---------------------------------------------------------
    print("==================================================")
    print("🚀 INITIATING TERMINAL PREDICTION TEST...")
    print("==================================================")
    
    # Example: A high-risk customer payload matching our new UI
    sample_customer = {
        "SeniorCitizen": 0,
        "tenure": 5,
        "MonthlyCharges": 85.0,
        "TotalCharges": 425.0,
        "gender_Male": 1,
        "Partner_Yes": 0,
        "InternetService_Fiber optic": 1,
        "TechSupport_Yes": 0,          # Added from new UI
        "StreamingTV_Yes": 1,          # Added from new UI
        "Contract_One year": 0,        # Added from new UI
        "Contract_Two year": 0,        # Added from new UI
        "PaymentMethod_Electronic check": 1
    }
    
    pred, prob = predict_churn(sample_customer)
    
    if pred is not None:
        print("\n------------------------------")
        print("🔍 CUSTOMER CHURN ANALYSIS")
        print("------------------------------")
        
        # Format the output beautifully for the terminal
        status = "High Flight Risk ❌" if pred == 1 else "Secure Account ✅"
        
        print(f"Predicted Outcome  : {status}")
        print(f"Churn Probability  : {prob * 100:.2f}%\n")