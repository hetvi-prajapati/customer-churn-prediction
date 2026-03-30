import os
import json
import pandas as pd
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# ==========================================
# SYSTEM INITIALIZATION & MODEL LOADING
# ==========================================
MODEL_PATH = "models/churn_model.pkl"
METRICS_PATH = "models/metrics.json"
METADATA_PATH = "models/model_metadata.json"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    try:
        MODEL_COLUMNS = model.feature_names_in_
    except AttributeError:
        MODEL_COLUMNS = None 
    print("✅ SYSTEM READY: Random Forest Model Loaded Successfully.")
else:
    model = None
    MODEL_COLUMNS = None
    print(f"⚠️ CRITICAL WARNING: Model not found at {MODEL_PATH}. Inference offline.")

# Helper function to safely load live ML pipeline data
def load_system_data(filepath, fallback):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return fallback

# ==========================================
# CORE AI PREDICTION ROUTE (Inference Engine)
# ==========================================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    risk = None
    error = None

    if request.method == "POST":
        if model is None:
            return render_template("index.html", error="Machine Learning Engine is currently offline. Please contact an administrator.")

        try:
            # 1. Extract Telemetry from Enterprise UI
            gender = int(request.form.get("gender", 0))
            senior_citizen = int(request.form.get("senior_citizen", 0))
            partner = int(request.form.get("partner", 0))
            
            internet_service = request.form.get("internet_service", "DSL")
            tech_support = int(request.form.get("tech_support", 0))
            streaming_tv = int(request.form.get("streaming_tv", 0))
            
            contract = int(request.form.get("contract", 0))
            payment_method = request.form.get("payment_method", "Electronic check")
            
            tenure = int(request.form.get("tenure", 0))
            monthly_charges = float(request.form.get("monthly_charges", 0.0))
            
            # Auto-Calculate LTV
            total_charges = tenure * monthly_charges

            # 2. Map directly to One-Hot Encoded ML Features
            data = {
                "SeniorCitizen": senior_citizen,
                "tenure": tenure,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "gender_Male": gender,
                "Partner_Yes": partner,
                
                # Advanced Service Architecture Mappings
                "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
                "InternetService_No": 1 if internet_service == "No" else 0,
                "TechSupport_Yes": tech_support,
                "StreamingTV_Yes": streaming_tv,
                
                # Advanced Financial Mappings
                "Contract_One year": 1 if contract == 1 else 0,
                "Contract_Two year": 1 if contract == 2 else 0,
                "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
                "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
                "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
            }

            # 3. Dynamic Shape Alignment (Bulletproof Failsafe)
            df = pd.DataFrame([data])
            if MODEL_COLUMNS is not None:
                df = df.reindex(columns=MODEL_COLUMNS, fill_value=0)

            # 4. Execute AI Inference
            pred = model.predict(df)[0]
            prob = model.predict_proba(df)[0][1]

            # 5. Format Output for UI
            result = "High Flight Risk ❌" if pred == 1 else "Account Stable ✅"
            risk = f"{round(prob * 100, 2)}%"

        except Exception as e:
            error = f"Pipeline Exception: {str(e)}"

    return render_template("index.html", result=result, risk=risk, error=error)


# ==========================================
# SYSTEM DASHBOARD & GOVERNANCE ROUTES
# ==========================================

@app.route('/overview')
def overview():
    # Read live performance data from evaluate_model.py
    ml_metrics = load_system_data(METRICS_PATH, {"accuracy": "N/A"})
    
    # Dynamic Business Telemetry
    stats = {
        "total_customers": "7,043",
        "current_churn_rate": "26.5%",
        "revenue_at_risk": "$142,000",
        "models_deployed": 1,
        "live_accuracy": ml_metrics.get("accuracy", "N/A")
    }
    return render_template('overview.html', stats=stats)

@app.route('/insights')
def insights():
    return render_template('insights.html')

@app.route('/config')
def config():
    # Read live system data from train_model.py
    metadata = load_system_data(METADATA_PATH, {
        "last_trained": "Offline", 
        "data_shape": "Unknown",
        "hyperparameters": {"n_estimators": 100, "max_depth": 10}
    })
    
    return render_template('config.html', sys_data=metadata)

@app.route('/preferences')
def preferences():
    return render_template('preferences.html')


if __name__ == "__main__":
    # Initialize Local Enterprise Server
    print("🌐 STARTING BETTERVERSE AI WEB SERVER ON PORT 5000...")
    app.run(debug=True, port=5000)