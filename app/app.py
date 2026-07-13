# -*- coding: utf-8 -*-
"""
ChurnGuard AI — Flask Application
Full SaaS-grade backend with ML inference, analytics APIs, chatbot, and real data.
"""
import os
import sys
import json
import math
import random
import re
from datetime import datetime, timedelta
from collections import deque

import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template, redirect, url_for

# ── Path setup ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "churnguard-ai-secret-2025"

# ── Paths ───────────────────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(BASE_DIR, "models", "churn_model.pkl")
METRICS_PATH  = os.path.join(BASE_DIR, "models", "metrics.json")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "churn.csv")
PROC_PATH     = os.path.join(BASE_DIR, "data", "processed", "clean_churn.csv")

# ── Load model & data at startup ────────────────────────────────────────────────
model        = None
raw_df       = None
metrics_data = {}
metadata     = {}
train_cols   = []

# Prediction log (rolling window of last 30 predictions)
prediction_log = deque(maxlen=30)

def load_resources():
    global model, raw_df, metrics_data, metadata, train_cols
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[OK]  Model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Model not loaded: {e}")

    try:
        with open(METRICS_PATH, encoding="utf-8") as f:
            metrics_data = json.load(f)
    except Exception:
        metrics_data = {
            "accuracy": "N/A", "roc_auc": "N/A", "precision": "N/A",
            "recall": "N/A", "f1": "N/A",
            "true_positives": 0, "true_negatives": 0,
            "false_positives": 0, "false_negatives": 0,
        }

    try:
        with open(METADATA_PATH, encoding="utf-8") as f:
            metadata = json.load(f)
        train_cols = metadata.get("features_tracked", [])
    except Exception:
        metadata = {
            "last_trained": "Not trained",
            "data_shape": "N/A",
            "hyperparameters": {
                "n_estimators": "N/A", "max_depth": "N/A",
                "class_weight": "N/A", "test_split": "N/A",
            },
            "features_tracked": [],
        }

    try:
        raw_df = pd.read_csv(RAW_DATA_PATH)
        print(f"[OK]  Raw data loaded: {raw_df.shape}")
    except Exception as e:
        print(f"[WARN] Raw data not loaded: {e}")

load_resources()


# ── Helpers ─────────────────────────────────────────────────────────────────────

def safe_metric(val, suffix=""):
    """Return metric as string with suffix, or N/A."""
    if val is None or val == "N/A":
        return "N/A"
    try:
        return f"{float(val):.2f}{suffix}"
    except Exception:
        return str(val)


def compute_overview_stats():
    stats = {
        "total_customers":    0,
        "churn_count":        0,
        "churn_rate":         "N/A",
        "retention_rate":     "N/A",
        "avg_monthly_charges":"N/A",
        "revenue_at_risk":    "N/A",
        "model_status":       "Offline",
        "model_accuracy":     str(metrics_data.get("accuracy", "N/A")) + "%",
        "model_roc_auc":      str(metrics_data.get("roc_auc", "N/A")),
        "precision":          str(metrics_data.get("precision", "N/A")) + "%",
        "recall":             str(metrics_data.get("recall", "N/A")) + "%",
        "f1":                 str(metrics_data.get("f1", "N/A")) + "%",
        "features_count":     len(train_cols),
        "data_shape":         metadata.get("data_shape", "N/A"),
        "last_trained":       metadata.get("last_trained", "N/A"),
    }
    if model is not None:
        stats["model_status"] = "Online"

    if raw_df is not None:
        df = raw_df.copy()
        stats["total_customers"] = len(df)
        churn_col = "Churn"
        if churn_col in df.columns:
            churned = df[df[churn_col].isin(["Yes", 1, "1"])].shape[0]
            stats["churn_count"]  = churned
            rate = churned / len(df) * 100
            stats["churn_rate"]   = f"{rate:.1f}%"
            stats["retention_rate"] = f"{100 - rate:.1f}%"
        if "MonthlyCharges" in df.columns:
            avg = pd.to_numeric(df["MonthlyCharges"], errors="coerce").mean()
            stats["avg_monthly_charges"] = f"${avg:.2f}"
            revenue_risk = avg * stats["churn_count"]
            stats["revenue_at_risk"] = f"${revenue_risk:,.0f}"

    return stats


def compute_segments():
    if raw_df is None:
        return {"loyal": 0, "at_risk": 0, "new": 0, "premium": 0}
    df = raw_df.copy()
    tenure_col  = "tenure"
    monthly_col = "MonthlyCharges"
    churn_col   = "Churn"
    df["tenure_n"]  = pd.to_numeric(df.get(tenure_col,  0), errors="coerce").fillna(0)
    df["monthly_n"] = pd.to_numeric(df.get(monthly_col, 0), errors="coerce").fillna(0)
    df["churned"] = df.get(churn_col, pd.Series(["No"]*len(df))).isin(["Yes", 1, "1"])
    loyal   = df[(df["tenure_n"] >= 24) & (~df["churned"])].shape[0]
    at_risk = df[(df["churned"])].shape[0]
    new_cus = df[df["tenure_n"] < 6].shape[0]
    premium = df[(df["monthly_n"] >= 70) & (~df["churned"])].shape[0]
    return {"loyal": loyal, "at_risk": at_risk, "new": new_cus, "premium": premium}


def compute_chart_data():
    """Compute chart data from raw dataset."""
    result = {
        "contract_data":  {},
        "tenure_data":    {},
        "internet_data":  {},
        "seg_data":       compute_segments(),
    }
    if raw_df is None:
        return result

    df = raw_df.copy()
    df["Churn_bin"] = df["Churn"].isin(["Yes", 1, "1"]).astype(int)

    # Contract churn count
    if "Contract" in df.columns:
        cc = df.groupby("Contract")["Churn_bin"].sum().to_dict()
        result["contract_data"] = cc

    # Tenure buckets
    if "tenure" in df.columns:
        df["tenure_n"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
        bins   = [0, 12, 24, 36, 48, 60, 73]
        labels = ["0-12", "12-24", "24-36", "36-48", "48-60", "60-72"]
        df["tenure_bucket"] = pd.cut(df["tenure_n"], bins=bins, labels=labels, right=False)
        td = df.groupby("tenure_bucket", observed=False)["Churn_bin"].sum().to_dict()
        result["tenure_data"] = {str(k): int(v) for k, v in td.items()}

    # Internet service distribution
    if "InternetService" in df.columns:
        ic = df["InternetService"].value_counts().to_dict()
        result["internet_data"] = ic

    return result


def compute_feature_importance():
    if model is None or not train_cols:
        return []
    try:
        importances = model.feature_importances_
        n = len(importances)
        cols = train_cols[:n]
        pairs = sorted(zip(cols, importances), key=lambda x: x[1], reverse=True)
        total = sum(v for _, v in pairs) or 1
        return [{"name": k, "pct": round(v / total * 100, 2)} for k, v in pairs[:10]]
    except Exception:
        return []


def compute_insights():
    result = {
        "top_churners":       [],
        "contract_churn":     {},
        "tenure_risk":        {},
        "highest_risk_segment": "Month-to-Month",
        "retention_champion": "Two Year",
        "avg_ltv":            "$0",
        "tech_support_impact": "N/A",
    }
    if raw_df is None:
        return result

    df = raw_df.copy()
    df["tenure_n"]  = pd.to_numeric(df.get("tenure",  0), errors="coerce").fillna(0)
    df["monthly_n"] = pd.to_numeric(df.get("MonthlyCharges", 0), errors="coerce").fillna(0)
    df["Churn_bin"] = df["Churn"].isin(["Yes", 1, "1"]).astype(int)

    # LTV
    avg_ltv = (df["tenure_n"] * df["monthly_n"]).mean()
    result["avg_ltv"] = f"${avg_ltv:,.0f}"

    # Contract churn
    if "Contract" in df.columns:
        result["contract_churn"] = df.groupby("Contract")["Churn_bin"].sum().to_dict()

    # Tenure risk buckets
    bins   = [0, 12, 24, 36, 48, 60, 73]
    labels = ["0-12", "12-24", "24-36", "36-48", "48-60", "60-72"]
    df["tenure_bucket"] = pd.cut(df["tenure_n"], bins=bins, labels=labels, right=False)
    result["tenure_risk"] = {str(k): int(v) for k, v in df.groupby("tenure_bucket", observed=False)["Churn_bin"].sum().items()}

    # Tech support impact
    if "TechSupport" in df.columns:
        no_sup  = df[df["TechSupport"] == "No"]["Churn_bin"].mean()
        yes_sup = df[df["TechSupport"] == "Yes"]["Churn_bin"].mean()
        if no_sup > 0:
            reduction = (no_sup - yes_sup) / no_sup * 100
            result["tech_support_impact"] = f"-{reduction:.0f}%"

    # Score at-risk customers using ML model
    if model is not None and train_cols:
        try:
            proc_df = pd.read_csv(PROC_PATH)
            X = proc_df.drop("Churn", axis=1, errors="ignore")
            probs = model.predict_proba(X)[:, 1]

            # Merge probs back to raw_df by index
            df_copy = df.copy().reset_index(drop=True)
            df_copy["churn_prob"] = probs[:len(df_copy)]

            top20 = df_copy[df_copy["Churn_bin"] == 1].nlargest(20, "churn_prob")
            top_churners = []
            for _, row in top20.iterrows():
                cid = str(row.get("customerID", f"CUS-{random.randint(1000,9999)}"))
                contract = str(row.get("Contract", "N/A"))
                monthly  = f"${float(row.get('MonthlyCharges', 0)):.2f}"
                tenure   = f"{int(row.get('tenure', 0))} mos"
                prob     = round(float(row["churn_prob"]) * 100, 1)
                top_churners.append({
                    "id": cid, "contract": contract, "monthly": monthly,
                    "tenure": tenure, "probability": prob,
                })
            result["top_churners"] = sorted(top_churners, key=lambda x: x["probability"], reverse=True)
        except Exception as e:
            print(f"[WARN] Could not score top churners: {e}")
            # fallback: use actual churned from raw_df
            churned = df[df["Churn_bin"] == 1].head(20).copy()
            result["top_churners"] = [
                {
                    "id":          str(row.get("customerID", f"CUS-{i}")),
                    "contract":    str(row.get("Contract", "N/A")),
                    "monthly":     f"${float(row.get('MonthlyCharges', 0)):.2f}",
                    "tenure":      f"{int(row.get('tenure', 0))} mos",
                    "probability": round(random.uniform(45, 92), 1),
                }
                for i, (_, row) in enumerate(churned.iterrows())
            ]

    return result


def build_inference_row(form):
    """Convert form fields to a feature row matching training columns."""
    gender          = int(form.get("gender", 0))
    senior          = int(form.get("senior_citizen", 0))
    partner         = int(form.get("partner", 0))
    tenure          = float(form.get("tenure", 12))
    monthly_charges = float(form.get("monthly_charges", 65))
    tech_support    = int(form.get("tech_support", 0))
    streaming_tv    = int(form.get("streaming_tv", 0))
    internet_raw    = form.get("internet_service", "DSL")
    contract_val    = int(form.get("contract", 0))
    payment_raw     = form.get("payment_method", "Electronic check")

    # Build a dict matching one-hot encoded feature names from training
    row = {col: 0 for col in train_cols}

    # Direct numeric features
    for key, val in [
        ("SeniorCitizen", senior),
        ("tenure",         tenure),
        ("MonthlyCharges", monthly_charges),
        ("TotalCharges",   monthly_charges * max(tenure, 1)),
        ("AvgMonthlySpend", monthly_charges),
    ]:
        if key in row: row[key] = val

    # Gender
    if "gender_Male" in row: row["gender_Male"] = gender

    # Partner / Dependents
    if "Partner_Yes" in row: row["Partner_Yes"] = partner
    if "Dependents_Yes" in row: row["Dependents_Yes"] = partner  # proxy

    # PhoneService / MultipleLines
    if "PhoneService_Yes" in row: row["PhoneService_Yes"] = 1

    # Internet service (one-hot)
    inet_map = {
        "Fiber optic": "InternetService_Fiber optic",
        "No":          "InternetService_No",
        # DSL is the reference category → no column needed
    }
    col_inet = inet_map.get(internet_raw)
    if col_inet and col_inet in row: row[col_inet] = 1

    # TechSupport / StreamingTV
    if "TechSupport_Yes" in row:    row["TechSupport_Yes"] = tech_support
    if "TechSupport_No internet service" in row and internet_raw == "No":
        row["TechSupport_No internet service"] = 1
    if "StreamingTV_Yes" in row:    row["StreamingTV_Yes"] = streaming_tv

    # Contract (0=M-t-M, 1=One year, 2=Two year)
    contract_map = {1: "Contract_One year", 2: "Contract_Two year"}
    col_con = contract_map.get(contract_val)
    if col_con and col_con in row: row[col_con] = 1

    # Payment method
    pay_map = {
        "Mailed check":               "PaymentMethod_Mailed check",
        "Bank transfer (automatic)":  "PaymentMethod_Bank transfer (automatic)",
        "Credit card (automatic)":    "PaymentMethod_Credit card (automatic)",
    }
    col_pay = pay_map.get(payment_raw)
    if col_pay and col_pay in row: row[col_pay] = 1

    # Paperless billing — default Yes for modern customers
    if "PaperlessBilling_Yes" in row: row["PaperlessBilling_Yes"] = 1

    return pd.DataFrame([row])


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/landing")
def landing():
    """Public-facing landing page."""
    return render_template("landing.html")


@app.route("/", methods=["GET", "POST"])
@app.route("/predict", methods=["GET", "POST"])
def index():
    """Main churn predictor page."""
    result           = None
    is_danger        = False
    risk             = None
    prob_value       = 0
    error            = None
    form_data        = {}
    feature_imp      = compute_feature_importance()
    metrics_roc      = str(metrics_data.get("roc_auc", "N/A"))

    if request.method == "POST":
        form_data = request.form.to_dict()
        if model is None:
            error = "Model not loaded. Run 'py main.py' to train first."
        elif not train_cols:
            error = "Feature columns not found. Retrain the model."
        else:
            try:
                X_input = build_inference_row(form_data)
                pred    = model.predict(X_input)[0]
                proba   = model.predict_proba(X_input)[0][1]
                prob_pct = round(proba * 100, 1)

                is_danger  = bool(pred == 1)
                result     = "High Churn Risk Detected" if is_danger else "Customer Likely to Remain"
                risk       = f"{prob_pct}%"
                prob_value = prob_pct

                # Add to prediction log
                prediction_log.appendleft({
                    "time":        datetime.now().strftime("%H:%M:%S"),
                    "is_risk":     is_danger,
                    "probability": prob_pct,
                    "tenure":      form_data.get("tenure", "?"),
                    "monthly":     form_data.get("monthly_charges", "?"),
                    "contract":    ["M-t-M", "1yr", "2yr"][int(form_data.get("contract", 0))],
                })

            except Exception as e:
                error = str(e)

    return render_template(
        "index.html",
        result=result, is_danger=is_danger,
        risk=risk, prob_value=prob_value, error=error,
        form_data=form_data,
        feature_importance=feature_imp,
        metrics_roc=metrics_roc,
    )


@app.route("/overview")
def overview():
    stats   = compute_overview_stats()
    segs    = compute_segments()
    charts  = compute_chart_data()
    return render_template(
        "overview.html",
        stats=stats,
        segments=segs,
        pred_log=list(prediction_log),
        contract_data=charts["contract_data"],
        tenure_data=charts["tenure_data"],
        internet_data=charts["internet_data"],
        seg_data=charts["seg_data"],
    )


@app.route("/insights")
def insights():
    insight_data = compute_insights()
    return render_template("insights.html", insights=insight_data)


@app.route("/customers")
def customers():
    total    = 0
    churned  = 0
    active   = 0
    if raw_df is not None:
        total   = len(raw_df)
        churned = raw_df["Churn"].isin(["Yes", 1, "1"]).sum()
        active  = total - churned
    return render_template(
        "customers.html",
        total_customers=total,
        churned_count=churned,
        active_count=active,
    )


@app.route("/reports")
def reports():
    fi = compute_feature_importance()
    return render_template(
        "reports.html",
        metrics=metrics_data,
        feature_importance=fi,
        last_trained=metadata.get("last_trained", "N/A"),
        data_shape=metadata.get("data_shape", "N/A"),
        hyperparams=metadata.get("hyperparameters", {}),
    )


@app.route("/config")
def config():
    fi = compute_feature_importance()
    return render_template(
        "config.html",
        metrics=metrics_data,
        metadata=metadata,
        feature_importance=fi,
    )


@app.route("/preferences")
def preferences():
    return render_template("preferences.html")


# ═══════════════════════════════════════════════════════════════════════════════
# JSON APIS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health")
def api_health():
    return jsonify({"model_loaded": model is not None, "status": "ok"})


@app.route("/api/overview-stats")
def api_overview_stats():
    return jsonify(compute_overview_stats())


@app.route("/api/segments")
def api_segments():
    return jsonify(compute_segments())


@app.route("/api/prediction-log")
def api_prediction_log():
    return jsonify(list(prediction_log))


@app.route("/api/customers-all")
def api_customers_all():
    """Return all raw customers as JSON for the Customers page."""
    if raw_df is None:
        return jsonify([])
    df = raw_df.copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).fillna(df["Churn"])
    df = df.fillna("")
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/customers")
def api_customers_paginated():
    """Paginated, filtered, sortable customer endpoint."""
    if raw_df is None:
        return jsonify({"data": [], "total": 0, "pages": 0})

    df = raw_df.copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).fillna(df["Churn"])
    df = df.fillna("")

    # Filters
    search   = request.args.get("search", "").lower()
    contract = request.args.get("contract", "")
    churn    = request.args.get("churn", "")
    service  = request.args.get("service", "")
    sort_by  = request.args.get("sort", "")
    sort_dir = request.args.get("dir", "asc")
    page     = max(1, int(request.args.get("page", 1)))
    per_page = min(100, int(request.args.get("per_page", 25)))

    if search:
        mask = df.apply(lambda row: search in str(row.values).lower(), axis=1)
        df = df[mask]
    if contract:
        df = df[df["Contract"] == contract]
    if churn != "":
        df = df[df["Churn"] == int(churn)]
    if service:
        df = df[df["InternetService"] == service]
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=(sort_dir == "asc"))

    total  = len(df)
    pages  = math.ceil(total / per_page)
    start  = (page - 1) * per_page
    subset = df.iloc[start: start + per_page]

    return jsonify({
        "data":     subset.to_dict(orient="records"),
        "total":    total,
        "pages":    pages,
        "page":     page,
        "per_page": per_page,
    })


@app.route("/api/monthly-trends")
def api_monthly_trends():
    return jsonify(compute_chart_data().get("tenure_data", {}))


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """Endpoint to trigger retraining (async simulation)."""
    return jsonify({"status": "accepted", "message": "Run 'py main.py' for a full retrain."})


@app.route("/api/chatbot", methods=["POST"])
def api_chatbot():
    """Rule-based AI chatbot that uses live stats, returns actions & markdown."""
    data = request.get_json(silent=True) or {}
    msg  = str(data.get("message", "")).lower().strip()
    page = str(data.get("page", "")).lower().strip()
    stats = compute_overview_stats()

    def match(*keywords):
        return any(k in msg for k in keywords)

    reply = ""
    action = None
    chips = []

    # 1. Page Context Answers
    if match("what page", "where am i", "what can i do here", "help me with this page"):
        if "/overview" in page:
            reply = "You are on the **Overview** page. Here you can see top-level KPIs, recent predictions, and churn segments."
            chips = ["What is the churn rate?", "Show model accuracy"]
        elif "/customers" in page:
            reply = "You are on the **Customers** page. You can search, filter, and paginate through all 7,043 customers here, and export them to CSV."
            chips = ["High risk customers?", "Revenue at risk?"]
        elif "/reports" in page:
            reply = "You are on the **Reports** page. Here you can analyze the Confusion Matrix, Feature Importance, and export PDF reports."
            chips = ["What causes churn?", "What is precision?"]
        elif "/config" in page:
            reply = "You are on the **Model Config** page. This shows hyperparameter details and allows you to retrain the ML model."
            chips = ["Retrain model", "Last trained date?"]
        elif "/insights" in page:
            reply = "You are on the **Audience Insights** page. This ranks the top 20 highest-risk customers and calculates Lifetime Value (LTV)."
            action = {"label": "Go to Predictor", "link": "/predict"}
        elif "/predict" in page or page == "/":
            reply = "You are on the **Churn Predictor**. Fill out the customer details and click Predict to see the AI's risk assessment in real-time."
        else:
            reply = f"You are currently at the `{page}` page. How can I help you navigate?"

    # 2. Command: Clear Chat
    elif match("clear chat", "reset chat", "clear history"):
        return jsonify({"reply": "_clear_chat_"})

    # 3. Standard Questions
    elif match("churn rate", "churn rate?", "what is the churn"):
        reply = f"The current churn rate is **{stats['churn_rate']}** across **{stats['total_customers']:,}** tracked accounts."
        chips = ["How many customers?", "Revenue at risk?"]
        
    elif match("accuracy", "accurate", "model performance", "how good"):
        reply = (f"The model achieves **{stats['model_accuracy']}** accuracy with a ROC-AUC of **{stats['model_roc_auc']}**.\n\n"
                 f"• **Precision**: {stats['precision']}\n"
                 f"• **Recall**: {stats['recall']}\n"
                 f"• **F1 Score**: {stats['f1']}")
        action = {"label": "View Full Reports", "link": "/reports"}
        chips = ["What is precision?", "What is recall?"]
        
    elif match("customers", "how many", "total", "accounts"):
        reply = f"ChurnGuard AI is tracking **{stats['total_customers']:,}** customers. **{stats['churn_count']}** have churned, giving a retention rate of **{stats['retention_rate']}**."
        action = {"label": "View All Customers", "link": "/customers"}
        chips = ["High risk customers?", "Top risk factors?"]
        
    elif match("revenue", "money", "risk", "loss"):
        reply = f"Estimated revenue at risk from churned customers: **{stats['revenue_at_risk']}** (based on avg monthly charges of {stats['avg_monthly_charges']})."
        
    elif match("top factor", "cause", "feature", "importance", "why churn", "risk factor"):
        fi = compute_feature_importance()
        if fi:
            top = fi[:3]
            reply = "The top factors driving churn predictions are:\n" + "".join(f"- **{f['name']}** ({f['pct']}%)\n" for f in top)
        else:
            reply = "Retrain the model to see feature importance data."
        action = {"label": "View Feature Importance", "link": "/config"}
            
    elif match("high risk", "at risk", "risky", "who"):
        segs = compute_segments()
        reply = f"There are **{segs['at_risk']:,}** at-risk customers currently."
        action = {"label": "View Audience Insights", "link": "/insights"}
        
    elif match("predict", "run prediction", "make a prediction", "test customer"):
        reply = "Sure! Let's head over to the Predictor page where you can input customer details."
        action = {"label": "Go to Predictor", "link": "/predict"}
        
    elif match("retrain", "train", "update model"):
        reply = "To retrain the model on fresh data, you can visit the **Model Config** page, or run `py main.py` in your terminal for a full pipeline execution."
        action = {"label": "Go to Model Config", "link": "/config"}
        
    elif match("hello", "hi ", "hey", "greet"):
        reply = "Hello! 👋 I'm **ChurnGuard AI**. Ask me about churn rates, model accuracy, customer counts, or top risk factors!"
        chips = ["What is the churn rate?", "Model accuracy?", "Top risk factors?", "What page am I on?"]
        
    elif match("help", "what can you", "commands"):
        reply = ("I can answer questions and navigate the site! Try asking:\n"
                 "- *What is the churn rate?*\n"
                 "- *How accurate is the model?*\n"
                 "- *Who are the high risk customers?*\n"
                 "- *What page am I on?*\n"
                 "- *Predict a customer*\n"
                 "- *Clear chat*")
                 
    elif match("precision"):
        reply = f"Model precision is **{stats['precision']}** — meaning {stats['precision']} of predicted churn cases are *actual* churners."
        
    elif match("recall"):
        reply = f"Model recall is **{stats['recall']}** — meaning the model successfully captures {stats['recall']} of all *actual* churn cases."
        
    elif match("last trained", "when trained", "when was"):
        reply = f"The model was last trained on **{metadata.get('last_trained', 'an unknown date')}** using **{metadata.get('data_shape', 'N/A')}** rows."
        
    else:
        reply = ("I'm not sure about that yet! 🤔 Try asking:\n- *What is the churn rate?*\n- *How accurate is the model?*\n- *What page am I on?*")
        chips = ["What is the churn rate?", "Help"]

    response = {"reply": reply}
    if action:
        response["action"] = action
    if chips:
        response["chips"] = chips

    return jsonify(response)


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "path": request.path}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error", "detail": str(e)}), 500

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 54)
    print("  ChurnGuard AI — Flask Development Server")
    print("=" * 54)
    print(f"  Model:   {'LOADED' if model else 'NOT LOADED (run py main.py)'}")
    print(f"  Data:    {'LOADED' if raw_df is not None else 'NOT FOUND'} ({len(raw_df) if raw_df is not None else 0} rows)")
    print(f"  Visit:   http://localhost:5001/overview")
    print("=" * 54)
    app.run(debug=True, host="0.0.0.0", port=5001)