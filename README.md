# BetterVerse AI: Enterprise Customer Retention Intelligence 🚀

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.3-lightgrey.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.2-orange.svg)
![Status](https://img.shields.io/badge/Status-Production_Ready-success.svg)

An enterprise-grade, end-to-end Machine Learning web application designed to predict customer churn and provide actionable retention intelligence. Built as a comprehensive Decision Support System (DSS) integrating a Random Forest pipeline with a dynamic, responsive Flask frontend.

## 🌟 Key Features

- **Advanced ML Pipeline:** Fully automated data preprocessing, stratified train/test splitting, and hyperparameter tuning to handle severe class imbalances.
- **Predictive Engine:** Utilizes a highly tuned `RandomForestClassifier` to generate real-time churn probabilities.
- **Dynamic JSON Bridge:** The frontend and backend are completely decoupled. Python ML scripts export live metadata and evaluation metrics to JSON, which the Flask server dynamically renders in the UI.
- **Enterprise Dashboard:** A 5-page, responsive "Light White" SaaS interface featuring:
  - Live Chart.js data visualizations.
  - CSS-animated Gauge Charts and probability meters.
  - Interactive hyperparameter controls and API key management.
- **Defensive Programming:** Implements robust `pandas.reindex` logic to ensure the web form never crashes the ML model, even if input features are missing.

## 🏗️ System Architecture

```text
CUSTOMER-CHURN-PREDICTION/
│
├── app.py                      # Main Flask Web Server & API Routing
├── requirements.txt            # Python Dependencies
├── README.md                   # System Documentation
│
├── data/
│   ├── raw/                    # Original dataset (e.g., churn.csv)
│   └── processed/              # Cleaned, one-hot encoded numeric data
│
├── models/
│   ├── churn_model.pkl         # Pickled Random Forest Classifier
│   ├── metrics.json            # Live ROC-AUC and Accuracy metrics
│   └── model_metadata.json     # Audit trail for training timestamps
│
├── src/                        # Machine Learning Pipeline
│   ├── data_preprocessing.py   # Cleans data & handles missing values
│   ├── train_model.py          # Executes RF algorithm & exports metadata
│   └── evaluate_model.py       # Validates unseen data & exports metrics
│
└── templates/                  # Jinja2 Frontend Architecture
    ├── layout.html             # Master UI Blueprint (Sidebar & CSS)
    ├── index.html              # The Core Prediction Engine
    ├── overview.html           # High-Level Business Telemetry
    ├── insights.html           # Data Visualization & Ledgers
    ├── config.html             # ML Pipeline Control Hub
    └── preferences.html        # System Governance & UX
```
