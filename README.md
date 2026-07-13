<div align="center">

# ChurnGuard AI: Enterprise Customer Retention Intelligence 🚀

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.3-black.svg?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.4.2-orange.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Production_Ready-success.svg?style=for-the-badge)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An enterprise-grade, end-to-end Machine Learning web application designed to predict customer churn and provide actionable retention intelligence. Built as a comprehensive Decision Support System (DSS) integrating a Random Forest pipeline with a dynamic, responsive Flask frontend.

</div>

---

## 🌟 Key Features

- **🧠 Advanced ML Pipeline:** Fully automated data preprocessing, stratified train/test splitting, and hyperparameter tuning to handle severe class imbalances.
- **⚡ Predictive Engine:** Utilizes a highly tuned `RandomForestClassifier` to generate real-time churn probabilities.
- **🔗 Dynamic JSON Bridge:** The frontend and backend are completely decoupled. Python ML scripts export live metadata and evaluation metrics to JSON, which the Flask server dynamically renders in the UI.
- **💻 Enterprise Dashboard:** A responsive, dark-mode SaaS interface featuring:
  - Live Chart.js data visualizations.
  - Interactive AI Chatbot assistant with context-aware navigation.
  - CSS-animated Gauge Charts and probability meters.
  - Interactive hyperparameter controls and API key management.
- **🛡️ Defensive Programming:** Implements robust `pandas.reindex` logic to ensure the web form never crashes the ML model, even if input features are missing.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/hetvi-prajapati/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the ML Pipeline (Optional but recommended)
Train the model and generate fresh metric files:
```bash
python main.py
```

### 4. Start the Web Server
Launch the Flask dashboard:
```bash
python app/app.py
```
> Navigate to `http://localhost:5001` in your browser.

---

## 🏗️ System Architecture

```text
CUSTOMER-CHURN-PREDICTION/
│
├── app/
│   ├── app.py                  # Main Flask Web Server & API Routing
│   ├── templates/              # Jinja2 Frontend Architecture
│   │   ├── layout.html         # Master UI Blueprint (Sidebar & CSS)
│   │   ├── index.html          # The Core Prediction Engine
│   │   ├── overview.html       # High-Level Business Telemetry
│   │   ├── insights.html       # Data Visualization & Ledgers
│   │   ├── config.html         # ML Pipeline Control Hub
│   │   └── preferences.html    # System Governance & UX
│   └── static/css/main.css     # Global Design System
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
├── main.py                     # Entry point for ML Pipeline execution
├── requirements.txt            # Python Dependencies
└── README.md                   # System Documentation
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
