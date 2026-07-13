<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=700&size=30&pause=1000&color=00D4FF&center=true&vCenter=true&width=800&lines=ChurnGuard+AI;Enterprise+Customer+Retention;Predictive+Machine+Learning+Pipeline;Real-Time+Analytics+Dashboard" alt="Typing SVG" />

**An enterprise-grade, end-to-end Machine Learning web application designed to predict customer churn and provide actionable retention intelligence.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.3-black.svg?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.4.2-orange.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.2-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![HTML5](https://img.shields.io/badge/HTML5-E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)]()
[![CSS3](https://img.shields.io/badge/CSS3-1572B6.svg?style=for-the-badge&logo=css3&logoColor=white)]()
[![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E.svg?style=for-the-badge&logo=javascript&logoColor=black)]()

[![Status](https://img.shields.io/badge/Status-Production_Ready-success.svg?style=for-the-badge)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

<br />

## 🌟 Executive Summary

ChurnGuard AI is a comprehensive **Decision Support System (DSS)** integrating a powerful Random Forest machine learning pipeline with a dynamic, responsive Flask frontend. It is designed to process raw telecommunications customer data, identify high-risk accounts, and provide business intelligence through an interactive dashboard.

---

## ✨ Core Capabilities

- 🧠 **Advanced ML Pipeline:** Fully automated data preprocessing, stratified train/test splitting, and hyperparameter tuning to handle severe class imbalances.
- ⚡ **Predictive Engine:** Utilizes a highly tuned `RandomForestClassifier` (achieving ~77% Accuracy and 0.83 ROC-AUC) to generate real-time churn probabilities.
- 🔗 **Dynamic JSON Bridge:** The frontend and backend are decoupled. Python ML scripts export live metadata and evaluation metrics to JSON, which the Flask server dynamically renders in the UI.
- 🤖 **Interactive AI Assistant:** A built-in, context-aware Chatbot that can answer questions about model accuracy, navigate the dashboard, and provide quick suggestions.
- 🛡️ **Defensive Programming:** Implements robust `pandas.reindex` logic to ensure the web form never crashes the ML model, even if input features are missing.

---

## 💻 Enterprise Dashboard Features

<details>
<summary><b>🖼️ Click to expand UI Features</b></summary>
<br>

- **Dark-Mode Glassmorphism UI:** A stunning, modern interface with neon-cyan accents.
- **Live Visualizations:** Powered by Chart.js for interactive business telemetry.
- **Audience Insights:** Ranks top 20 highest-risk customers and calculates Lifetime Value (LTV).
- **Model Configuration Hub:** Real-time visibility into ML hyperparameter details and pipeline health.

</details>

---

## 🚀 Quick Start Guide

### 1. Clone the repository
```bash
git clone https://github.com/hetvi-prajapati/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
It is highly recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the ML Pipeline (Optional but recommended)
Train the model and generate fresh metric files based on `data/raw/churn.csv`:
```bash
python main.py
```

### 4. Start the Web Server
Launch the Flask dashboard to interact with the predictive engine:
```bash
python app/app.py
```
> 🌐 Navigate to `http://localhost:5001` in your browser.

---

## 🏗️ System Architecture

```text
CUSTOMER-CHURN-PREDICTION/
├── app/
│   ├── app.py                  # Main Flask Web Server & API Routing
│   ├── templates/              # Jinja2 Frontend Architecture
│   └── static/css/main.css     # Global Design System
│
├── data/
│   ├── raw/                    # Original dataset (e.g., churn.csv)
│   └── processed/              # Cleaned, one-hot encoded numeric data
│
├── models/
│   ├── churn_model.pkl         # Pickled Random Forest Classifier
│   ├── metrics.json            # Live evaluation metrics
│   └── model_metadata.json     # Audit trail for training timestamps
│
├── src/                        # Machine Learning Pipeline
│   ├── data_preprocessing.py   # Cleans data & handles missing values
│   ├── train_model.py          # Executes RF algorithm & exports metadata
│   └── evaluate_model.py       # Validates unseen data & exports metrics
│
├── main.py                     # Entry point for ML Pipeline execution
├── requirements.txt            # Python Dependencies
└── LICENSE                     # MIT License
```

---

<div align="center">
  <p>Built with ❤️ by Hetvi Prajapati</p>
</div>
