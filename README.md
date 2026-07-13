<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=800&size=32&pause=1000&color=00D4FF&center=true&vCenter=true&width=800&lines=ChurnGuard+AI;Enterprise+Customer+Retention;Predictive+Machine+Learning+Pipeline;Real-Time+Analytics+Dashboard" alt="Typing SVG" />

**An enterprise-grade, end-to-end Machine Learning web application designed to predict customer churn and provide actionable retention intelligence.**

[![Python](https://img.shields.io/badge/Python-3.9+-00D4FF.svg?style=for-the-badge&logo=python&logoColor=white&labelColor=0f172a)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.3-00D4FF.svg?style=for-the-badge&logo=flask&logoColor=white&labelColor=0f172a)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.4.2-00D4FF.svg?style=for-the-badge&logo=scikit-learn&logoColor=white&labelColor=0f172a)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.2-00D4FF.svg?style=for-the-badge&logo=pandas&logoColor=white&labelColor=0f172a)](https://pandas.pydata.org/)
<br>
[![HTML5](https://img.shields.io/badge/HTML5-00D4FF.svg?style=for-the-badge&logo=html5&logoColor=white&labelColor=0f172a)]()
[![CSS3](https://img.shields.io/badge/CSS3-00D4FF.svg?style=for-the-badge&logo=css3&logoColor=white&labelColor=0f172a)]()
[![JavaScript](https://img.shields.io/badge/JavaScript-00D4FF.svg?style=for-the-badge&logo=javascript&logoColor=white&labelColor=0f172a)]()
<br><br>
[![Status](https://img.shields.io/badge/Status-Production_Ready-00D4FF.svg?style=for-the-badge&labelColor=0f172a)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-00D4FF.svg?style=for-the-badge&labelColor=0f172a)](https://opensource.org/licenses/MIT)

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

## 🛑 CORE ARCHITECTURE (THE PIPELINE)

The system is built on a decoupled architecture where the Machine Learning pipeline operates independently of the Flask web server, connected via JSON metadata bridges.

```mermaid
graph TD
    %% Custom Neon-Cyan & Navy Dark Mode Styling %%
    classDef default fill:#0f172a,stroke:#00D4FF,stroke-width:2px,color:#ffffff,rx:8px,ry:8px;
    classDef accent fill:#00D4FF,stroke:#0f172a,stroke-width:2px,color:#000000,rx:8px,ry:8px,font-weight:bold;
    classDef database fill:#020617,stroke:#00D4FF,stroke-width:2px,color:#00D4FF;

    subgraph Data Pipeline
        A[(Raw Data: churn.csv)]:::database --> B[Data Preprocessing<br/>handling missing, encoding]
        B --> C[Cleaned Data<br/>clean_churn.csv]:::database
    end

    subgraph Machine Learning Engine
        C --> D[Model Training<br/>Random Forest]
        D --> E{Hyperparameter<br/>Tuning}
        E --> F((Trained Model<br/>churn_model.pkl)):::accent
        D --> G[Model Evaluation<br/>Accuracy, ROC-AUC, F1]
        G --> H[/Metrics Bridge<br/>metrics.json/]:::database
    end

    subgraph Web Application Frontend
        F --> I[Flask Backend API]
        H --> I
        I --> J[Jinja2 Dynamic Templates]
        J --> K[Interactive Dashboard<br/>Chart.js & AI Chatbot]:::accent
    end
```

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

<div align="center">
  <p>Built with ❤️ by Hetvi Prajapati</p>
</div>
