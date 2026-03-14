# Fraud Detection System

> End-to-end real-time fraud detection pipeline — from raw transaction data to a deployed REST API with explainability and drift monitoring.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange) ![FastAPI](https://img.shields.io/badge/API-FastAPI-green) ![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue) ![Docker](https://img.shields.io/badge/Deploy-Docker-blue) ![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

## Problem Statement

Credit card fraud costs the global financial industry over $30 billion annually. Traditional rule-based systems generate too many false positives (blocking legitimate customers) while missing novel fraud patterns. This project builds a machine learning system that scores transactions in real time, explains its decisions, and monitors itself for model degradation — the way production fraud systems actually work at financial institutions.

---

## Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
Real anonymised credit card transactions from European cardholders (September 2013).

| Metric | Value |
|---|---|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 (0.1727%) |
| Legitimate transactions | 284,315 |
| Features | 30 (V1–V28 PCA + Time + Amount) |
| Missing values | None |
| Time period | 2 days |

### Class imbalance
The dataset is severely imbalanced — **only 1 in 579 transactions is fraudulent**. This means:
- A naive model predicting all-legitimate achieves 99.83% accuracy but catches zero fraud
- Standard accuracy is a useless metric for this problem
- We evaluate using **AUC-PR (Precision-Recall)** and **F1-score**, not accuracy
- Training requires imbalance handling (SMOTE + class weights)

### Amount distribution

| | Fraud | Legitimate |
|---|---|---|
| Mean | $122.21 | $88.29 |
| Median | $9.25 | $22.00 |
| Max | $2,125.87 | $25,691.16 |
| Std | $256.68 | $250.11 |

Key finding: fraudulent transactions cluster heavily at low amounts (median $9.25 vs $22.00 for legitimate). Fraudsters test stolen cards with small transactions before attempting larger ones.

---

## Architecture

```
Transaction Source (CSV / Kafka)
        │
        ▼
Data Ingestion & Validation
        │
        ▼
Feature Engineering Pipeline      ← SMOTE, RobustScaler, derived features
        │
        ▼
ML Model Training                  ← XGBoost + Optuna hyperparameter search
        │
        ▼
Model Evaluation                   ← AUC-PR, F1, SHAP explainability
        │
        ▼
MLflow Model Registry              ← Versioning, experiment tracking
        │
        ▼
FastAPI REST Endpoint              ← /predict returns score + SHAP explanation
        │
        ▼
Streamlit Dashboard                ← Live alerts, KPIs, SHAP plots
        │
        ▼
Drift Monitoring (Evidently AI)    ← Automated weekly reports
        │
        ▼
CI/CD → Docker → GCP Cloud Run     ← Auto-deploy on every push to main
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | Pandas, NumPy, scikit-learn |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Modelling | XGBoost, LightGBM, Random Forest |
| Hyperparameter tuning | Optuna |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| Data versioning | DVC |
| API serving | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit |
| Drift monitoring | Evidently AI |
| Containerisation | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Cloud deployment | GCP Cloud Run |

---

## Project Structure

```
fraud-detection-system/
├── data/
│   ├── raw/               # creditcard.csv (not tracked — see .gitignore)
│   └── processed/         # train/test splits
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory data analysis ← YOU ARE HERE
│   ├── 02_features.ipynb   # Feature engineering
│   └── 03_modelling.ipynb  # Model training & evaluation
├── src/
│   ├── features.py         # FeaturePipeline class
│   ├── train.py            # Training script with MLflow logging
│   ├── evaluate.py         # Metrics + SHAP plots
│   └── utils.py
├── api/
│   ├── main.py             # FastAPI app
│   ├── schemas.py          # Pydantic request/response models
│   └── predictor.py        # Model loading from MLflow registry
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── tests/
│   ├── test_features.py
│   └── test_api.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml
├── config.yaml
├── requirements.txt
└── README.md
```

---

## EDA Key Findings

From `notebooks/01_eda.ipynb`:

1. **Severe class imbalance** — 0.1727% fraud rate makes accuracy meaningless as a metric
2. **Fraud clusters at low amounts** — median fraud amount ($9.25) is less than half the median legitimate amount ($22.00), consistent with card-testing behaviour
3. **High-signal features** — V17, V14, V12, V10 show the strongest correlation with the fraud label across the PCA-transformed feature space
4. **Time-of-day patterns** — fraud rate is elevated during late night / early morning hours; engineered as a binary feature `is_night`
5. **No missing data** — dataset requires no imputation

---

## Getting Started

### Prerequisites
```bash
python 3.11+
pip install -r requirements.txt
```

### Download dataset
```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/ --unzip
```

### Run EDA notebook
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Run API locally (once model is trained)
```bash
uvicorn api.main:app --reload
# API docs at http://localhost:8000/docs
```

### Run with Docker
```bash
docker-compose -f docker/docker-compose.yml up
```

---

## API Usage (coming in Week 5)

```bash
curl -X POST "https://your-api-url/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 149.62,
    "hour": 2,
    "v1": -1.35, "v2": -0.07, "v3": 2.53,
    ...
  }'
```

Response:
```json
{
  "fraud_probability": 0.9231,
  "is_fraud": true,
  "risk_level": "HIGH",
  "top_features": [
    {"feature": "V14", "impact": -0.842},
    {"feature": "V17", "impact": -0.631},
    {"feature": "V12", "impact": -0.419}
  ],
  "latency_ms": 12.4
}
```

---

## Progress

- [x] Week 1 — Repository setup, dataset acquisition, EDA
- [ ] Week 2 — Feature engineering pipeline
- [ ] Week 3 — Model training & evaluation
- [ ] Week 4 — MLflow experiment tracking
- [ ] Week 5 — FastAPI serving layer
- [ ] Week 6 — Docker + CI/CD pipeline
- [ ] Week 7 — Streamlit dashboard
- [ ] Week 8 — Drift monitoring + final polish

---

## Results (updating as project progresses)

| Model | AUC-ROC | AUC-PR | F1 |
|---|---|---|---|
| Logistic Regression (baseline) | — | — | — |
| Random Forest | — | — | — |
| LightGBM | — | — | — |
| XGBoost (tuned) | — | — | — |

---

## Author

**Kousthubh N Gowda**  
Data Analyst Intern@ JAR  
[LinkedIn](https://www.linkedin.com/in/kousthubh-n-gowda-6698b02a9/) 

---

