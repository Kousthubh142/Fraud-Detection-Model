# Fraud Detection System

> End-to-end real-time fraud detection pipeline — from raw transaction data to a deployed REST API with explainability and drift monitoring.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange) ![FastAPI](https://img.shields.io/badge/API-FastAPI-green) ![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue) ![Docker](https://img.shields.io/badge/Deploy-Docker-blue) ![CI](https://github.com/kousthubhng/fraud-detection-system/actions/workflows/ci.yml/badge.svg) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

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

---

## Model Performance

| Metric | Value |
|---|---|
| AUC-ROC | **0.9803** |
| AUC-PR | **0.8814** |
| Precision | 77.27% |
| Recall | 86.73% |
| F1 Score | 0.8173 |
| Fraud caught | 85 / 98 (86.73%) |
| False positives | 25 |

---

## Architecture

```
Raw Data (CSV)
     │
     ▼
Feature Engineering Pipeline
  - Time → hour-of-day, is_night
  - Amount → log_amount, amount_bin
  - Interaction terms: V14×Amount, V17×Amount, V12×Amount, V10×Amount
  - StandardScaler  (36 features total)
     │
     ▼
XGBoost Training + SMOTE oversampling
     │
     ▼
MLflow Experiment Tracking & Model Registry
     │
     ├──▶ FastAPI REST API
     │      POST /predict  →  { fraud_probability, is_fraud, risk_level }
     │      GET  /health  |  GET /metrics
     │
     ├──▶ Streamlit Dashboard
     │      KPI cards · transaction feed · precision/recall curve · SHAP waterfall
     │
     └──▶ Evidently AI Drift Monitor
            Weekly HTML report (GitHub Actions cron)
```

---

## Project Structure

```
fraud-detection-system/
├── api/
│   ├── main.py             # FastAPI app
│   ├── predictor.py        # Model loading + inference
│   └── schemas.py          # Pydantic request/response models
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── data/
│   ├── raw/                # creditcard.csv (gitignored)
│   └── processed/          # scaled arrays, feature pipeline
├── docs/
│   └── project_summary.md  # 1-page project summary
├── models/
│   ├── xgb_fraud_v1.pkl    # trained model
│   └── metrics.json        # evaluation metrics
├── src/
│   ├── features.py         # FeaturePipeline class
│   ├── train.py            # Training script with MLflow logging
│   ├── registry.py         # MLflow model registry helpers
│   └── monitor.py          # Evidently AI drift report
├── tests/                  # 66 unit + integration tests
├── .github/workflows/
│   ├── ci.yml              # test → docker build → security scan
│   └── drift.yml           # weekly drift monitor (Monday 00:00 UTC)
├── docker/
│   └── docker-compose.yml
├── Dockerfile
├── config.yaml
└── requirements.txt
```

---

## Getting Started

### Prerequisites
```bash
python 3.12+
pip install -r requirements.txt
```

### Download dataset
```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/ --unzip
```

### Train model
```bash
python src/features.py
python src/train.py
```

### Run API locally
```bash
uvicorn api.main:app --reload
# Docs at http://localhost:8000/docs
```

### Run dashboard
```bash
streamlit run dashboard/app.py
```

### Run with Docker
```bash
docker-compose up
```

### Run drift monitor
```bash
python src/monitor.py --threshold 0.5
# Report saved to reports/drift_report.html
```

---

## API Usage

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406.0,
    "V1": -2.31, "V2": 1.95, "V3": -1.60,
    "V4": 3.99, "V5": -0.52, "V6": -1.43,
    "V7": -2.77, "V8": 0.10, "V9": -0.33,
    "V10": -1.47, "V11": 1.18, "V12": -2.83,
    "V13": -0.17, "V14": -2.43, "V15": 1.20,
    "V16": -2.26, "V17": 0.52, "V18": -1.35,
    "V19": -0.42, "V20": 0.31, "V21": 0.03,
    "V22": 0.41, "V23": -0.17, "V24": 0.13,
    "V25": -0.08, "V26": 0.41, "V27": 0.06,
    "V28": 0.02, "Amount": 149.62
  }'
```

Response:
```json
{
  "fraud_probability": 0.9231,
  "is_fraud": true,
  "risk_level": "HIGH"
}
```

---

## CI/CD

- **ci.yml** — triggers on every push/PR to `main`: runs 66 tests, builds and pushes Docker image to Docker Hub, runs Trivy security scan
- **drift.yml** — runs every Monday 00:00 UTC (+ manual trigger via `workflow_dispatch`): generates Evidently drift report, uploads HTML artifact with 30-day retention, exits non-zero if drift share > 50%

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | Pandas, NumPy, scikit-learn |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Modelling | XGBoost |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| API serving | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit |
| Drift monitoring | Evidently AI |
| Containerisation | Docker, Docker Compose |
| CI/CD | GitHub Actions |

---

## Author

**Kousthubh N Gowda**  
Data Analyst Intern @ JAR  
[LinkedIn](https://www.linkedin.com/in/kousthubh-n-gowda-6698b02a9/)
