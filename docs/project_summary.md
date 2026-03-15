# Fraud Detection System — Project Summary

## Overview

A production-ready credit card fraud detection system built on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 0.17% fraud rate). The system covers the full ML lifecycle: feature engineering, model training, experiment tracking, REST API serving, interactive dashboard, and automated drift monitoring.

---

## Architecture

```
Raw Data (CSV)
     │
     ▼
Feature Pipeline (src/features.py)
  - Time → hour-of-day, is_night
  - Amount → log_amount, amount_bin
  - Interaction terms: V14×Amount, V17×Amount, V12×Amount, V10×Amount
  - StandardScaler
     │
     ▼
Model Training (src/train.py)
  - XGBoost classifier
  - SMOTE oversampling to handle class imbalance
  - MLflow experiment tracking (SQLite backend)
     │
     ▼
Model Registry (src/registry.py)
  - MLflow Model Registry
  - Promotes best model to "Production" stage
     │
     ├──▶ REST API (api/main.py — FastAPI)
     │      POST /predict  →  { fraud_probability, is_fraud, threshold }
     │      GET  /health
     │      GET  /metrics
     │
     ├──▶ Dashboard (dashboard/app.py — Streamlit)
     │      KPI cards, transaction feed, precision/recall curve, SHAP waterfall
     │
     └──▶ Drift Monitor (src/monitor.py — Evidently AI)
            Weekly HTML report comparing training vs current distribution
```

---

## Model Performance

| Metric | Value |
|---|---|
| AUC-ROC | 0.9803 |
| AUC-PR | 0.8814 |
| Precision | 77.27% |
| Recall | 86.73% |
| F1 Score | 0.8173 |
| Fraud caught | 85 / 98 (86.73%) |
| False positives | 25 |

---

## CI/CD

- **ci.yml** — runs on every push/PR to `main`: 66 unit + integration tests, Docker build & push to Docker Hub, Trivy security scan
- **drift.yml** — runs every Monday 00:00 UTC (+ manual trigger): generates Evidently drift report, uploads HTML artifact (30-day retention), exits non-zero if drift share exceeds 50%

---

## Key Tech Stack

| Layer | Technology |
|---|---|
| Model | XGBoost + SMOTE (imbalanced-learn) |
| Experiment tracking | MLflow |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + SHAP + Matplotlib |
| Drift monitoring | Evidently AI |
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Testing | Pytest + Hypothesis (property-based) |
