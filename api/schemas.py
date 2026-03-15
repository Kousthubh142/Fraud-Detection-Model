"""
api/schemas.py

Pydantic request and response models for the fraud detection API.
These handle validation, serialization, and auto-generate OpenAPI docs.
"""

from pydantic import BaseModel, Field
from typing import Optional


class TransactionRequest(BaseModel):
    """
    A single transaction to score for fraud.
    V1-V28 are PCA-transformed features from the original dataset.
    """
    # PCA features (anonymised)
    V1:  float; V2:  float; V3:  float; V4:  float
    V5:  float; V6:  float; V7:  float; V8:  float
    V9:  float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float

    # Raw features
    Amount: float = Field(..., gt=0,  description="Transaction amount in USD")
    Time:   float = Field(..., ge=0,  description="Seconds elapsed since first transaction")

    model_config = {
        "json_schema_extra" : {
            "example": {
                "V1": -1.3598071336738, "V2": -0.0727811733098497,
                "V3": 2.53634673796914, "V4": 1.37815522427443,
                "V5": -0.338320769942518, "V6": 0.462387777762292,
                "V7": 0.239598554061257, "V8": 0.0986979012610507,
                "V9": 0.363786969611213, "V10": 0.0907941719789316,
                "V11": -0.551599533260813, "V12": -0.617800855762348,
                "V13": -0.991389847235408, "V14": -0.311169353699879,
                "V15": 1.46817697209427,  "V16": -0.470400525259478,
                "V17": 0.207971241929242, "V18": 0.0257905801985591,
                "V19": 0.403992960255733, "V20": 0.251412098239705,
                "V21": -0.018306777944153,"V22": 0.277837575558899,
                "V23": -0.110473910188767,"V24": 0.0669280749146731,
                "V25": 0.128539358273528, "V26": -0.189114843888824,
                "V27": 0.133558376740387, "V28": -0.0210530534538215,
                "Amount": 149.62,
                "Time": 0.0
            }
        }
    }


class FeatureContribution(BaseModel):
    """A single SHAP feature contribution."""
    feature: str
    impact:  float
    value:   float


class PredictionResponse(BaseModel):
    """Full fraud scoring response."""
    fraud_probability: float = Field(..., description="Probability of fraud (0–1)")
    is_fraud:          bool  = Field(..., description="True if above decision threshold")
    risk_level:        str   = Field(..., description="LOW / MEDIUM / HIGH")
    threshold_used:    float = Field(..., description="Decision threshold applied")
    top_features:      list[FeatureContribution] = Field(
        ..., description="Top 5 SHAP feature contributions"
    )
    latency_ms:        float = Field(..., description="Inference latency in milliseconds")
    model_version:     str   = Field(..., description="Model version used")


class HealthResponse(BaseModel):
    status:        str
    model_name:    str
    model_version: str
    model_stage:   str


class BatchTransactionRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(
        ..., min_length=1, max_length=100, description="Max 100 transactions per batch"
    )


class BatchPredictionResponse(BaseModel):
    predictions:    list[PredictionResponse]
    total:          int
    fraud_count:    int
    latency_ms:     float