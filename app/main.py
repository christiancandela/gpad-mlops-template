"""API de inferencia del modelo de churn (FastAPI).

Endpoints:
  GET  /health    → estado y versión del modelo servido (trazabilidad)
  POST /predict   → probabilidad y decisión para uno o varios clientes
  GET  /metrics   → métricas para Prometheus (latencia, volumen, distribución de predicciones)

Variables de entorno:
  MODEL_PATH       ruta del modelo (por defecto models/model.joblib)
  MODEL_META_PATH  ruta de la metadata (por defecto models/model_meta.json)
  PREDICTION_LOG   si se define, cada predicción se agrega a ese CSV (insumo para monitoreo)
"""

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "models" / "model.joblib"))
MODEL_META_PATH = Path(os.environ.get("MODEL_META_PATH", ROOT / "models" / "model_meta.json"))
PREDICTION_LOG = os.environ.get("PREDICTION_LOG")

REQUESTS = Counter("api_requests_total", "Solicitudes recibidas", ["endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Latencia de las solicitudes", ["endpoint"])
PREDICTIONS = Counter("model_predictions_total", "Predicciones emitidas", ["label"])
SCORES = Histogram(
    "model_prediction_score",
    "Distribución de la probabilidad de churn",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


class Customer(BaseModel):
    """Esquema de entrada. Debe ser coherente con contracts/churn_contract.py."""

    tenure_months: int = Field(ge=0, le=72)
    contract_type: Literal["month-to-month", "one-year", "two-year"]
    internet_service: Literal["dsl", "fiber", "none"]
    payment_method: Literal["electronic_check", "mailed_check", "bank_transfer", "credit_card"]
    monthly_charges: float = Field(ge=0, le=500)
    support_tickets: int = Field(ge=0, le=50)
    senior_citizen: Literal[0, 1]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tenure_months": 3,
                    "contract_type": "month-to-month",
                    "internet_service": "fiber",
                    "payment_method": "electronic_check",
                    "monthly_charges": 95.5,
                    "support_tickets": 4,
                    "senior_citizen": 0,
                }
            ]
        }
    }


class PredictRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    churn_probability: float
    churn: int


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["model"] = joblib.load(MODEL_PATH)
    state["meta"] = json.loads(MODEL_META_PATH.read_text()) if MODEL_META_PATH.exists() else {}
    yield
    state.clear()


app = FastAPI(title="GPAD – API de churn", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    endpoint = request.url.path
    if endpoint != "/metrics":
        LATENCY.labels(endpoint).observe(time.perf_counter() - start)
        REQUESTS.labels(endpoint, str(response.status_code)).inc()
    return response


@app.get("/health")
def health():
    meta = state.get("meta", {})
    return {
        "status": "ok" if "model" in state else "sin modelo",
        "model_name": meta.get("model_name"),
        "model_version": meta.get("model_version"),
        "model_type": meta.get("model_type"),
        "run_id": meta.get("run_id"),
        "trained_at": meta.get("trained_at"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="modelo no cargado")
    threshold = float(state["meta"].get("decision_threshold", 0.5))

    df = pd.DataFrame([c.model_dump() for c in req.customers])
    scores = model.predict_proba(df)[:, 1]
    labels = (scores >= threshold).astype(int)

    for score, label in zip(scores, labels, strict=True):
        SCORES.observe(float(score))
        PREDICTIONS.labels(str(label)).inc()
    if PREDICTION_LOG:
        logged = df.assign(
            churn_probability=scores,
            prediction=labels,
            model_version=state["meta"].get("model_version"),
            ts=pd.Timestamp.now(tz="UTC").isoformat(),
        )
        log_path = Path(PREDICTION_LOG)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logged.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

    return PredictResponse(
        model_version=str(state["meta"].get("model_version", "desconocida")),
        predictions=[
            Prediction(churn_probability=round(float(s), 4), churn=int(lbl))
            for s, lbl in zip(scores, labels, strict=True)
        ],
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
