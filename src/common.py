"""Configuración compartida: rutas, columnas y lectura de parámetros."""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = ROOT / "data" / "raw" / "churn.csv"
TRAIN_DATA = ROOT / "data" / "processed" / "train.csv"
TEST_DATA = ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
MODEL_META = ROOT / "models" / "model_meta.json"
METRICS_PATH = ROOT / "metrics.json"
REPORTS_DIR = ROOT / "reports"

ID_COL = "customer_id"
TARGET = "churn"
NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "support_tickets", "senior_citizen"]
CATEGORICAL_FEATURES = ["contract_type", "internet_service", "payment_method"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CATEGORIES = {
    "contract_type": ["month-to-month", "one-year", "two-year"],
    "internet_service": ["dsl", "fiber", "none"],
    "payment_method": ["electronic_check", "mailed_check", "bank_transfer", "credit_card"],
}


def load_params() -> dict:
    with open(ROOT / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_mlflow():
    """Configura MLflow. Por defecto usa una base SQLite local (`mlflow.db`).

    Para usar el servidor del docker-compose: export MLFLOW_TRACKING_URI=http://localhost:5001
    """
    import mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(load_params()["mlflow"]["experiment"])
    return mlflow
