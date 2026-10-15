"""Pruebas de la API (requieren modelo entrenado)."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

pytestmark = pytest.mark.model

VALID = {
    "tenure_months": 3,
    "contract_type": "month-to-month",
    "internet_service": "fiber",
    "payment_method": "electronic_check",
    "monthly_charges": 95.5,
    "support_tickets": 4,
    "senior_citizen": 0,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_model_version(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] is not None


def test_predict_valid_batch(client):
    resp = client.post("/predict", json={"customers": [VALID, {**VALID, "contract_type": "two-year"}]})
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 2
    assert all(0 <= p["churn_probability"] <= 1 for p in preds)


@pytest.mark.parametrize("field, value", [("contract_type", "weekly"), ("tenure_months", -1)])
def test_predict_rejects_invalid_input(client, field, value):
    resp = client.post("/predict", json={"customers": [{**VALID, field: value}]})
    assert resp.status_code == 422


def test_metrics_endpoint_exposes_prometheus_metrics(client):
    client.post("/predict", json={"customers": [VALID]})
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "model_prediction_score_bucket" in text
