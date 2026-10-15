"""Pruebas de comportamiento del modelo entrenado (ML Test Score).

No miden exactitud: verifican que el modelo se comporte como el negocio espera.
Se ejecutan después de `dvc repro` (marcador `model`).
"""

import json

import joblib
import pytest

from src.common import FEATURES, METRICS_PATH, MODEL_PATH, load_params
from src.synthetic import generate_customers

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture
def customer():
    return generate_customers(n=1, seed=99)[FEATURES]


def test_probabilities_are_valid(model):
    X = generate_customers(n=200, seed=5)[FEATURES]
    proba = model.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_direction_contract(model, customer):
    """Direccionalidad: un contrato mes a mes debe tener mayor riesgo que uno a dos años."""
    monthly, two_year = customer.copy(), customer.copy()
    monthly["contract_type"] = "month-to-month"
    two_year["contract_type"] = "two-year"
    assert model.predict_proba(monthly)[0, 1] > model.predict_proba(two_year)[0, 1]


def test_direction_support_tickets(model, customer):
    """Direccionalidad: más tickets de soporte no debe reducir el riesgo."""
    few, many = customer.copy(), customer.copy()
    few["support_tickets"], many["support_tickets"] = 0, 8
    assert model.predict_proba(many)[0, 1] >= model.predict_proba(few)[0, 1]


def test_invariance_to_customer_id(model):
    """Invariancia: el identificador del cliente no debe influir en la predicción."""
    df = generate_customers(n=20, seed=8)
    renamed = df.copy()
    renamed["customer_id"] = "OTRO-ID"
    assert (model.predict_proba(df[FEATURES]) == model.predict_proba(renamed[FEATURES])).all()


def test_metrics_meet_minimum_thresholds():
    metrics = json.loads(METRICS_PATH.read_text())
    gate = load_params()["gate"]
    assert metrics["roc_auc"] >= gate["min_roc_auc"]
