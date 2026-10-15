"""Pruebas unitarias de las piezas del pipeline (no requieren modelo entrenado)."""

import numpy as np

from src.common import FEATURES, TARGET
from src.evaluate import business_impact
from src.features import build_pipeline
from src.prepare import split
from src.synthetic import generate_customers


def test_generator_is_deterministic():
    a = generate_customers(n=200, seed=11)
    b = generate_customers(n=200, seed=11)
    assert a.equals(b)


def test_split_is_stratified_and_disjoint(sample_df):
    train, test = split(sample_df, test_size=0.2, seed=1)
    assert set(train["customer_id"]).isdisjoint(test["customer_id"])
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.03


def test_pipeline_handles_unknown_category(sample_df):
    model = build_pipeline("logistic_regression", {"max_iter": 500}).fit(
        sample_df[FEATURES], sample_df[TARGET]
    )
    row = sample_df[FEATURES].head(1).copy()
    row["internet_service"] = "satellite"
    proba = model.predict_proba(row)[0, 1]
    assert 0.0 <= proba <= 1.0


def test_business_impact_beats_random_with_perfect_model():
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    biz = {"customer_value": 100.0, "contact_cost": 5.0, "retention_success_rate": 0.5}
    result = business_impact(y_true, y_true.copy(), biz)
    assert result["customers_contacted"] == 2
    assert result["net_savings_model"] == 2 * 50 - 2 * 5
    assert result["net_savings_vs_random"] > 0
