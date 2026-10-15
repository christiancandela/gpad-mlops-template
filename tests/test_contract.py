"""Pruebas del contrato de datos: los datos válidos pasan y los inválidos se detectan."""

import pandera.errors
import pytest

from contracts.churn_contract import features_schema, training_schema
from src.synthetic import generate_customers


def test_valid_data_passes(sample_df):
    training_schema.validate(sample_df, lazy=True)


@pytest.mark.parametrize(
    "column, bad_value",
    [
        ("tenure_months", -3),
        ("contract_type", "weekly"),
        ("senior_citizen", 2),
        ("churn", 5),
    ],
)
def test_invalid_values_are_rejected(sample_df, column, bad_value):
    df = sample_df.copy()
    df[column] = df[column].astype(object)
    df.loc[df.index[0], column] = bad_value
    with pytest.raises(pandera.errors.SchemaErrors):
        training_schema.validate(df, lazy=True)


def test_duplicate_ids_are_rejected(sample_df):
    df = sample_df.copy()
    df.loc[df.index[1], "customer_id"] = df.loc[df.index[0], "customer_id"]
    with pytest.raises(pandera.errors.SchemaErrors):
        training_schema.validate(df, lazy=True)


def test_schema_change_scenario_breaks_contract():
    """El escenario 'esquema' del simulacro debe ser detectado por el contrato."""
    batch = generate_customers(n=300, seed=3, scenario="esquema")
    with pytest.raises(pandera.errors.SchemaErrors):
        features_schema.validate(batch, lazy=True)
