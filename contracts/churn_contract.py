"""Contrato de datos ejecutable (Pandera).

Traduce a reglas verificables el contrato acordado con el dueño del dato (docs/datos_gobernanza.md).
Si los datos no cumplen, el pipeline se detiene antes de entrenar.

Hay dos esquemas:
- `features_schema`: lo que el modelo necesita para predecir (sirve también para lotes de producción).
- `training_schema`: features + variable objetivo + identificador único (datos de entrenamiento).
"""

import pandera.pandas as pa

from src.common import CATEGORIES

_features = {
    "tenure_months": pa.Column(int, pa.Check.in_range(0, 72)),
    "contract_type": pa.Column(str, pa.Check.isin(CATEGORIES["contract_type"])),
    "internet_service": pa.Column(str, pa.Check.isin(CATEGORIES["internet_service"])),
    "payment_method": pa.Column(str, pa.Check.isin(CATEGORIES["payment_method"])),
    "monthly_charges": pa.Column(float, pa.Check.in_range(0, 500)),
    "support_tickets": pa.Column(int, pa.Check.in_range(0, 50)),
    "senior_citizen": pa.Column(int, pa.Check.isin([0, 1])),
}

features_schema = pa.DataFrameSchema(
    columns=_features,
    strict=False,  # se permiten columnas adicionales (ej. customer_id, churn)
    coerce=True,
    name="churn_features",
)

training_schema = pa.DataFrameSchema(
    columns={
        "customer_id": pa.Column(str, unique=True),
        **_features,
        "churn": pa.Column(int, pa.Check.isin([0, 1])),
    },
    checks=[
        # Regla de negocio: la tasa de churn histórica debe ser plausible.
        pa.Check(lambda df: 0.05 <= df["churn"].mean() <= 0.60, error="tasa de churn fuera de rango"),
    ],
    strict=True,
    coerce=True,
    name="churn_training",
)
