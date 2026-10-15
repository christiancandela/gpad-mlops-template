"""Generador de datos sintéticos de deserción (churn) para telecomunicaciones.

Imita la estructura del dataset Telco Customer Churn (Ficha 1 del proyecto) para que la
plantilla funcione sin descargar datos. Los equipos reemplazan `src/ingest.py` por la
ingesta de su caso.

También genera lotes "de producción" con escenarios de cambio (drift) para el monitoreo
y el simulacro de incidente.
"""

import numpy as np
import pandas as pd

from src.common import CATEGORIES

SCENARIOS = {
    "normal": "Mismo proceso generador que el entrenamiento (no debería alertar).",
    "precios": "Alza de tarifas (+20 %) y migración a fibra: drift en monthly_charges e internet_service.",
    "contratos": "Campaña de captación: más clientes nuevos y contratos mes a mes.",
    "concepto": "Llega un competidor: cambia la relación X→churn (drift de concepto, requiere etiquetas).",
    "esquema": "Cambio en el sistema origen: nueva categoría y columna renombrada (rompe el contrato).",
}


def generate_customers(n: int, seed: int, scenario: str = "normal") -> pd.DataFrame:
    if scenario not in SCENARIOS:
        raise ValueError(f"Escenario desconocido: {scenario}. Opciones: {list(SCENARIOS)}")
    rng = np.random.default_rng(seed)

    contract_p = [0.55, 0.21, 0.24]
    internet_p = [0.34, 0.44, 0.22]
    tenure_scale = 30.0
    if scenario == "contratos":
        contract_p = [0.80, 0.12, 0.08]
        tenure_scale = 10.0
    if scenario == "precios":
        internet_p = [0.20, 0.70, 0.10]

    contract = rng.choice(CATEGORIES["contract_type"], size=n, p=contract_p)
    internet = rng.choice(CATEGORIES["internet_service"], size=n, p=internet_p)
    payment = rng.choice(CATEGORIES["payment_method"], size=n, p=[0.34, 0.23, 0.22, 0.21])
    senior = rng.binomial(1, 0.16, size=n)

    tenure = np.clip(rng.exponential(tenure_scale, size=n), 0, 72).astype(int)
    tenure = np.where(contract == "two-year", np.clip(tenure + 24, 0, 72), tenure)

    base_charge = np.select([internet == "fiber", internet == "dsl"], [85.0, 55.0], default=22.0)
    charges = base_charge + rng.normal(0, 10, size=n)
    if scenario == "precios":
        charges = charges * 1.20
    charges = np.round(np.clip(charges, 18.0, 150.0), 2)

    tickets = rng.poisson(1.2 + 1.0 * (internet == "fiber"), size=n).clip(0, 10)

    logit = (
        -1.85
        + 1.3 * (contract == "month-to-month")
        - 1.0 * (contract == "two-year")
        + 0.5 * (internet == "fiber")
        + 0.6 * (payment == "electronic_check")
        - 0.035 * tenure
        + 0.012 * (charges - 65)
        + 0.30 * tickets
        + 0.35 * senior
    )
    if scenario == "concepto":
        # Un competidor capta a los clientes antiguos con contratos largos: la antigüedad y el
        # tipo de contrato dejan de proteger contra la deserción (el patrón aprendido se invierte).
        logit = logit + 1.0 * (contract != "month-to-month") + 0.035 * tenure - 0.5 * (internet == "fiber")
    churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    df = pd.DataFrame(
        {
            "customer_id": [f"C{seed:04d}-{i:06d}" for i in range(n)],
            "tenure_months": tenure,
            "contract_type": contract,
            "internet_service": internet,
            "payment_method": payment,
            "monthly_charges": charges,
            "support_tickets": tickets,
            "senior_citizen": senior,
            "churn": churn,
        }
    )

    if scenario == "esquema":
        df.loc[df.sample(frac=0.15, random_state=seed).index, "internet_service"] = "fiber_plus"
        df = df.rename(columns={"monthly_charges": "monthly_charge_usd"})
    return df
