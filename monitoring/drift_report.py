"""Monitoreo de un lote de producción en tres niveles:

1. Contrato de datos (¿el lote cumple el esquema?)            → si falla: CRÍTICO (código 2)
2. Drift de datos con Evidently (referencia = datos de entrenamiento) → si supera el umbral: ALERTA (código 1)
3. Desempeño del modelo, solo si el lote trae etiquetas        → si el ROC-AUC cae > 0.05: ALERTA (código 1)

  python -m monitoring.drift_report data/production/batch_precios.csv

Salidas en reports/drift/: reporte HTML de Evidently y resumen JSON.
La acción a tomar ante cada nivel se define en docs/runbook.md.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
import pandera.errors
from sklearn.metrics import roc_auc_score

from contracts.churn_contract import features_schema
from src.common import (
    CATEGORICAL_FEATURES,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    REPORTS_DIR,
    TARGET,
    TRAIN_DATA,
    load_params,
)

warnings.filterwarnings("ignore", category=SyntaxWarning)
from evidently import DataDefinition, Dataset, Report  # noqa: E402
from evidently.presets import DataDriftPreset  # noqa: E402

PERFORMANCE_DROP_TOLERANCE = 0.05
# senior_citizen es 0/1: se trata como categórica para el drift.
DRIFT_NUMERIC = [c for c in NUMERIC_FEATURES if c != "senior_citizen"]
DRIFT_CATEGORICAL = CATEGORICAL_FEATURES + ["senior_citizen"]


def check_contract(batch: pd.DataFrame) -> list[dict]:
    try:
        features_schema.validate(batch, lazy=True)
        return []
    except pandera.errors.SchemaErrors as err:
        fc = err.failure_cases[["column", "check", "failure_case"]].astype(str).drop_duplicates()
        return fc.head(20).to_dict(orient="records")


def data_drift(reference: pd.DataFrame, current: pd.DataFrame, drift_share: float, html_path: Path) -> dict:
    definition = DataDefinition(numerical_columns=DRIFT_NUMERIC, categorical_columns=DRIFT_CATEGORICAL)
    report = Report([DataDriftPreset(columns=FEATURES, drift_share=drift_share)], include_tests=True)
    snapshot = report.run(
        current_data=Dataset.from_pandas(current[FEATURES], data_definition=definition),
        reference_data=Dataset.from_pandas(reference[FEATURES], data_definition=definition),
    )
    snapshot.save_html(str(html_path))
    result = snapshot.dict()

    summary = next(m["value"] for m in result["metrics"] if "DriftedColumnsCount" in m["metric_name"])
    drifted = [
        t["metric_config"]["params"]["column"]
        for t in result["tests"]
        if t["metric_config"]["params"]["type"].endswith("ValueDrift") and t["status"] == "FAIL"
    ]
    return {
        "drift_share": round(float(summary["share"]), 3),
        "drifted_columns": drifted,
        "threshold": drift_share,
        "alert": float(summary["share"]) >= drift_share,
    }


def performance(current: pd.DataFrame) -> dict | None:
    if TARGET not in current.columns or not MODEL_PATH.exists():
        return None
    model = joblib.load(MODEL_PATH)
    auc = roc_auc_score(current[TARGET], model.predict_proba(current[FEATURES])[:, 1])
    baseline = json.loads(METRICS_PATH.read_text())["roc_auc"]
    return {
        "roc_auc_current": round(float(auc), 4),
        "roc_auc_baseline": baseline,
        "alert": auc < baseline - PERFORMANCE_DROP_TOLERANCE,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitoreo de un lote de producción")
    parser.add_argument("batch", type=Path, help="CSV del lote de producción")
    parser.add_argument("--reference", type=Path, default=TRAIN_DATA)
    args = parser.parse_args()

    out_dir = REPORTS_DIR / "drift"
    out_dir.mkdir(parents=True, exist_ok=True)
    current = pd.read_csv(args.batch)
    summary: dict = {"batch": str(args.batch), "rows": len(current)}

    violations = check_contract(current)
    summary["contract_violations"] = violations
    if violations:
        summary["status"] = "CRÍTICO"
        print("[monitor] 🛑 CRÍTICO: el lote no cumple el contrato de datos. No se debe predecir.")
        for v in violations[:10]:
            print(f"    - {v['column']}: {v['check']} → {v['failure_case']}")
        code = 2
    else:
        reference = pd.read_csv(args.reference)
        threshold = load_params()["monitoring"]["drift_share_threshold"]
        html = out_dir / f"{args.batch.stem}.html"
        summary["data_drift"] = data_drift(reference, current, threshold, html)
        summary["performance"] = performance(current)

        drift = summary["data_drift"]
        perf = summary["performance"]
        print(
            f"[monitor] drift: {drift['drift_share']:.0%} de las columnas "
            f"(umbral {threshold:.0%}) → {drift['drifted_columns'] or 'ninguna'}"
        )
        if perf:
            print(
                f"[monitor] ROC-AUC en el lote: {perf['roc_auc_current']} "
                f"(referencia {perf['roc_auc_baseline']})"
            )
        alert = drift["alert"] or bool(perf and perf["alert"])
        summary["status"] = "ALERTA" if alert else "OK"
        verdict = "⚠️  ALERTA: aplicar el runbook" if alert else "✅ OK"
        print(f"[monitor] {verdict}  ·  reporte: {html.relative_to(REPORTS_DIR.parent)}")
        code = 1 if alert else 0

    (out_dir / f"{args.batch.stem}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
