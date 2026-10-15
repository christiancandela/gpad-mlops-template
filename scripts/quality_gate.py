"""Quality gate: decide si el modelo @challenger puede promoverse a @champion.

Reglas (params.yaml → gate):
1. Umbrales absolutos: ROC-AUC, recall y precisión mínimos.
2. No regresión: el ROC-AUC del challenger no puede caer más de `max_drop_vs_champion`
   frente al @champion actual (si existe en el registro de MLflow).

Uso: python -m scripts.quality_gate      → código de salida 0 (pasa) o 1 (falla)
"""

import json
import sys

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.common import METRICS_PATH, MODEL_META, REPORTS_DIR, load_params, setup_mlflow


def champion_roc_auc(model_name: str) -> tuple[str, float | None] | None:
    """Versión y ROC-AUC del @champion actual; None si todavía no hay champion."""
    client = MlflowClient()
    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
    except MlflowException:
        return None
    run = client.get_run(champion.run_id)
    return champion.version, run.data.metrics.get("roc_auc")


def evaluate_gate() -> dict:
    params = load_params()
    gate = params["gate"]
    metrics = json.loads(METRICS_PATH.read_text())
    meta = json.loads(MODEL_META.read_text())
    checks = []

    for metric, key in [("roc_auc", "min_roc_auc"), ("recall", "min_recall"), ("precision", "min_precision")]:
        checks.append(
            {
                "rule": f"{metric} >= {gate[key]}",
                "value": metrics[metric],
                "passed": metrics[metric] >= gate[key],
            }
        )

    setup_mlflow()
    champ = champion_roc_auc(params["mlflow"]["registered_model"])
    if champ is None:
        checks.append({"rule": "no regresión vs @champion", "value": "sin champion previo", "passed": True})
    elif champ[1] is None:
        # Un champion sin métricas no se puede comparar: es un problema de trazabilidad, no se ignora.
        checks.append(
            {
                "rule": f"no regresión vs @champion v{champ[0]}",
                "value": "champion sin roc_auc registrado",
                "passed": False,
            }
        )
    else:
        version, champ_auc = champ
        limit = round(champ_auc - gate["max_drop_vs_champion"], 4)
        checks.append(
            {
                "rule": f"roc_auc >= champion v{version} ({champ_auc}) - {gate['max_drop_vs_champion']}",
                "value": metrics["roc_auc"],
                "passed": metrics["roc_auc"] >= limit,
            }
        )

    result = {
        "challenger_version": meta["model_version"],
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "gate.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> int:
    result = evaluate_gate()
    for c in result["checks"]:
        print(f"  {'✅' if c['passed'] else '❌'} {c['rule']}  (valor: {c['value']})")
    if result["passed"]:
        print(f"[gate] ✅ challenger v{result['challenger_version']} aprobado")
        return 0
    print(f"[gate] ❌ challenger v{result['challenger_version']} rechazado")
    return 1


if __name__ == "__main__":
    sys.exit(main())
