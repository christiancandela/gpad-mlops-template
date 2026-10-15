"""Promoción y rollback de modelos en el registro de MLflow (aliases).

  python -m scripts.promote             → ejecuta el quality gate y, si pasa, @challenger → @champion
                                          (el champion anterior queda como @previous)
  python -m scripts.promote --rollback  → @previous vuelve a ser @champion

Recuerde que el artefacto servido por la API (models/model.joblib) proviene del commit desplegado:
un rollback completo también restaura los archivos del modelo desde el tag anterior y redespliega
(procedimiento en docs/runbook.md, sección 3.3).
Documente cada promoción o rollback en docs/decision_log.md.
"""

import argparse
import sys

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from scripts.quality_gate import evaluate_gate
from src.common import load_params, setup_mlflow


def alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    try:
        return client.get_model_version_by_alias(name, alias).version
    except MlflowException:
        return None


def promote(client: MlflowClient, name: str) -> int:
    result = evaluate_gate()
    if not result["passed"]:
        print("[promote] ❌ el quality gate no pasó; no se promueve. Ver reports/gate.json")
        return 1
    challenger = alias_version(client, name, "challenger")
    current = alias_version(client, name, "champion")
    if challenger is None:
        print("[promote] no hay modelo @challenger; ejecute `dvc repro` primero")
        return 1
    if current == challenger:
        print(f"[promote] v{challenger} ya es @champion")
        return 0
    if current is not None:
        client.set_registered_model_alias(name, "previous", current)
    client.set_registered_model_alias(name, "champion", challenger)
    previous = f"v{current}" if current else "ninguno"
    print(f"[promote] ✅ {name}: v{challenger} es ahora @champion (anterior: {previous})")
    return 0


def rollback(client: MlflowClient, name: str) -> int:
    previous = alias_version(client, name, "previous")
    current = alias_version(client, name, "champion")
    if previous is None:
        print("[rollback] no existe @previous; no es posible revertir")
        return 1
    client.set_registered_model_alias(name, "champion", previous)
    if current is not None:
        client.set_registered_model_alias(name, "previous", current)
    print(f"[rollback] ↩️  {name}: @champion vuelve a v{previous} (se retiró v{current})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rollback", action="store_true", help="revertir al champion anterior")
    args = parser.parse_args()
    setup_mlflow()
    client = MlflowClient()
    name = load_params()["mlflow"]["registered_model"]
    return rollback(client, name) if args.rollback else promote(client, name)


if __name__ == "__main__":
    sys.exit(main())
