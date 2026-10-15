"""Etapa `train`: entrena el modelo, lo registra en MLflow y le asigna el alias @challenger.

Salidas:
- models/model.joblib       → artefacto que sirve la API (versionado por DVC)
- models/model_meta.json   → trazabilidad: corrida MLflow, versión registrada, hash de datos
"""

import hashlib
import json
import warnings
from datetime import UTC, datetime

import joblib
import pandas as pd
from mlflow import MlflowClient

from src.common import (
    FEATURES,
    MODEL_META,
    MODEL_PATH,
    TARGET,
    TRAIN_DATA,
    load_params,
    setup_mlflow,
)
from src.features import SKOPS_TRUSTED_TYPES, build_pipeline

# MLflow advierte que las columnas enteras no admiten nulos en la firma del modelo;
# nuestro contrato de datos ya prohíbe nulos, así que silenciamos esa advertencia.
warnings.filterwarnings("ignore", message=".*integer column.*")


def file_md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    params = load_params()
    mp = params["model"]
    model_type = mp["type"]
    hyperparams = mp[model_type]
    registered_name = params["mlflow"]["registered_model"]

    train = pd.read_csv(TRAIN_DATA)
    X, y = train[FEATURES], train[TARGET]
    pipeline = build_pipeline(model_type, hyperparams, seed=params["split"]["seed"])

    mlflow = setup_mlflow()
    with mlflow.start_run(run_name=f"train-{model_type}") as run:
        pipeline.fit(X, y)
        data_md5 = file_md5(TRAIN_DATA)
        mlflow.log_params({"model_type": model_type, **hyperparams})
        mlflow.log_params(
            {
                "decision_threshold": mp["decision_threshold"],
                "train_rows": len(train),
                "train_data_md5": data_md5,
            }
        )
        mlflow.set_tags({"stage": "train", "course": "gpad"})
        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=registered_name,
            input_example=X.head(3),
            skops_trusted_types=SKOPS_TRUSTED_TYPES[model_type],
        )

    version = info.registered_model_version
    MlflowClient().set_registered_model_alias(registered_name, "challenger", version)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    meta = {
        "model_name": registered_name,
        "model_version": str(version),
        "model_type": model_type,
        "decision_threshold": mp["decision_threshold"],
        "run_id": run.info.run_id,
        "train_data_md5": data_md5,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    MODEL_META.write_text(json.dumps(meta, indent=2))
    print(f"[train] {model_type} registrado como {registered_name} v{version} (@challenger)")


if __name__ == "__main__":
    main()
