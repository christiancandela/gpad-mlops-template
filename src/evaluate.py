"""Etapa `evaluate`: métricas técnicas, de equidad y de impacto estimado sobre el conjunto de prueba.

Salidas:
- metrics.json               → métricas versionadas en Git (`dvc metrics show` / `dvc metrics diff`)
- reports/roc_curve.csv      → curva ROC (`dvc plots show`)
- reports/confusion_matrix.png
Las métricas también se registran en la misma corrida de MLflow que entrenó el modelo.
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.common import (
    FEATURES,
    METRICS_PATH,
    MODEL_META,
    MODEL_PATH,
    REPORTS_DIR,
    TARGET,
    TEST_DATA,
    load_params,
    setup_mlflow,
)


def business_impact(y_true, y_pred, biz: dict) -> dict:
    """Ahorro neto estimado de una campaña de retención dirigida por el modelo,
    comparado con contactar la misma cantidad de clientes al azar (línea base)."""
    contacted = int(y_pred.sum())
    true_pos = int(((y_pred == 1) & (y_true == 1)).sum())
    base_rate = float(y_true.mean())
    gain = biz["customer_value"] * biz["retention_success_rate"]
    model_net = true_pos * gain - contacted * biz["contact_cost"]
    random_net = contacted * base_rate * gain - contacted * biz["contact_cost"]
    return {
        "customers_contacted": contacted,
        "net_savings_model": round(model_net, 2),
        "net_savings_random": round(random_net, 2),
        "net_savings_vs_random": round(model_net - random_net, 2),
    }


def group_recall_gap(df: pd.DataFrame, y_pred, group_col: str) -> float:
    recalls = []
    for _, idx in df.groupby(group_col).groups.items():
        recalls.append(recall_score(df.loc[idx, TARGET], pd.Series(y_pred, index=df.index)[idx]))
    return round(max(recalls) - min(recalls), 4)


def main():
    params = load_params()
    threshold = params["model"]["decision_threshold"]
    test = pd.read_csv(TEST_DATA)
    model = joblib.load(MODEL_PATH)
    y_true = test[TARGET]
    y_score = model.predict_proba(test[FEATURES])[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "roc_auc": round(roc_auc_score(y_true, y_score), 4),
        "pr_auc": round(average_precision_score(y_true, y_score), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "recall_gap_senior_citizen": group_recall_gap(test, y_pred, "senior_citizen"),
        **business_impact(y_true.to_numpy(), y_pred, params["business"]),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    REPORTS_DIR.mkdir(exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(REPORTS_DIR / "roc_curve.csv", index=False)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["no churn", "churn"])
    plt.title(f"Matriz de confusión (umbral = {threshold})")
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=120, bbox_inches="tight")

    meta = json.loads(MODEL_META.read_text())
    mlflow = setup_mlflow()
    with mlflow.start_run(run_id=meta["run_id"]):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(REPORTS_DIR / "confusion_matrix.png"))

    print("[evaluate] " + ", ".join(f"{k}={v}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
