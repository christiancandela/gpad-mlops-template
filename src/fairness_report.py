"""Análisis de equidad del modelo por grupos.

  python -m src.fairness_report                                   # grupo: senior_citizen
  python -m src.fairness_report --group internet_service
  python -m src.fairness_report --thresholds 0=0.30,1=0.40        # mitigación: umbral por grupo

Métricas por grupo: tasa de selección, TPR (recall), FPR, precisión (PPV) y ROC-AUC.
Brechas (máximo − mínimo entre grupos):
  - paridad demográfica   → diferencia en la tasa de selección
  - igualdad de oportunidades → diferencia en TPR
  - probabilidades igualadas  → máximo entre la diferencia de TPR y la de FPR
  - paridad predictiva   → diferencia en precisión
Salida: reports/fairness/<grupo>.json
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.common import FEATURES, MODEL_META, MODEL_PATH, REPORTS_DIR, TARGET, TEST_DATA


def group_metrics(y: pd.Series, pred: pd.Series, scores: pd.Series) -> dict:
    pos, neg = y == 1, y == 0
    return {
        "n": int(len(y)),
        "base_rate": round(float(y.mean()), 3),
        "selection_rate": round(float(pred.mean()), 3),
        "tpr": round(float(pred[pos].mean()), 3) if pos.any() else np.nan,
        "fpr": round(float(pred[neg].mean()), 3) if neg.any() else np.nan,
        "precision": round(float(y[pred == 1].mean()), 3) if (pred == 1).any() else np.nan,
        "roc_auc": round(roc_auc_score(y, scores), 3) if pos.any() and neg.any() else np.nan,
    }


def fairness_gaps(table: pd.DataFrame) -> dict:
    def gap(col):
        return round(float(table[col].max() - table[col].min()), 3)

    return {
        "demographic_parity_diff": gap("selection_rate"),
        "equal_opportunity_diff": gap("tpr"),
        "equalized_odds_diff": max(gap("tpr"), gap("fpr")),
        "predictive_parity_diff": gap("precision"),
    }


def parse_thresholds(text: str | None) -> dict:
    if not text:
        return {}
    return {k.strip(): float(v) for k, v in (item.split("=") for item in text.split(","))}


def main():
    parser = argparse.ArgumentParser(description="Análisis de equidad por grupos")
    parser.add_argument("--group", default="senior_citizen")
    parser.add_argument("--thresholds", help="umbral por grupo, ej. 0=0.30,1=0.40")
    args = parser.parse_args()

    test = pd.read_csv(TEST_DATA)
    meta = json.loads(MODEL_META.read_text())
    default_t = float(meta["decision_threshold"])
    per_group = parse_thresholds(args.thresholds)

    scores = pd.Series(joblib.load(MODEL_PATH).predict_proba(test[FEATURES])[:, 1], index=test.index)
    groups = test[args.group].astype(str)
    thresholds = groups.map(lambda g: per_group.get(g, default_t))
    pred = (scores >= thresholds).astype(int)

    table = pd.DataFrame(
        {
            g: group_metrics(test.loc[idx, TARGET], pred[idx], scores[idx])
            for g, idx in groups.groupby(groups).groups.items()
        }
    ).T
    table.insert(0, "threshold", [per_group.get(g, default_t) for g in table.index])
    gaps = fairness_gaps(table)
    tp = int(((pred == 1) & (test[TARGET] == 1)).sum())

    print(f"Grupo: {args.group} · modelo v{meta['model_version']} ({meta['model_type']})")
    print(table.to_string())
    print("\nBrechas entre grupos:")
    for k, v in gaps.items():
        print(f"  {k:<26}{v}")
    print(f"\nGlobal: contactados {int(pred.sum())} · recall {tp / test[TARGET].sum():.3f}")

    out = REPORTS_DIR / "fairness"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.group}.json").write_text(
        json.dumps({"group": args.group, "by_group": table.to_dict(orient="index"), "gaps": gaps}, indent=2)
    )


if __name__ == "__main__":
    main()
