"""Impacto de negocio estimado: ¿qué umbral genera más valor y qué tan sensible es a los supuestos?

  python -m src.business_analysis

Usa la matriz de costos de params.yaml (business) sobre el conjunto de prueba:
  ahorro neto = verdaderos positivos × valor del cliente × tasa de éxito de la retención
                − clientes contactados × costo del contacto
Salida: reports/business/threshold_curve.csv
"""

import json

import joblib
import numpy as np
import pandas as pd

from src.common import FEATURES, MODEL_META, MODEL_PATH, REPORTS_DIR, TARGET, TEST_DATA, load_params

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def threshold_curve(y_true: np.ndarray, scores: np.ndarray, biz: dict) -> pd.DataFrame:
    gain = biz["customer_value"] * biz["retention_success_rate"]
    rows = []
    for t in THRESHOLDS:
        pred = scores >= t
        contacted = int(pred.sum())
        tp = int((pred & (y_true == 1)).sum())
        rows.append(
            {
                "threshold": t,
                "contacted": contacted,
                "true_positives": tp,
                "recall": round(tp / max(int(y_true.sum()), 1), 3),
                "precision": round(tp / max(contacted, 1), 3),
                "net_savings": round(tp * gain - contacted * biz["contact_cost"], 2),
            }
        )
    return pd.DataFrame(rows)


def main():
    params = load_params()
    biz = params["business"]
    current = json.loads(MODEL_META.read_text())["decision_threshold"]
    test = pd.read_csv(TEST_DATA)
    y = test[TARGET].to_numpy()
    scores = joblib.load(MODEL_PATH).predict_proba(test[FEATURES])[:, 1]

    curve = threshold_curve(y, scores, biz)
    out = REPORTS_DIR / "business"
    out.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out / "threshold_curve.csv", index=False)

    best = curve.loc[curve["net_savings"].idxmax()]
    print(
        f"Supuestos: valor cliente {biz['customer_value']} · costo contacto {biz['contact_cost']} · "
        f"éxito retención {biz['retention_success_rate']}"
    )
    print(curve.to_string(index=False))
    print(
        f"\nUmbral actual {current}: ahorro "
        f"{curve.loc[np.isclose(curve['threshold'], current), 'net_savings'].iloc[0]:,.0f}"
    )
    print(
        f"Umbral que maximiza el ahorro: {best['threshold']} → {best['net_savings']:,.0f} "
        f"({int(best['contacted'])} clientes contactados)"
    )

    print("\nSensibilidad a la tasa de éxito de la retención (supuesto más incierto):")
    for rate in (0.15, 0.30, 0.45):
        alt = threshold_curve(y, scores, {**biz, "retention_success_rate": rate})
        b = alt.loc[alt["net_savings"].idxmax()]
        print(
            f"  éxito {rate:.0%}: mejor umbral {b['threshold']} · contactar {int(b['contacted'])} · "
            f"ahorro {b['net_savings']:,.0f}"
        )


if __name__ == "__main__":
    main()
