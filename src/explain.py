"""Explicabilidad global del modelo.

  python -m src.explain

1. Importancia por permutación (cualquier modelo): cuánto cae el ROC-AUC al desordenar cada variable.
2. Coeficientes (solo regresión logística): dirección y magnitud del efecto de cada variable transformada.
Salida: reports/explain/permutation_importance.csv
"""

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from src.common import FEATURES, MODEL_PATH, REPORTS_DIR, TARGET, TEST_DATA


def main():
    test = pd.read_csv(TEST_DATA)
    model = joblib.load(MODEL_PATH)

    result = permutation_importance(
        model, test[FEATURES], test[TARGET], scoring="roc_auc", n_repeats=10, random_state=0
    )
    importance = (
        pd.DataFrame(
            {
                "feature": FEATURES,
                "auc_drop_mean": result.importances_mean,
                "auc_drop_std": result.importances_std,
            }
        )
        .sort_values("auc_drop_mean", ascending=False)
        .round(4)
    )
    out = REPORTS_DIR / "explain"
    out.mkdir(parents=True, exist_ok=True)
    importance.to_csv(out / "permutation_importance.csv", index=False)
    print("Importancia por permutación (caída de ROC-AUC al desordenar la variable):")
    print(importance.to_string(index=False))

    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        names = model.named_steps["preprocess"].get_feature_names_out()
        coefs = pd.Series(estimator.coef_[0], index=names).sort_values()
        print("\nCoeficientes de la regresión logística (positivo = más riesgo de churn):")
        print(coefs.round(3).to_string())


if __name__ == "__main__":
    main()
