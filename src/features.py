"""Transformaciones de features y construcción del modelo.

El preprocesamiento vive dentro del Pipeline de scikit-learn: así entrenamiento e inferencia
usan exactamente la misma transformación (evita el *training-serving skew*).
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common import CATEGORICAL_FEATURES, CATEGORIES, NUMERIC_FEATURES

ESTIMATORS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}

# MLflow serializa los modelos sklearn con skops, que solo carga tipos explícitamente confiables.
# Los modelos de árboles necesitan declarar su estructura interna (los entrenamos nosotros: es seguro).
SKOPS_TRUSTED_TYPES = {
    "logistic_regression": [],
    "random_forest": ["sklearn.tree._tree.Tree"],
    "gradient_boosting": ["sklearn.tree._tree.Tree"],
}


def build_pipeline(model_type: str, hyperparams: dict, seed: int = 42) -> Pipeline:
    if model_type not in ESTIMATORS:
        raise ValueError(f"model.type debe ser uno de {list(ESTIMATORS)}")
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    categories=[CATEGORIES[c] for c in CATEGORICAL_FEATURES],
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    estimator = ESTIMATORS[model_type](random_state=seed, **hyperparams)
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])
