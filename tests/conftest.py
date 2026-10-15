import pytest

from src.common import MODEL_PATH
from src.synthetic import generate_customers


@pytest.fixture
def sample_df():
    return generate_customers(n=500, seed=7)


def pytest_collection_modifyitems(config, items):
    """Las pruebas marcadas con `model` se omiten si aún no existe un modelo entrenado."""
    if MODEL_PATH.exists():
        return
    skip = pytest.mark.skip(reason="no hay models/model.joblib: ejecute `dvc repro` primero")
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip)
