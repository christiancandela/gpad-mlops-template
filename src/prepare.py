"""Etapa `prepare`: partición estratificada train/test.

El conjunto de entrenamiento también sirve como datos de *referencia* para el monitoreo de drift.
Si su caso es una serie de tiempo, reemplace la partición aleatoria por una partición temporal.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import RAW_DATA, TARGET, TEST_DATA, TRAIN_DATA, load_params


def split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df[TARGET])


def main():
    p = load_params()["split"]
    df = pd.read_csv(RAW_DATA)
    train, test = split(df, p["test_size"], p["seed"])
    TRAIN_DATA.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_DATA, index=False)
    test.to_csv(TEST_DATA, index=False)
    print(f"[prepare] train={len(train)} test={len(test)}")


if __name__ == "__main__":
    main()
