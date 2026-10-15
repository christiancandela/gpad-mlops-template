"""Etapa `ingest`: obtiene los datos crudos y los deja en data/raw/.

👉 Reemplace esta etapa por la ingesta de su caso. Ejemplos:
   - Descargar el dataset desde una URL pública (UCI permite descarga directa).
   - Leer un archivo agregado con `dvc add data/raw/<archivo>.csv` (y quitar esta etapa de dvc.yaml).
Mantenga siempre la salida en una ruta fija para que DVC la versione.
"""

from src.common import RAW_DATA, load_params
from src.synthetic import generate_customers


def main():
    p = load_params()["data"]
    df = generate_customers(n=p["n_rows"], seed=p["seed"])
    RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA, index=False)
    print(f"[ingest] {len(df)} filas → {RAW_DATA.relative_to(RAW_DATA.parents[2])}")
    print(f"[ingest] tasa de churn: {df['churn'].mean():.1%}")


if __name__ == "__main__":
    main()
