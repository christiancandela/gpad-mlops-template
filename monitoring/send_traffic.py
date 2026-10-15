"""Envía un lote de producción a la API para simular tráfico (y ver el tablero de Grafana moverse).

  python -m monitoring.send_traffic data/production/batch_precios.csv --url http://localhost:8000

Las filas que no cumplen el esquema de la API se reportan como errores 422 (visibles en Grafana).
"""

import argparse
import json
import time
import urllib.error
import urllib.request

import pandas as pd

from src.common import FEATURES


def post(url: str, rows: list[dict]) -> int:
    req = urllib.request.Request(
        f"{url}/predict",
        data=json.dumps({"customers": rows}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code


def main():
    parser = argparse.ArgumentParser(description="Simula tráfico hacia la API")
    parser.add_argument("batch")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--pause", type=float, default=0.2, help="segundos entre solicitudes")
    args = parser.parse_args()

    df = pd.read_csv(args.batch)
    cols = [c for c in FEATURES if c in df.columns]
    if len(cols) < len(FEATURES):
        print(f"⚠️  faltan columnas {set(FEATURES) - set(cols)}: la API responderá 422")
    statuses: dict[int, int] = {}
    for start in range(0, len(df), args.batch_size):
        chunk = df.iloc[start : start + args.batch_size][cols].to_dict(orient="records")
        code = post(args.url, chunk)
        statuses[code] = statuses.get(code, 0) + 1
        time.sleep(args.pause)
    print(f"[traffic] solicitudes por código HTTP: {statuses}")


if __name__ == "__main__":
    main()
