"""Genera un lote "de producción" con un escenario de cambio, para practicar monitoreo.

  python -m monitoring.make_production_batch --scenario precios
  python -m monitoring.make_production_batch --scenario concepto --with-labels

Escenarios disponibles: ver src/synthetic.py (SCENARIOS) o ejecutar con --list.
Por defecto el lote NO trae la variable objetivo (en producción las etiquetas llegan tarde).
"""

import argparse

from src.common import ROOT, TARGET
from src.synthetic import SCENARIOS, generate_customers


def main():
    parser = argparse.ArgumentParser(description="Genera un lote de producción simulado")
    parser.add_argument("--scenario", default="normal", choices=list(SCENARIOS))
    parser.add_argument("--rows", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--with-labels", action="store_true", help="incluir la columna churn")
    parser.add_argument("--list", action="store_true", help="listar escenarios y salir")
    args = parser.parse_args()

    if args.list:
        for name, desc in SCENARIOS.items():
            print(f"  {name:<10} {desc}")
        return

    batch = generate_customers(n=args.rows, seed=args.seed, scenario=args.scenario)
    if not args.with_labels:
        batch = batch.drop(columns=[TARGET])
    out = ROOT / "data" / "production" / f"batch_{args.scenario}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(out, index=False)
    print(f"[batch] {len(batch)} filas, escenario '{args.scenario}' → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
