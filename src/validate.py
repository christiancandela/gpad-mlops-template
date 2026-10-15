"""Etapa `validate`: verifica el contrato de datos antes de continuar el pipeline.

Genera reports/validation.json. Si hay violaciones, termina con código 1 y DVC detiene el pipeline.
"""

import json
import sys

import pandas as pd
import pandera.errors

from contracts.churn_contract import training_schema
from src.common import RAW_DATA, REPORTS_DIR


def main() -> int:
    df = pd.read_csv(RAW_DATA)
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / "validation.json"
    try:
        training_schema.validate(df, lazy=True)
    except pandera.errors.SchemaErrors as err:
        failures = err.failure_cases[["column", "check", "failure_case"]].astype(str)
        summary = {
            "valid": False,
            "rows": len(df),
            "n_failures": len(failures),
            "failures_sample": failures.head(20).to_dict(orient="records"),
        }
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"[validate] ❌ el contrato de datos falló ({len(failures)} violaciones):")
        print(failures.head(10).to_string(index=False))
        return 1

    summary = {"valid": True, "rows": len(df), "churn_rate": round(float(df["churn"].mean()), 4)}
    out.write_text(json.dumps(summary, indent=2))
    print(f"[validate] ✅ contrato de datos OK ({len(df)} filas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
