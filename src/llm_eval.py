"""Evaluación del asistente LLM: el "quality gate" del prompt.

Genera explicaciones para una muestra fija de casos y mide lo que importa en operación:
tasa de guardrails superados (sin usar el respaldo), latencia p95, tokens y costo estimado.
Compara contra params.yaml → llm_gate y registra el resultado en MLflow.

  python -m src.llm_eval                     # proveedor de params.yaml (mock por defecto)
  python -m src.llm_eval --provider ollama
  python -m src.llm_eval --no-mlflow         # sin registro en MLflow

Salidas: reports/llm_eval.json (métricas + gate) y reports/llm_eval_samples.jsonl (casos).
Código de salida 1 si el gate falla.
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict

import numpy as np

from src.common import REPORTS_DIR, ROOT, setup_mlflow
from src.llm_assistant import build_case, explain, load_context


def summarize(explanations: list, params: dict, provider: str) -> dict:
    n = len(explanations)
    price = params["llm"]["price_per_mtok"].get(provider, {"input": 0.0, "output": 0.0})
    tokens_in = sum(e.input_tokens for e in explanations)
    tokens_out = sum(e.output_tokens for e in explanations)
    cost = (tokens_in * price["input"] + tokens_out * price["output"]) / 1_000_000
    latencias = [e.latency_s for e in explanations]
    return {
        "provider": provider,
        "model": explanations[0].model if explanations else "",
        "prompt_version": params["llm"]["prompt_version"],
        "cases": n,
        "guardrail_pass_rate": round(sum(not e.used_fallback for e in explanations) / n, 3),
        "errors": sum(bool(e.error) for e in explanations),
        "violations": dict(Counter(v for e in explanations for v in e.violations)),
        "latency_p50_s": round(float(np.percentile(latencias, 50)), 3),
        "latency_p95_s": round(float(np.percentile(latencias, 95)), 3),
        "tokens_input": tokens_in,
        "tokens_output": tokens_out,
        "cost_usd_total": round(cost, 6),
        "cost_usd_per_1000": round(cost / n * 1000, 4),
    }


def check_gate(summary: dict, gate: dict) -> dict:
    checks = {
        "guardrail_pass_rate": summary["guardrail_pass_rate"] >= gate["min_guardrail_pass_rate"],
        "latency_p95_s": summary["latency_p95_s"] <= gate["max_p95_latency_s"],
    }
    return {"passed": all(checks.values()), "checks": checks, "thresholds": gate}


def main():
    parser = argparse.ArgumentParser(description="Evalúa el asistente LLM sobre casos fijos")
    parser.add_argument("--provider", choices=["mock", "ollama", "anthropic"], default=None)
    parser.add_argument("--model", default=None, help="sobrescribe el modelo de params.yaml")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    params, model, meta, reference, test = load_context()
    provider = args.provider or params["llm"]["provider"]
    if args.model:
        params["llm"]["model"][provider] = args.model
    n = min(params["llm"]["eval_cases"], len(test))
    indices = test.sample(n=n, random_state=42).index.tolist()

    explanations = []
    for i, idx in enumerate(indices, 1):
        case = build_case(model, test, test.index.get_loc(idx), reference, params, meta)
        exp = explain(case, params, provider)
        estado = "✅" if not exp.used_fallback else f"⚠️  respaldo ({', '.join(exp.violations)})"
        print(
            f"[{i:>2}/{n}] fila {exp.case_index:>4} · p={exp.probability_pct:>3} % · "
            f"{exp.latency_s:>6.2f} s · {estado}"
        )
        explanations.append(exp)

    summary = summarize(explanations, params, provider)
    gate = check_gate(summary, params["llm_gate"])
    result = {**summary, "gate": gate}

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "llm_eval.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    with open(REPORTS_DIR / "llm_eval_samples.jsonl", "w", encoding="utf-8") as f:
        for e in explanations:
            f.write(json.dumps(asdict(e), ensure_ascii=False, default=str) + "\n")

    if not args.no_mlflow:
        mlflow = setup_mlflow()
        with mlflow.start_run(run_name=f"llm-{provider}-{summary['prompt_version']}"):
            mlflow.set_tag("stage", "llm_eval")
            mlflow.log_params(
                {
                    "provider": provider,
                    "llm_model": summary["model"],
                    "prompt_version": summary["prompt_version"],
                    "top_factors": params["llm"]["top_factors"],
                    "max_words": params["llm"]["max_words"],
                }
            )
            metricas = ("guardrail_pass_rate", "latency_p95_s", "cost_usd_per_1000", "errors")
            mlflow.log_metrics({k: summary[k] for k in metricas})
            mlflow.log_artifact(str(REPORTS_DIR / "llm_eval_samples.jsonl"))
            mlflow.log_artifact(str(ROOT / "prompts" / f"explicacion_{summary['prompt_version']}.md"))

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\n✅ El asistente supera el gate" if gate["passed"] else "\n❌ El asistente NO supera el gate")
    sys.exit(0 if gate["passed"] else 1)


if __name__ == "__main__":
    main()
