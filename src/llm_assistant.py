"""Asistente de explicación con LLM (Entrega 4 · taller LLMOps de la S7).

Convierte una predicción del modelo en una explicación en lenguaje natural para la persona
que toma la decisión (en el caso churn, el agente del call center).

Flujo:
  1. El modelo predice la probabilidad y se calculan los factores locales más influyentes
     (oclusión: cuánto cambia la probabilidad si una variable toma su valor de referencia).
  2. Se arma el prompt versionado (prompts/explicacion_<version>.md) SOLO con esos datos:
     nunca se envían identificadores ni variables excluidas (params.yaml → llm.exclude).
  3. El proveedor configurado genera el texto: mock | ollama | anthropic.
  4. Los guardrails verifican el texto; si falla alguno, se usa la explicación de respaldo.

  python -m src.llm_assistant --row 0
  python -m src.llm_assistant --row 0 --provider ollama
"""

import argparse
import json
import re
import time
import urllib.request
from dataclasses import asdict, dataclass, field

import joblib
import pandas as pd

from src.common import (
    CATEGORICAL_FEATURES,
    FEATURES,
    MODEL_META,
    MODEL_PATH,
    ROOT,
    TEST_DATA,
    TRAIN_DATA,
    load_params,
)

# Nombres legibles de las variables. 👉 Adáptelos a su caso.
FEATURE_LABELS = {
    "tenure_months": "antigüedad del cliente (meses)",
    "contract_type": "tipo de contrato",
    "internet_service": "servicio de internet",
    "payment_method": "método de pago",
    "monthly_charges": "cargo mensual (USD)",
    "support_tickets": "tickets de soporte recientes",
    "senior_citizen": "adulto mayor",
}

PROHIBIDAS = ["garantiza", "garantizado", "seguro que", "con certeza", "100 %", "100%", "sin duda"]


# ---------------------------------------------------------------------------------------------
# 1. Factores locales
# ---------------------------------------------------------------------------------------------
def reference_values(train: pd.DataFrame) -> dict:
    """Valor de referencia por variable: mediana (numéricas) o moda (categóricas)."""
    ref = {}
    for col in FEATURES:
        ref[col] = train[col].mode().iloc[0] if col in CATEGORICAL_FEATURES else train[col].median()
    return ref


def local_factors(model, row: pd.DataFrame, reference: dict, exclude: list[str], k: int) -> list[dict]:
    """Oclusión: efecto = p(cliente) − p(cliente con esa variable en su valor de referencia)."""
    base = float(model.predict_proba(row[FEATURES])[0, 1])
    efectos = []
    for col in FEATURES:
        if col in exclude:
            continue
        alterado = row[FEATURES].copy()
        alterado[col] = reference[col]
        delta = base - float(model.predict_proba(alterado)[0, 1])
        efectos.append(
            {
                "feature": col,
                "label": FEATURE_LABELS.get(col, col.replace("_", " ")),
                "value": row[col].iloc[0].item() if hasattr(row[col].iloc[0], "item") else row[col].iloc[0],
                "delta_pp": round(delta * 100, 1),
            }
        )
    efectos.sort(key=lambda e: abs(e["delta_pp"]), reverse=True)
    return efectos[:k]


# ---------------------------------------------------------------------------------------------
# 2. Prompt versionado
# ---------------------------------------------------------------------------------------------
def load_prompt(version: str) -> tuple[str, str]:
    text = (ROOT / "prompts" / f"explicacion_{version}.md").read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S).strip()
    system, user = text.split("---USER---")
    return system.strip(), user.strip()


def render(template: str, case: dict, max_words: int) -> str:
    factores = "\n".join(
        f"- {f['label']} = {f['value']} ({'aumenta' if f['delta_pp'] > 0 else 'reduce'} el riesgo "
        f"en {abs(f['delta_pp'])} puntos)"
        for f in case["factors"]
    )
    return template.format(
        probabilidad=case["probability_pct"],
        decision=case["decision"],
        umbral=case["threshold_pct"],
        factores=factores,
        max_palabras=max_words,
    )


# ---------------------------------------------------------------------------------------------
# 3. Proveedores
# ---------------------------------------------------------------------------------------------
@dataclass
class Generation:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    provider: str = ""
    model: str = ""
    error: str = ""


def _mock(system: str, user: str, case: dict) -> Generation:
    """Respuesta determinista que imita un LLM bien comportado (CI, pruebas, clase sin red)."""
    f = case["factors"]
    principales = " y ".join(x["label"] for x in f[:2])
    texto = (
        f"El modelo estima una probabilidad de cancelación de {case['probability_pct']} %, "
        f"por lo que la decisión es: {case['decision']}. Lo que más pesa en este caso es {principales}. "
        f"Sugerencia: conversar con el cliente sobre {f[0]['label']} y ofrecer una alternativa que se ajuste "
        "a sus necesidades."
    )
    return Generation(
        text=texto,
        input_tokens=len((system + user).split()),
        output_tokens=len(texto.split()),
        model="mock-determinista",
    )


def _ollama(system: str, user: str, model: str) -> Generation:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.2},
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.load(resp)
    return Generation(
        text=data["message"]["content"].strip(),
        input_tokens=data.get("prompt_eval_count", 0),
        output_tokens=data.get("eval_count", 0),
    )


def _anthropic(system: str, user: str, model: str) -> Generation:
    """Claude API con el SDK oficial. Credenciales: ANTHROPIC_API_KEY (nunca en el repositorio)."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "low"},
        # Si el modelo declina por políticas de seguridad, la API reintenta con el modelo
        # de respaldo recomendado por Anthropic dentro de la misma llamada.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        return Generation(text="", error="refusal")
    texto = "".join(b.text for b in response.content if b.type == "text").strip()
    return Generation(
        text=texto,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        model=response.model,
    )


def generate(system: str, user: str, case: dict, provider: str, params: dict) -> Generation:
    model = params["llm"]["model"].get(provider, "")
    start = time.perf_counter()
    try:
        if provider == "mock":
            gen = _mock(system, user, case)
        elif provider == "ollama":
            gen = _ollama(system, user, model)
        elif provider == "anthropic":
            gen = _anthropic(system, user, model)
        else:
            raise ValueError(f"Proveedor desconocido: {provider}")
    except Exception as exc:  # el asistente nunca debe tumbar el flujo de decisión
        gen = Generation(text="", error=f"{type(exc).__name__}: {exc}")
    gen.latency_s = round(time.perf_counter() - start, 3)
    gen.provider = provider
    gen.model = gen.model or model
    return gen


# ---------------------------------------------------------------------------------------------
# 4. Guardrails y respaldo
# ---------------------------------------------------------------------------------------------
def guardrails(text: str, case: dict, max_words: int, forbidden_values: list[str]) -> list[str]:
    """Controles automáticos sobre el texto generado. Devuelve la lista de violaciones."""
    violaciones = []
    if not text:
        return ["sin_texto"]
    porcentajes = [int(round(float(p.replace(",", ".")))) for p in re.findall(r"(\d+(?:[.,]\d+)?)\s*%", text)]
    if case["probability_pct"] not in porcentajes:
        violaciones.append("no_menciona_probabilidad")
    if any(abs(p - case["probability_pct"]) > 1 for p in porcentajes):
        violaciones.append("cifra_inventada")
    palabras_clave = {w for f in case["factors"] for w in re.findall(r"\w{5,}", f["label"].lower())}
    if not any(w in text.lower() for w in palabras_clave):
        violaciones.append("no_menciona_factores")
    if len(text.split()) > max_words * 1.2:
        violaciones.append("demasiado_largo")
    if any(p in text.lower() for p in PROHIBIDAS):
        violaciones.append("afirmacion_absoluta")
    if any(v and str(v) in text for v in forbidden_values):
        violaciones.append("dato_sensible")
    return violaciones


def fallback_text(case: dict) -> str:
    f = case["factors"]
    detalle = "; ".join(
        f"{x['label']}: {x['value']} ({'sube' if x['delta_pp'] > 0 else 'baja'} el riesgo)" for x in f
    )
    return (
        f"Probabilidad estimada de cancelación: {case['probability_pct']} % ({case['decision']}). "
        f"Factores principales: {detalle}."
    )


@dataclass
class Explanation:
    case_index: int
    probability_pct: int
    decision: str
    factors: list
    text: str
    used_fallback: bool
    violations: list = field(default_factory=list)
    provider: str = ""
    model: str = ""
    prompt_version: str = ""
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""


def build_case(model, test: pd.DataFrame, idx: int, reference: dict, params: dict, meta: dict) -> dict:
    row = test.iloc[[idx]]
    prob = float(model.predict_proba(row[FEATURES])[0, 1])
    threshold = float(meta.get("decision_threshold", 0.5))
    return {
        "index": idx,
        "probability_pct": int(round(prob * 100)),
        "threshold_pct": int(round(threshold * 100)),
        "decision": "riesgo alto: contactar al cliente" if prob >= threshold else "riesgo bajo: no contactar",
        "factors": local_factors(
            model, row, reference, params["llm"].get("exclude", []), params["llm"]["top_factors"]
        ),
        "forbidden_values": [str(row[c].iloc[0]) for c in ("customer_id",) if c in row],
    }


def explain(case: dict, params: dict, provider: str | None = None) -> Explanation:
    cfg = params["llm"]
    provider = provider or cfg["provider"]
    system_t, user_t = load_prompt(cfg["prompt_version"])
    system, user = render(system_t, case, cfg["max_words"]), render(user_t, case, cfg["max_words"])
    gen = generate(system, user, case, provider, params)
    violations = guardrails(gen.text, case, cfg["max_words"], case["forbidden_values"])
    used_fallback = bool(violations)
    return Explanation(
        case_index=case["index"],
        probability_pct=case["probability_pct"],
        decision=case["decision"],
        factors=case["factors"],
        text=fallback_text(case) if used_fallback else gen.text,
        used_fallback=used_fallback,
        violations=violations,
        provider=provider,
        model=gen.model,
        prompt_version=cfg["prompt_version"],
        latency_s=gen.latency_s,
        input_tokens=gen.input_tokens,
        output_tokens=gen.output_tokens,
        error=gen.error,
    )


def load_context():
    params = load_params()
    model = joblib.load(MODEL_PATH)
    meta = json.loads(MODEL_META.read_text())
    reference = reference_values(pd.read_csv(TRAIN_DATA))
    test = pd.read_csv(TEST_DATA)
    return params, model, meta, reference, test


def main():
    parser = argparse.ArgumentParser(description="Explica una predicción con el asistente LLM")
    parser.add_argument("--row", type=int, default=0, help="fila de data/processed/test.csv")
    parser.add_argument("--provider", choices=["mock", "ollama", "anthropic"], default=None)
    parser.add_argument("--model", default=None, help="sobrescribe el modelo de params.yaml")
    args = parser.parse_args()

    params, model, meta, reference, test = load_context()
    if args.model:
        params["llm"]["model"][args.provider or params["llm"]["provider"]] = args.model
    case = build_case(model, test, args.row, reference, params, meta)
    exp = explain(case, params, args.provider)
    print(json.dumps(asdict(exp), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
