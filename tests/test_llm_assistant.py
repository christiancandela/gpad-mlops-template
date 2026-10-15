"""Pruebas del asistente LLM (sin red: proveedor mock).

Verifican los controles que hacen operable el asistente: el prompt versionado, los
guardrails, el respaldo ante fallas y que nunca se envíen atributos sensibles al LLM.
"""

import pytest

from src.common import load_params
from src.llm_assistant import build_case, explain, guardrails, load_context, load_prompt, render

CASE = {
    "index": 0,
    "probability_pct": 62,
    "threshold_pct": 35,
    "decision": "riesgo alto: contactar al cliente",
    "factors": [
        {"feature": "contract_type", "label": "tipo de contrato", "value": "monthly", "delta_pp": 18.0},
        {"feature": "tenure_months", "label": "antigüedad del cliente (meses)", "value": 3, "delta_pp": 9.5},
    ],
    "forbidden_values": ["C0042-000123"],
}
OK = (
    "El riesgo de cancelación es 62 % por el tipo de contrato mes a mes y la poca antigüedad. "
    "Sugerencia: ofrecer un plan anual."
)


def test_prompt_renders_all_placeholders():
    params = load_params()
    system, user = load_prompt(params["llm"]["prompt_version"])
    texto = render(system, CASE, 90) + render(user, CASE, 90)
    assert "{" not in texto and "62 %" in texto and "tipo de contrato" in texto


def test_guardrails_accept_good_text():
    assert guardrails(OK, CASE, 90, CASE["forbidden_values"]) == []


@pytest.mark.parametrize(
    "texto, violacion",
    [
        (OK.replace("62 %", "80 %"), "cifra_inventada"),
        (OK + " Es seguro que cancelará.", "afirmacion_absoluta"),
        (OK + " Cliente C0042-000123.", "dato_sensible"),
        ("", "sin_texto"),
    ],
)
def test_guardrails_detect_violations(texto, violacion):
    assert violacion in guardrails(texto, CASE, 90, CASE["forbidden_values"])


@pytest.mark.model
def test_explanation_never_uses_sensitive_attribute():
    params, model, meta, reference, test = load_context()
    excluidas = params["llm"]["exclude"]
    for idx in range(20):
        case = build_case(model, test, idx, reference, params, meta)
        assert not {f["feature"] for f in case["factors"]} & set(excluidas)


@pytest.mark.model
def test_provider_failure_falls_back_to_deterministic_text():
    params, model, meta, reference, test = load_context()
    case = build_case(model, test, 0, reference, params, meta)
    exp = explain(case, params, provider="proveedor_caido")
    assert exp.used_fallback and exp.error
    assert f"{case['probability_pct']} %" in exp.text
