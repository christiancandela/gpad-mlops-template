# Asistente de explicación con LLM – Ficha de operación

> Producto de la **Entrega 4** (criterio *Asistente LLM*). Se construye en el taller LLMOps de la **S7**.
> Complete cada sección con **evidencia** del repositorio (`reports/llm_eval.json`, MLflow, `prompts/`).

## 1. Propósito y usuario

| Pregunta | Respuesta |
|---|---|
| ¿Quién lee la explicación? | *(ej.: agente del call center antes de llamar al cliente)* |
| ¿Qué decisión apoya? | *(ej.: si contactar al cliente y qué ofrecerle)* |
| ¿Qué **no** debe hacer el asistente? | *(ej.: decidir por el agente, prometer descuentos, hablar de atributos sensibles)* |
| ¿Lo ve el cliente final? | *(sí / no; si sí, ¿con qué revisión humana?)* |

## 2. Diseño

```
predicción (modelo @champion) ──► factores locales (oclusión) ──► prompt versionado ──► LLM ──► guardrails ──► texto
                                                                                                    └─ si falla: respaldo determinista
```

| Elemento | Decisión del equipo | Dónde |
|---|---|---|
| Proveedor y modelo | *(mock / ollama `<modelo>` / anthropic `<modelo>`) y por qué* | `params.yaml → llm` |
| Versión del prompt | *(v1, v2…) y qué cambió* | `prompts/` |
| Datos que recibe el LLM | Probabilidad, decisión, umbral y top-k factores. **Nunca** identificadores. | `src/llm_assistant.py` |
| Variables excluidas | *(ej.: `senior_citizen`, atributo sensible)* | `params.yaml → llm.exclude` |
| Guardrails | *(los de la plantilla + los que agregaron)* | `guardrails()` |
| Respaldo | Texto determinista con la probabilidad y los factores | `fallback_text()` |

## 3. Evaluación (quality gate del prompt)

Ejecute `uv run python -m src.llm_eval --provider <proveedor>` y copie la evidencia:

| Métrica | Umbral (`llm_gate`) | v1 | v2 |
|---|---|---|---|
| Tasa de guardrails superados | ≥ 0.90 | | |
| Latencia p95 (s) | ≤ 20 | | |
| Costo por 1 000 explicaciones (USD) | *(presupuesto)* | | |
| Violaciones más frecuentes | — | | |
| Run de MLflow | — | | |

**Revisión humana** (5 casos de `reports/llm_eval_samples.jsonl`, leídos por alguien que no escribió el prompt):

| Caso | ¿Fiel a los datos? | ¿Útil para el agente? | ¿Tono adecuado? | Observación |
|---|---|---|---|---|
| | sí / no | 1–5 | sí / no | |

## 4. Riesgos y controles

| Riesgo | Ejemplo | Control | Evidencia |
|---|---|---|---|
| Alucinación (cifras o causas inventadas) | "La probabilidad es 80 %" cuando es 62 % | Guardrail `cifra_inventada` + respaldo | prueba / caso |
| Fuga de datos personales | El texto incluye el `customer_id` | No se envía al LLM + guardrail `dato_sensible` | |
| Datos enviados a un tercero | Proveedor en la nube | Minimización; proveedor local si la finalidad no lo autoriza (Ley 1581) | |
| Sesgo o trato diferencial | Menciona la edad del cliente | `llm.exclude` + prueba `test_explanation_never_uses_sensitive_attribute` | |
| Inyección de instrucciones | Texto libre de un campo que dice "ignora las reglas…" | El prompt solo recibe campos estructurados del contrato | |
| Caída o lentitud del proveedor | Timeout | Respaldo determinista; latencia p95 en el gate | |
| Costo descontrolado | Volumen × precio por token | Costo por 1 000 explicaciones vs. presupuesto | |
| Deriva del LLM | El proveedor actualiza el modelo | Versión del modelo fijada en `params.yaml`; re-evaluar antes de cambiarla | |

## 5. Operación

| Pregunta | Respuesta |
|---|---|
| ¿Qué se monitorea en producción? | *(tasa de respaldo, latencia, costo, quejas de agentes)* |
| ¿Qué dispara una alerta? | *(ej.: tasa de respaldo > 20 % en un día)* |
| ¿Cómo se cambia el prompt? | PR con `prompts/explicacion_vN.md`, `llm_eval` en verde y registro en `docs/decision_log.md` |
| ¿Cómo se revierte? | `llm.prompt_version` o `llm.provider: mock` en `params.yaml` (el respaldo nunca falla) |
| ¿Quién aprueba un cambio de proveedor? | *(según la RACI)* |
