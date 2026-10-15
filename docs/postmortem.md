# Post-mortem: INC-<equipo> · <título breve del incidente>

> **Formato sin culpables (*blameless*):** se analizan procesos, herramientas y decisiones, no personas.
> El objetivo es que el mismo problema no vuelva a sorprendernos.

| Campo | Valor |
|---|---|
| Fecha del incidente | AAAA-MM-DD |
| Autor(es) del post-mortem | |
| Líder del incidente | |
| Estado | Resuelto / Mitigado / En seguimiento |
| Nivel según el runbook | ✅ OK · ⚠️ ALERTA – drift · ⚠️ ALERTA – desempeño · 🛑 CRÍTICO – contrato · 🛑 CRÍTICO – servicio |
| Versión del modelo afectada | `model_version` de `/health` o `models/model_meta.json` |
| Enlace al *issue* | |

## 1. Resumen (3–5 líneas, sin jerga)

Qué pasó, a quién afectó, cuánto duró y cómo se resolvió.

## 2. Impacto

| Dimensión | Descripción | Estimación |
|---|---|---|
| Decisiones de negocio afectadas | (ej. lista de la campaña, aprobación de créditos) | |
| Clientes / registros afectados | | |
| Impacto económico estimado | (use la matriz de costos de `src/business_analysis.py`) | |
| Tiempo sin servicio o sin predicciones confiables | | |

## 3. Línea de tiempo

| Hora (T+min) | Evento | Quién | Evidencia |
|---|---|---|---|
| T+0 | Llega el lote de producción | Operaciones | *issue* |
| T+… | Validación del contrato | | salida de `validate` / `drift_report` |
| T+… | Primera señal de alerta (o confirmación de que todo está bien) | | reporte |
| T+… | Diagnóstico | | |
| T+… | Decisión | | `docs/decision_log.md` |
| T+… | Ejecución | | commit / tag / corrida MLflow |
| T+… | Comunicado al negocio | | *issue* |
| T+… | Cierre | | |

**Tiempo hasta detectar:** … min · **Tiempo hasta decidir:** … min · **Tiempo hasta resolver o mitigar:** … min

## 4. Detección

- **Señal que lo reveló:** (contrato, drift, desempeño con etiquetas, alerta de Prometheus, reclamo del negocio…)
- **Herramienta y umbral:** (ej. `drift_report`: 43 % de columnas con drift, umbral 25 %)
- **¿Lo detectó el monitoreo o una persona?**
- **¿Se pudo haber detectado antes? ¿Cómo?**

## 5. Causa raíz

Use la técnica de los **cinco porqués**:

1. ¿Por qué falló / cambió …? Porque …
2. ¿Por qué …? Porque …
3. ¿Por qué …? Porque …
4. ¿Por qué …? Porque …
5. ¿Por qué …? Porque … ← **causa raíz**

**Tipo de problema:** drift de datos · drift de concepto · cambio en la proporción del objetivo · cambio de esquema · calidad de datos · infraestructura · ninguno (falsa alarma)

## 6. Decisión y respuesta

| Pregunta | Respuesta |
|---|---|
| ¿Qué nivel del runbook se aplicó? | |
| ¿Qué alternativas se consideraron? | (no predecir, predecir con advertencia, reentrenar, revertir, plan B…) |
| ¿Por qué se eligió esta? | |
| ¿Quién la aprobó? | |
| ¿Qué evidencia la respalda? | commit, tag, corrida MLflow, reporte |
| ¿Se usó el modelo para decidir durante el incidente? ¿Con qué precauciones? | |

## 7. Qué funcionó y qué no

| Funcionó bien ✅ | No funcionó o faltó ❌ |
|---|---|
| | |

## 8. Acciones de mejora

| # | Acción | Tipo (runbook, contrato, monitoreo, pruebas, proceso, comunicación) | Responsable | Fecha | Estado |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |

## 9. Lecciones para el runbook

¿Qué cambia en `docs/runbook.md` a partir de este incidente? (Enlace al PR que lo actualiza.)
