# Runbook de operación del modelo

> Este runbook es el que el equipo sigue durante el **simulacro de incidente** (sesión 8). Ajuste los
> umbrales a su caso y justifíquelos.

## 1. Objetivos de nivel de servicio (SLO)
| Señal | SLO | Dónde se observa |
|---|---|---|
| Disponibilidad de la API | ≥ 99 % en horario laboral | Grafana – Solicitudes / `/health` |
| Latencia p95 de `/predict` | < 300 ms | Grafana – Latencia p95 |
| Tasa de errores | < 2 % | Grafana – Tasa de errores |
| Desempeño (cuando llegan etiquetas) | ROC-AUC ≥ referencia − 0.05 | `monitoring/drift_report.py` |
| Drift de datos | < 25 % de columnas con drift | `monitoring/drift_report.py` |

## 2. Niveles de alerta y respuesta
| Nivel | Condición | Acción | Responsable |
|---|---|---|---|
| ✅ OK | Sin violaciones | Registrar el reporte | MLOps Engineer |
| ⚠️ ALERTA – drift de datos | `drift_share ≥ umbral` | Analizar columnas afectadas con negocio; si hay etiquetas recientes, reentrenar como @challenger | Data Scientist |
| ⚠️ ALERTA – desempeño | ROC-AUC cae > 0.05 | Reentrenar con datos recientes; si no mejora, escalar al Líder | Data Scientist |
| 🛑 CRÍTICO – contrato | El lote viola el esquema | **No predecir**. Notificar al dueño del dato; corregir origen o actualizar contrato con aprobación | Líder + Data Steward |
| 🛑 CRÍTICO – servicio | API caída o errores > 10 % | Rollback a la versión anterior | MLOps Engineer |

## 3. Procedimientos

### 3.1 Detectar
```bash
uv run python -m monitoring.drift_report data/production/<lote>.csv
```

### 3.2 Reentrenar y promover
```bash
# incorporar los datos nuevos a la ingesta (o actualizar params.yaml) y luego:
uv run dvc repro                       # genera un nuevo @challenger
uv run python -m scripts.promote       # quality gate → @champion si pasa
git add -A && git commit -m "retrain: <motivo>" && git tag <version>
```

### 3.3 Rollback
```bash
uv run python -m scripts.promote --rollback     # registro: @previous → @champion
# Restaurar SOLO los archivos que definen el modelo (no la configuración de monitoreo ni el código nuevo):
git checkout <tag-anterior> -- params.yaml dvc.lock metrics.json models/model_meta.json reports/
uv run dvc checkout                             # datos y modelo de esa versión
docker compose up -d --build api                # redesplegar
curl -s localhost:8000/health                   # verificar la versión servida
git commit -am "rollback a <tag-anterior>: <motivo>"
```

## 4. Política de reentrenamiento
- **Por calendario:** cada ___ (justificar según la velocidad de cambio del negocio).
- **Por evento:** ante ALERTA de drift o de desempeño confirmada.
- **Nunca** promover sin pasar el quality gate.

## 5. Escalamiento y comunicación
Quién decide, a quién se informa y en cuánto tiempo. Registrar toda decisión en `docs/decision_log.md`.
