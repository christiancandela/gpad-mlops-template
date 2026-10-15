# Model Card – gpad-churn-model

> Basada en Mitchell et al. (2019), *Model Cards for Model Reporting*. Actualícela en cada promoción a @champion.

| Sección | Contenido |
|---|---|
| **Versión / fecha / responsables** | vX (ver `models/model_meta.json`) · AAAA-MM-DD · equipo |
| **Tipo de modelo** | Regresión logística con preprocesamiento (escalado + one-hot) en un Pipeline de scikit-learn |
| **Uso previsto** | Priorizar clientes para una campaña de retención con presupuesto limitado |
| **Usos no previstos** | Decisiones automáticas sobre precio, crédito o terminación de servicio |
| **Datos de entrenamiento** | `data/processed/train.csv` (hash en `model_meta.json`) |
| **Datos de evaluación** | `data/processed/test.csv` (20 %, estratificado) |
| **Métricas** | Ver `metrics.json`: ROC-AUC, PR-AUC, precisión, recall, F1 con umbral `decision_threshold` |
| **Equidad** | `recall_gap_senior_citizen` (brecha de recall entre grupos); discutir causas y mitigación |
| **Explicabilidad** | Coeficientes del modelo / SHAP (extensión) |
| **Limitaciones** | Datos sintéticos en la plantilla; desempeño sensible a cambios de tarifas (ver escenario `precios`) |
| **Asistente LLM** | Explicaciones en lenguaje natural para el agente (`docs/asistente_llm.md`): proveedor, versión del prompt y resultado de `reports/llm_eval.json` |
| **Monitoreo** | `monitoring/drift_report.py`; umbrales en `params.yaml` y `docs/runbook.md` |
| **Consideraciones éticas** | Riesgo de trato diferencial por edad (senior_citizen); transparencia ante el cliente |
