# Plantilla MLOps – Gerencia de Proyectos de Analítica de Datos

Repositorio base del **Proyecto Final** (Maestría en Ingeniería – Universidad del Quindío).
Implementa de punta a punta el ciclo de vida de un modelo de **deserción de clientes (churn)**:
datos versionados → contrato de datos → entrenamiento registrado → quality gate → API en Docker → monitoreo de drift → runbook.

Funciona sin descargar nada: los datos se generan sintéticamente (imitando *Telco Customer Churn*).
Cada equipo reemplaza la ingesta y el contrato por los de su caso.

| Capacidad | Herramienta | Dónde |
|---|---|---|
| Entorno reproducible | **uv** (`pyproject.toml` + `uv.lock`) | raíz |
| Versionado de datos y pipeline | **DVC** | `dvc.yaml`, `params.yaml`, `dvc.lock` |
| Contrato de datos | **Pandera** | `contracts/` |
| Experimentos y registro de modelos | **MLflow 3** (aliases `@challenger`, `@champion`, `@previous`) | `src/train.py`, `scripts/promote.py` |
| Pruebas | **pytest** (contrato, unitarias, comportamiento del modelo, API) | `tests/` |
| Integración continua | **GitHub Actions** (+ release a GHCR opcional) | `.github/workflows/` |
| Servicio | **FastAPI** + **Docker** | `app/`, `Dockerfile` |
| Monitoreo | **Evidently** (drift), **Prometheus** + **Grafana** (operación) | `monitoring/`, `docker-compose.yml` |
| Documentación de gestión | Charter, gobernanza, model card, runbook, decisiones, post-mortem | `docs/` |

---

## 1. Inicio rápido (5 comandos)

Requisitos: Git, [uv](https://docs.astral.sh/uv/getting-started/installation/) y, para la parte de servicio, Docker Desktop.

```bash
git clone <url-de-su-repositorio> && cd <repositorio>
uv sync                                   # crea .venv con Python 3.12 y las versiones exactas de uv.lock
uv run dvc repro                          # ingest → validate → prepare → train → evaluate
uv run python -m scripts.promote          # quality gate y promoción a @champion
uv run pytest                             # 21 pruebas
```

> ¿Sin uv? `python -m venv .venv`, active el entorno, `pip install -r requirements.txt` y ejecute los
> mismos comandos sin el prefijo `uv run`.

## 2. El pipeline

```
ingest ──► validate ──► prepare ──► train ──► evaluate
(datos)   (contrato)   (train/test) (MLflow)  (métricas, equidad, impacto)
```

```bash
uv run dvc dag                 # ver el grafo
uv run dvc repro               # re-ejecuta solo lo que cambió
uv run dvc metrics show        # métricas actuales
uv run dvc params diff         # qué parámetros cambiaron vs. el último commit
uv run dvc metrics diff main   # comparar con otra rama o tag
```

Experimente cambiando `params.yaml` (por ejemplo `model.type: random_forest`) y vuelva a ejecutar `dvc repro`.

### Métricas que produce `evaluate` (`metrics.json`)
- **Técnicas:** `roc_auc`, `pr_auc`, `precision`, `recall`, `f1` (con el umbral de decisión de `params.yaml`).
- **Equidad:** `recall_gap_senior_citizen` (diferencia de recall entre grupos).
- **Negocio (estimado):** ahorro neto de la campaña de retención vs. contactar al azar (`business` en `params.yaml`).

## 3. MLflow: experimentos y registro de modelos

Por defecto MLflow guarda todo en `mlflow.db` (local, ignorado por Git).

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050   # http://localhost:5050
```

Ciclo de vida con **aliases** (MLflow 3 ya no usa *stages*):

| Alias | Significado | Quién lo asigna |
|---|---|---|
| `@challenger` | último modelo entrenado | `src/train.py` |
| `@champion` | modelo aprobado para producción | `scripts/promote.py` (solo si pasa el quality gate) |
| `@previous` | champion anterior, para rollback | `scripts/promote.py` |

```bash
uv run python -m scripts.quality_gate        # solo evaluar (reports/gate.json)
uv run python -m scripts.promote             # gate + promoción
uv run python -m scripts.promote --rollback  # volver al champion anterior
```

**Trabajo en equipo:** levante el servidor compartido (`docker compose up -d mlflow`) y registre allí:
`export MLFLOW_TRACKING_URI=http://localhost:5001` (en Windows PowerShell: `$env:MLFLOW_TRACKING_URI="http://localhost:5001"`).
Se usa el puerto 5001 porque en macOS el 5000 está ocupado por AirPlay Receiver.

## 4. Pruebas y CI

```bash
uv run pytest -m "not model"   # contrato y unitarias (no requieren modelo)
uv run pytest -m model         # comportamiento del modelo y API (después de dvc repro)
uv run ruff check .            # estilo
uv run pre-commit install      # verificaciones automáticas en cada commit
```

El workflow `.github/workflows/ci.yml` corre en cada push y pull request: lint → pruebas → `dvc repro` →
quality gate → pruebas del modelo → monitoreo de humo → imagen Docker + prueba de humo.
En los pull requests publica en el resumen del job la **diferencia de métricas y parámetros** con la rama base.

> El quality gate en CI solo aplica umbrales absolutos (el registro de MLflow del runner es temporal).
> La comparación contra `@champion` se hace al promover desde el servidor MLflow del equipo.

## 5. Servicio (API) y stack de operación

```bash
uv run uvicorn app.main:app --reload       # API local: http://localhost:8000/docs
docker compose up -d --build               # API + MLflow + Prometheus + Grafana
docker compose down
```

| Servicio | URL |
|---|---|
| API (Swagger) | http://localhost:8000/docs |
| MLflow | http://localhost:5001 |
| Prometheus | http://localhost:9090 |
| Grafana (admin / admin) | http://localhost:3000 → tablero *GPAD – Operación del modelo* |

Endpoints: `GET /health` (versión servida), `POST /predict`, `GET /metrics`.
En el stack Docker cada predicción se guarda en `data/production/predictions.csv` (insumo de monitoreo).

## 6. Monitoreo y simulacro de incidente

```bash
uv run python -m monitoring.make_production_batch --list                 # escenarios
uv run python -m monitoring.make_production_batch --scenario precios     # lote con drift
uv run python -m monitoring.drift_report data/production/batch_precios.csv
uv run python -m monitoring.send_traffic data/production/batch_precios.csv   # tráfico hacia la API
```

| Escenario | Qué simula | Qué lo detecta |
|---|---|---|
| `normal` | mismo proceso del entrenamiento | nada (OK) |
| `precios` | alza de tarifas y migración a fibra | drift de datos (Evidently) |
| `contratos` | captación masiva de clientes nuevos | drift de datos (Evidently) |
| `concepto` | un competidor cambia la relación X → churn (`--with-labels`) | **solo** la caída de desempeño |
| `esquema` | columna renombrada y categoría nueva | contrato de datos (CRÍTICO) |

`drift_report` devuelve código 0 (OK), 1 (ALERTA) o 2 (CRÍTICO). La respuesta a cada nivel está en `docs/runbook.md`.

## 7. Impacto, equidad y explicabilidad (insumos de la model card)

```bash
uv run python -m src.business_analysis                       # ahorro por umbral + sensibilidad a supuestos
uv run python -m src.fairness_report                         # métricas y brechas por grupo (senior_citizen)
uv run python -m src.fairness_report --group internet_service
uv run python -m src.fairness_report --thresholds 0=0.30,1=0.40   # mitigación: umbral por grupo
uv run python -m src.explain                                 # importancia por permutación + coeficientes
```

Resultados en `reports/business/`, `reports/fairness/` y `reports/explain/`: cítelos en `docs/model_card.md`.

## 8. Cómo adaptar la plantilla a su caso

1. **Ingesta** – reemplace `src/ingest.py` (descarga del dataset) o use `dvc add data/raw/<archivo>` y elimine la etapa `ingest` de `dvc.yaml`. Configure un remoto: `uv run dvc remote add -d storage <ruta-o-url>` y `uv run dvc push`.
2. **Columnas** – actualice `src/common.py` (`TARGET`, `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`, `CATEGORIES`).
3. **Contrato** – reescriba `contracts/churn_contract.py` y `tests/test_contract.py`.
4. **API** – ajuste el esquema `Customer` en `app/main.py` para que coincida con el contrato.
5. **Métricas, umbrales e impacto** – `params.yaml` (`gate`, `business`) y `src/evaluate.py` (regresión/pronóstico: cambie las métricas y use partición temporal en `src/prepare.py`).
6. **Pruebas de comportamiento** – reescriba `tests/test_model_behavior.py` con las expectativas de su negocio.
7. **Monitoreo** – adapte los escenarios de `src/synthetic.py` o use lotes reales en `data/production/`.
8. **Documentación** – complete `docs/` en cada entrega y marque el repositorio con los tags `e1` … `e4`.

## 9. Estructura

```
├── app/                    API FastAPI
├── contracts/              contrato de datos (Pandera)
├── data/                   raw/ y processed/ (DVC) · production/ (lotes y log de predicciones)
├── docs/                   charter, gobernanza, model card, runbook, decisiones, post-mortem
├── models/                 model.joblib (DVC) + model_meta.json (trazabilidad)
├── monitoring/             drift (Evidently), lotes simulados, tráfico, Prometheus, Grafana, MLflow server
├── reports/                validación, curva ROC, matriz de confusión, gate, drift
├── scripts/                quality gate, promoción y rollback
├── src/                    etapas del pipeline
├── tests/                  pytest
├── dvc.yaml  params.yaml   pipeline y parámetros
├── Dockerfile  docker-compose.yml
└── pyproject.toml  uv.lock  requirements.txt
```
