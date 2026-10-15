# Atajos (opcionales). En Windows sin make, ejecute directamente los comandos `uv run ...`.
.PHONY: setup repro test gate promote rollback api mlflow stack monitor

setup:      ## instala dependencias y hooks
	uv sync
	uv run pre-commit install

repro:      ## ejecuta el pipeline completo (solo las etapas que cambiaron)
	uv run dvc repro

test:       ## todas las pruebas
	uv run pytest

gate:       ## quality gate sobre el último modelo
	uv run python -m scripts.quality_gate

promote:    ## gate + @challenger → @champion
	uv run python -m scripts.promote

rollback:   ## @previous → @champion
	uv run python -m scripts.promote --rollback

api:        ## API local en http://localhost:8000/docs
	uv run uvicorn app.main:app --reload

mlflow:     ## interfaz de MLflow sobre el registro local
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050

stack:      ## API + MLflow + Prometheus + Grafana con Docker
	docker compose up -d --build

monitor:    ## ejemplo: lote con alza de precios + reporte de drift
	uv run python -m monitoring.make_production_batch --scenario precios
	uv run python -m monitoring.drift_report data/production/batch_precios.csv
