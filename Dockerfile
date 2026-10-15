# Imagen de la API de inferencia. Solo instala las dependencias de ejecución (sección [project]
# de pyproject.toml) con las versiones exactas de uv.lock: mismas versiones que en el entrenamiento.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups

COPY app/ app/
# El modelo se genera con `dvc repro` (o `dvc pull`) antes de construir la imagen.
COPY models/model.joblib models/model_meta.json models/

# Se ejecuta sin privilegios de root (los archivos de /app solo necesitan lectura).
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
