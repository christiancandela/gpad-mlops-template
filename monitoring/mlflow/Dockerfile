# Servidor de tracking y registro de modelos compartido por el equipo.
FROM python:3.12-slim
RUN pip install --no-cache-dir mlflow==3.17.1
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", "--serve-artifacts", \
     "--allowed-hosts", "localhost,localhost:*,127.0.0.1,127.0.0.1:*,mlflow,mlflow:*"]
