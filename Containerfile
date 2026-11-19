FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/model_bundle

COPY requirements-serving.txt ./
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY pyproject.toml README.md LICENSE ./
COPY mibici/ ./mibici/
RUN pip install --no-cache-dir --no-deps .

# Copia opcional del champion (make export-model). El patrón [e] no falla si la carpeta no existe:
# así un clon limpio (CI) también construye, y en Docker Compose el modelo llega desde MLflow.
COPY model_bundl[e] ./model_bundle/

RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --home-dir /app --no-create-home appuser && \
    mkdir -p /mlflow && \
    chown -R appuser:appgroup /app /mlflow

USER appuser

EXPOSE 8000

CMD ["uvicorn", "mibici.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
