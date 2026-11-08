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

COPY model_bundle/ ./model_bundle/

RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --home-dir /app --no-create-home appuser && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "mibici.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
