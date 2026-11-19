#!/usr/bin/env bash
# Prueba de humo del stack de Docker Compose (Entregable 7), local o en CI:
#   1. levanta mlflow, model-init y api con una sola orden (docker compose up --build -d);
#   2. espera a que la API esté sana (modelo cargado desde el MLflow del stack);
#   3. comprueba que model-init terminó con éxito y que MLflow responde desde el host;
#   4. envía la petición del contrato (docs/serving/ejemplo_entrada_p1.json) y valida la respuesta;
#   5. con PERSISTENCIA=1, recrea los contenedores sin borrar el volumen y verifica que el registry
#      conserva la misma versión y corrida (no se registra de nuevo);
#   6. detiene el stack (down -v), salvo MANTENER=1. Si algo falla, muestra los logs.
#
#   bash scripts/smoke_compose.sh                 # Docker
#   COMPOSE="podman compose" bash scripts/smoke_compose.sh
#   PERSISTENCIA=1 MANTENER=1 bash scripts/smoke_compose.sh
set -euo pipefail

cd "$(dirname "$0")/.."
COMPOSE=${COMPOSE:-docker compose}
API="http://127.0.0.1:${API_HOST_PORT:-8000}"
MLFLOW="http://127.0.0.1:${MLFLOW_HOST_PORT:-5001}"
EJEMPLO=docs/serving/ejemplo_entrada_p1.json
ESPERA=${ESPERA:-300}

fallar() {
  echo "ERROR: $*" >&2
  $COMPOSE ps -a >&2 || true
  $COMPOSE logs --no-color --tail=80 >&2 || true
  exit 1
}

terminar() {
  if [[ "${MANTENER:-0}" != "1" ]]; then
    $COMPOSE down -v --remove-orphans >/dev/null 2>&1 || true
  fi
}
trap terminar EXIT

esperar_api() {
  local limite=$((SECONDS + ESPERA))
  until curl -fsS "$API/health" >/dev/null 2>&1; do
    ((SECONDS < limite)) || fallar "la API no estuvo sana en ${ESPERA} s"
    sleep 3
  done
}

salud() { curl -fsS "$API/health"; }

echo "== Levantando el stack"
$COMPOSE up --build -d
esperar_api

estado=$($COMPOSE ps -a --format '{{.Service}} {{.State}} {{.ExitCode}}' | awk '$1=="model-init"{print $2, $3}')
[[ "$estado" == "exited 0" ]] || fallar "model-init no terminó con éxito ($estado)"
curl -fsS "$MLFLOW/health" >/dev/null || fallar "MLflow no responde en $MLFLOW"
$COMPOSE ps

echo "== Salud de la API"
antes=$(salud)
echo "$antes"

echo "== Predicción con $EJEMPLO"
respuesta=$(curl -fsS -X POST "$API/predict" -H 'Content-Type: application/json' -d @"$EJEMPLO")
echo "$respuesta"
python3 - "$respuesta" "$EJEMPLO" <<'PY' || fallar "la respuesta de /predict no cumple el contrato"
import json, math, sys
respuesta, ejemplo = json.loads(sys.argv[1]), json.load(open(sys.argv[2], encoding="utf-8"))
predicciones = respuesta["predicciones"]
assert len(predicciones) == len(ejemplo["registros"]), "una predicción por registro"
assert all(isinstance(p, float) and math.isfinite(p) and p > 0 for p in predicciones)
assert respuesta["modelo"]["nombre"] == "mibici-equipo16-p1-demanda"
assert respuesta["modelo"]["alias"] == "champion"
PY

if [[ "${PERSISTENCIA:-0}" == "1" ]]; then
  echo "== Persistencia: down (sin -v) y up"
  $COMPOSE down
  $COMPOSE up -d
  esperar_api
  despues=$(salud)
  echo "$despues"
  [[ "$antes" == "$despues" ]] || fallar "el registry cambió al recrear los contenedores"
  $COMPOSE logs --no-color model-init | grep -q '"accion": "sin_cambios"' \
    || fallar "model-init volvió a registrar el modelo"
fi

echo "OK: stack sano, modelo cargado desde MLflow y predicción válida"
