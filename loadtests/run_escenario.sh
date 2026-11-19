#!/usr/bin/env bash
# Corre un escenario de Locust contra el stack de Compose ya levantado y deja la evidencia completa.
#
#   bash loadtests/run_escenario.sh u05 5 1      # nombre, usuarios, spawn-rate (usuarios/s)
#   bash loadtests/run_escenario.sh u20 20 2
#
# Salida en docs/evidencias/operacion/carga/<nombre>/:
#   locust_stats.csv, locust_failures.csv   ventana estable (tras el calentamiento)
#   locust_stats_history.csv                 toda la corrida, una fila por segundo
#   locust.html                              reporte de Locust
#   recursos.csv                             docker stats de la API y MLflow durante la corrida
#   ventana.json, manifiesto.json            ventana, conteos, commit, modelo y máquina
#
# Variables: CALENTAMIENTO (s, 60), VENTANA (s, 180), ESPERA_MIN/ESPERA_MAX (s, 0.5/1.5),
# API_URL, MLFLOW_URL, DOCKER (docker|podman), PERMITIR_MODELO_PRUEBA=1 para ensayos con el
# modelo de prueba (por defecto se rechaza: las cifras oficiales deben ser del champion real).
set -euo pipefail

NOMBRE="${1:?uso: run_escenario.sh <nombre> <usuarios> <spawn-rate>}"
USUARIOS="${2:?usuarios}"
SPAWN="${3:?spawn-rate}"
CALENTAMIENTO="${CALENTAMIENTO:-60}"
VENTANA="${VENTANA:-180}"
API_URL="${API_URL:-http://127.0.0.1:8000}"
MLFLOW_URL="${MLFLOW_URL:-http://127.0.0.1:5001}"
DOCKER="${DOCKER:-docker}"
PYTHON="${PYTHON:-python}"
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$RAIZ/docs/evidencias/operacion/carga/$NOMBRE"

cd "$RAIZ"
mkdir -p "$DIR"

# 1) El stack debe estar sano y servir el champion real (origen = bundle)
curl -sf "$API_URL/health" > "$DIR/health.json" || { echo "La API no está sana en $API_URL" >&2; exit 1; }
curl -sf "$MLFLOW_URL/api/2.0/mlflow/model-versions/get?name=mibici-equipo16-p1-demanda&version=1" \
  > "$DIR/modelo_registry.json"
ORIGEN="$("$PYTHON" -c "import json,sys; d=json.load(open(sys.argv[1]))['model_version']; print({t['key']:t['value'] for t in d.get('tags',[])}.get('origen',''))" "$DIR/modelo_registry.json")"
if [[ "$ORIGEN" != "bundle" && "${PERMITIR_MODELO_PRUEBA:-0}" != "1" ]]; then
  echo "El registry del stack tiene origen='$ORIGEN', no el champion real (bundle). Ver contrato_arranque.md" >&2
  exit 1
fi

# 2) CPU/RAM en paralelo durante toda la corrida
bash loadtests/capturar_recursos.sh "$DIR/recursos.csv" &
PID_RECURSOS=$!
trap 'kill "$PID_RECURSOS" 2>/dev/null || true' EXIT

# 3) Locust headless: calentamiento + ventana estable
DURACION=$(( CALENTAMIENTO + VENTANA ))
echo ">> $NOMBRE: $USUARIOS usuarios, spawn $SPAWN/s, ${CALENTAMIENTO}s calentamiento + ${VENTANA}s ventana"
LOCUST_CALENTAMIENTO="$CALENTAMIENTO" LOCUST_VENTANA_JSON="$DIR/ventana.json" \
LOCUST_ESPERA_MIN="${ESPERA_MIN:-0.5}" LOCUST_ESPERA_MAX="${ESPERA_MAX:-1.5}" \
  "$PYTHON" -m locust -f loadtests/locustfile.py --host "$API_URL" --headless \
    -u "$USUARIOS" -r "$SPAWN" --run-time "${DURACION}s" --stop-timeout 10 \
    --csv "$DIR/locust" --csv-full-history --html "$DIR/locust.html" --only-summary \
    --exit-code-on-error 0 2>&1 | tee "$DIR/locust_resumen.txt"

kill "$PID_RECURSOS" 2>/dev/null || true
wait "$PID_RECURSOS" 2>/dev/null || true

# 4) Manifiesto de la corrida (lo consume FinOps)
USUARIOS="$USUARIOS" SPAWN="$SPAWN" DOCKER="$DOCKER" \
  "$PYTHON" loadtests/manifiesto.py "$DIR"
echo ">> Evidencia en $DIR"
