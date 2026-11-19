#!/usr/bin/env bash
# Muestrea CPU y RAM de los contenedores del stack con `docker stats` hasta recibir SIGTERM/SIGINT.
#
#   bash loadtests/capturar_recursos.sh docs/evidencias/operacion/carga/u05/recursos.csv
#
# Cada fila: timestamp UTC, contenedor, % de CPU (100 % = 1 núcleo) y memoria usada / límite.
# DOCKER="podman" usa `podman stats`. CONTENEDORES e INTERVALO se pueden cambiar por entorno.
set -euo pipefail

SALIDA="${1:?uso: capturar_recursos.sh <salida.csv>}"
DOCKER="${DOCKER:-docker}"
INTERVALO="${INTERVALO:-1}"
read -r -a CONTENEDORES <<< "${CONTENEDORES:-mibici-api-1 mibici-mlflow-1}"

mkdir -p "$(dirname "$SALIDA")"
echo "timestamp_utc,contenedor,cpu_pct,memoria" > "$SALIDA"
trap 'exit 0' TERM INT
while true; do
  "$DOCKER" stats --no-stream \
    --format "$(date -u +%FT%TZ),{{.Name}},{{.CPUPerc}},{{.MemUsage}}" \
    "${CONTENEDORES[@]}" >> "$SALIDA"
  sleep "$INTERVALO"
done
