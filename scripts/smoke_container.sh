#!/usr/bin/env bash
# Construye la imagen y comprueba el contrato HTTP contra el contenedor real.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ENGINE=${CONTAINER_ENGINE:-podman}
if ! command -v "$ENGINE" >/dev/null 2>&1; then
    ENGINE=docker
fi
if ! command -v "$ENGINE" >/dev/null 2>&1; then
    echo "No se encontró un motor de contenedores; instala Podman o Docker." >&2
    exit 1
fi

IMAGEN=${IMAGEN:-mibici-api:actividad8}
NOMBRE=${NOMBRE:-mibici-api}
PUERTO=${PUERTO:-8000}
URL_BASE="http://127.0.0.1:$PUERTO"

if [[ ! -f model_bundle/MLmodel ]]; then
    echo "Falta model_bundle/MLmodel; ejecuta 'make export-model' antes de continuar." >&2
    exit 1
fi

iniciado=0
limpiar() {
    local codigo=$?
    trap - EXIT
    if (( codigo != 0 )); then
        echo "Falló la comprobación; logs del contenedor $NOMBRE:" >&2
        "$ENGINE" logs "$NOMBRE" >&2 || true
    fi
    if (( iniciado )); then
        "$ENGINE" stop "$NOMBRE" >/dev/null || true
    fi
    exit "$codigo"
}
trap limpiar EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ${SIN_BUILD:-0} != 1 ]]; then
    "$ENGINE" build -t "$IMAGEN" -f Containerfile .
fi
"$ENGINE" run -d --rm --name "$NOMBRE" -p "$PUERTO:8000" "$IMAGEN"
iniciado=1

limite=$((SECONDS + 60))
listo=0
while (( SECONDS < limite )); do
    if codigo=$(curl --silent --output /dev/null --write-out '%{http_code}' \
        --max-time "$((limite - SECONDS))" "$URL_BASE/health") && [[ $codigo == 200 ]]; then
        listo=1
        break
    fi
    if (( SECONDS < limite )); then
        sleep 1
    fi
done
if (( ! listo )); then
    echo "La API no respondió 200 en /health durante los 60 segundos de espera." >&2
    exit 1
fi

echo "Respuesta de GET /health:"
curl --fail --show-error --silent --max-time 10 "$URL_BASE/health"
echo
echo "Respuesta de POST /predict con el ejemplo real de P1:"
curl --fail --show-error --silent --max-time 10 \
    -H 'Content-Type: application/json' \
    --data-binary @docs/serving/ejemplo_entrada_p1.json "$URL_BASE/predict"
echo
MIBICI_API_URL="$URL_BASE" python -m pytest tests/test_serving_e2e.py -q -m contenedor
