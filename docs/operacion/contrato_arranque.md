# Contrato de arranque del sistema (Entregable 7)

Acuerdo entre el stack de Docker Compose (PR de CH1) y las pruebas de carga (LH3), el pipeline de
CI/CD (HH2) y el análisis FinOps y el reporte (DH4). Todo lo que sigue se verificó el 2026-11-18 con
Docker 29.6 y Docker Compose 5.3 en macOS arm64; la evidencia está en
[`docs/evidencias/operacion/compose/`](../evidencias/operacion/compose/).

## Arquitectura

```
                  red "mibici" (bridge)
 host:5001 ──► mlflow:5000  ◄── registry + artefactos servidos ── volumen mlflow-datos (/mlflow)
                  ▲   ▲
     (1) registra │   │ (2) carga models:/…@champion por HTTP
                  │   │
           model-init  api:8000 ◄── host:8000   POST /predict · GET /health · GET /docs
           (una vez)
```

| Servicio | Qué hace | Salud | Recursos (límite por defecto) |
|---|---|---|---|
| `mlflow` | Servidor de tracking y Model Registry. SQLite y artefactos en el volumen `mlflow-datos`; artefactos servidos por HTTP (`--serve-artifacts`) | `GET /health` | 1 CPU, 1 GiB |
| `model-init` | Registra el champion de P1 en ese registry y termina (`python -m mibici.serving.inicializar`) | Sale con código 0 | Sin límite (dura segundos) |
| `api` | La API FastAPI de la semana 8. Carga `models:/mibici-equipo16-p1-demanda@champion` desde `mlflow` | `GET /health` = 200 solo con el modelo cargado | 2 CPU, 2 GiB |

Orden garantizado: `mlflow` sano → `model-init` terminado con éxito → `api` arranca y queda sana al
cargar el modelo. Los tres usan **una sola imagen** (`mibici-ops:local`, del `Containerfile`), que
ya trae MLflow 3.17.2 fijado; cliente y servidor tienen la misma versión.

## Arranque

```bash
make export-model MLFLOW_PORT=5001     # opcional: copia del champion real (ver "Modelo")
docker compose up --build -d           # o: make up      (Podman: podman compose up --build -d)
docker compose ps                      # api y mlflow "healthy"; model-init "Exited (0)"
make smoke-compose                     # salud, predicción y persistencia, y detiene el stack
docker compose down                    # conserva el volumen; down -v (make down-v) lo borra
```

Interfaz de MLflow del stack: <http://127.0.0.1:5001>. Variables de puertos, procesos, recursos y
modelo: [`.env.example`](../../.env.example).

## Modelo que sirve el stack

`model-init` decide qué registrar (idempotente; nunca duplica versiones):

| Situación | Qué registra | Uso |
|---|---|---|
| Se exportó el champion antes del build (`make export-model` → `model_bundle/`, que entra a la imagen) | La copia exacta (misma firma, archivos y ejemplo) con su linaje | **Mediciones oficiales** (Locust y FinOps) |
| Clon limpio sin `model_bundle/` (runner de CI) | Un **modelo de prueba** determinista con la misma firma de 11 variables, etiquetado `origen=modelo-de-prueba` | Solo verificar que el sistema arranca y predice |
| Registry ya inicializado (volumen existente) | Nada (`"accion": "sin_cambios"`) | Recrear contenedores conserva el estado |

En el registry del stack la copia queda como **versión 1**, con alias `champion` y estas etiquetas
de linaje: `version_origen = 2` (champion del registry del equipo), `run_id_origen =
47933c5b0df749edaeb159289053f70e`, `git_commit = c30cd49`, `version_datos = 4d8c2a1292fc`.
`/health` reporta la versión del registry del stack (1); para saber de dónde viene:

```bash
curl -s "http://127.0.0.1:5001/api/2.0/mlflow/model-versions/get?name=mibici-equipo16-p1-demanda&version=1"
```

**Antes de medir, confirmen `"origen": "bundle"` en esas etiquetas**: si dice `modelo-de-prueba`,
el stack se construyó sin exportar el champion y las cifras no son del modelo real.

## Petición y respuesta

Las mismas de la semana 8 ([`docs/serving/contrato_p1.md`](../serving/contrato_p1.md)).

```bash
curl -s -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
  -d @docs/serving/ejemplo_entrada_p1.json
# {"predicciones":[40.697957479000046,276.77983750628846],"modelo":{"nombre":"mibici-equipo16-p1-demanda","version":"1",...}}
```

El ejemplo trae **2 registros, así que cada petición produce 2 predicciones**. Para FinOps, cuenten
predicciones exitosas (no peticiones) o usen un payload de un registro. Respuesta 200 válida:
`predicciones` con un `float` > 0 por registro; 422 si la entrada no cumple el contrato.

## Notas por integrante

**LH3 · Locust y consumo.**
- Contenedores a medir: `mibici-api-1` y `mibici-mlflow-1`. Durante la carga, MLflow casi no
  trabaja: el modelo se carga una vez al arrancar la API.
  ```bash
  while true; do docker stats --no-stream --format "$(date -u +%FT%TZ),{{.Name}},{{.CPUPerc}},{{.MemUsage}}" \
    mibici-api-1 mibici-mlflow-1; sleep 2; done >> docs/evidencias/operacion/carga/recursos.csv
  ```
- Consumo medido en reposo: API ~200 MiB y MLflow ~360 MiB.
- `API_WORKERS` cambia el número de procesos de Uvicorn. Cada uno carga el modelo, y es la palanca
  natural para el escenario de más usuarios.
- Si Locust corre en la misma máquina, indíquenlo en el reporte: compite por CPU con el stack.

**HH2 · GitHub Actions.**
- `docker compose up --build -d` funciona en un clon limpio sin `model_bundle/`: se registra el
  modelo de prueba y la API predice. Lo comprueba
  [`clon_limpio_sin_modelo.txt`](../evidencias/operacion/compose/clon_limpio_sin_modelo.txt).
- [`scripts/smoke_compose.sh`](../../scripts/smoke_compose.sh) sirve tal cual como paso de smoke
  test en CI. Sale con error y muestra los logs si algo falla, y siempre detiene el stack.
- `IMAGEN_TAG=${{ github.sha }}` etiqueta la imagen con el commit. Localmente, si construyen dos
  copias del repo en la misma máquina, usen un `IMAGEN_TAG` distinto en cada una: con la misma
  etiqueta, el último build reemplaza la imagen del otro (por ejemplo, una imagen sin el champion).
- `tests/test_compose.py` valida la estructura del Compose sin Docker; corre dentro de `pytest`.
- Antes del workflow hay que corregir el Ruff de `mibici/monitoring/train_challenger.py`, que hoy
  falla en `main` por el orden de imports. Se resuelve con `ruff check --fix` y `ruff format`.

**DH4 · FinOps y reporte.**
- Los límites (API 2 CPU y 2 GiB; MLflow 1 CPU y 1 GiB) acotan el stack al tamaño de una instancia
  de referencia. Se cambian con `API_CPUS`, `API_MEMORIA`, `MLFLOW_CPUS` y `MLFLOW_MEMORIA`.
- La imagen pesa 1.3 GB, por MLflow completo, scikit-learn y SciPy.
- Optimización ya medida para el reporte: MLflow 3 arranca por defecto siete procesos `huey` para
  tareas de GenAI que el proyecto no usa. Con ellos usa **1.6 GiB y ~100 % de CPU en reposo**; con
  `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` baja a **~360 MiB y 0.2 % de CPU**. Ese cambio hace
  que MLflow quepa en una instancia más pequeña.

## Decisiones verificadas

- **Registry propio del stack.** El `mlflow.db` de cada integrante guarda rutas absolutas de su
  máquina (semana 8). El stack usa un registry nuevo con artefactos servidos por HTTP: funciona
  igual en cualquier equipo y en CI, y la API no depende de ningún disco local.
- **`--allowed-hosts`.** MLflow 3.17 responde 403 a peticiones con `Host: mlflow:5000`; por defecto
  solo acepta `localhost` e IP privadas (comprobado con `curl`). Se permite explícitamente el host
  interno en lugar de desactivar la protección.
- **Copia del modelo opcional en la imagen.** `COPY model_bundl[e]` incluye `model_bundle/` si
  existe y no falla si no existe. La imagen de la semana 8 sigue funcionando sola con el modelo
  empaquetado, y un clon limpio también construye.
- **Persistencia.** `docker compose down` + `up` conserva la misma versión y corrida: `model-init`
  responde `sin_cambios`. Lo comprueba `make smoke-compose`.
- **Reinicio.** `restart: unless-stopped` en `api` y `mlflow`: cuando Docker Desktop se cerró
  durante las pruebas, ambos servicios volvieron solos y sanos al reabrirlo.
