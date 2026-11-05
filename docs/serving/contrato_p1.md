# Contrato de inferencia del modelo servido (P1)

Acuerdo entre el módulo de carga (`mibici/serving/model_loader.py`, PR de CH1) y la API
(PR de LH3), el contenedor (PR de HH2) y la documentación final (PR de DH4). Todo lo que sigue se
verificó contra el Model Registry del equipo el 2026-11-04.

## Modelo

| Campo | Valor |
|---|---|
| Nombre en el registry | `mibici-equipo16-p1-demanda` |
| Alias / versión | `champion` → versión **2** (la versión 1 conserva `challenger`) |
| URI | `models:/mibici-equipo16-p1-demanda@champion` |
| Corrida de origen | `p1-hgb-featuresv1-learning_rate0.05-max_iter100-max_leaf_nodes63` (`47933c5b0df749edaeb159289053f70e`) |
| Linaje | commit `c30cd49`, versión de datos `4d8c2a1292fc`, `val_mae` = 6.14, supera al baseline naive |
| Algoritmo | `HistGradientBoostingRegressor` con pérdida Poisson (scikit-learn 1.9.1, serializado con skops) |
| Qué predice | Viajes que inician en una estación en un día, con información disponible hasta el día anterior |

Se eligió P1 porque su firma es tabular y completa (11 variables numéricas), el modelo predice
observaciones nuevas sin reentrenar y la salida (viajes esperados) es directa de interpretar.

## Entrada

Una petición contiene uno o varios registros. Cada registro tiene **exactamente** estas variables;
el módulo rechaza variables faltantes o no esperadas. Las columnas se reordenan según la firma, así
que el orden en el JSON no importa.

| Variable | Tipo (firma) | Obligatoria | Significado | Rango en train |
|---|---|---|---|---|
| `dia_semana` | integer | sí | 0 = lunes … 6 = domingo | 0–6 |
| `mes` | integer | sí | Mes del día a predecir | 1–12 |
| `es_fin_semana` | long | sí | 1 si `dia_semana` ≥ 5 | 0–1 |
| `dia_anio` | integer | sí | Día del año | 1–366 |
| `es_festivo` | long | sí | 1 si es festivo oficial en México | 0–1 |
| `demanda_lag_1` | double | no | Viajes del día anterior | 0–561 |
| `demanda_lag_7` | double | no | Viajes de siete días antes | 0–561 |
| `demanda_media_7` | double | no | Media de los siete días anteriores | 0–289.1 |
| `capacidad` | integer | sí | Anclajes de la estación | 11–39 |
| `latitud` | double | sí | Latitud de la estación | 20.636–20.738 |
| `longitud` | double | sí | Longitud de la estación | −103.425 – −103.301 |

- Los tres rezagos son opcionales porque faltan al inicio del historial de una estación (0.2 % a
  1.3 % de train); el modelo maneja los nulos de forma nativa. En JSON se envían como `null` o se
  omiten.
- Las variables obligatorias no admiten nulos, texto, booleanos ni infinitos; las enteras no
  admiten decimales. Estas reglas vienen de la firma y las aplica `predecir`.
- Los rangos son los observados en entrenamiento, no límites del modelo. Para la validación en
  Pydantic se sugieren reglas de dominio: `dia_semana` 0–6, `mes` 1–12, indicadores 0/1, `dia_anio`
  1–366, rezagos ≥ 0, `capacidad` > 0 y coordenadas dentro del área metropolitana de Guadalajara.

Ejemplo real: [`ejemplo_entrada_p1.json`](ejemplo_entrada_p1.json), con las estaciones 55 y 51 del
30 de septiembre de 2026 (conjunto de prueba, sin datos personales). La demanda observada ese día
fue de 32 y 442 viajes.

## Salida

Una lista de `float` finitos y positivos, en el mismo orden de los registros: viajes esperados.
Para el ejemplo anterior, el champion versión 2 devuelve:

```json
{
  "modelo": {"nombre": "mibici-equipo16-p1-demanda", "version": "2", "alias": "champion",
             "run_id": "47933c5b0df749edaeb159289053f70e"},
  "predicciones": [40.697957479000046, 276.77983750628846]
}
```

## Uso desde la API

```python
from mibici.serving.model_loader import ErrorEntrada, ModeloNoDisponible, cargar_modelo

servido = cargar_modelo()          # una vez, al arrancar la aplicación
servido.predecir(registros)        # dict o lista de dicts -> list[float]
servido.metadatos                  # nombre, version, alias, run_id, linaje y origen
servido.contrato()                 # variables, tipos y obligatoriedad en orden de la firma
```

Correspondencia sugerida con HTTP: `ErrorEntrada` → 422; `ModeloNoDisponible` → 503 (por ejemplo,
en `/health`); `RuntimeError` del modelo → 500 sin exponer la traza.

## Carga del modelo y contenedor

Estrategia elegida: **empaquetar en la imagen la versión descargada del registry**.

```bash
make mlflow                 # servidor del equipo; en macOS el 5000 lo ocupa AirPlay
make export-model           # descarga @champion a model_bundle/ y predice el ejemplo
# con otro puerto, el mismo en ambos: make mlflow MLFLOW_PORT=5001 / make export-model MLFLOW_PORT=5001
```

`make export-model` resuelve el alias a la versión 2, descarga esa versión fija a
`model_bundle/` (ignorada por Git), escribe el linaje en `model_bundle/modelo_servido.json` y
comprueba que la copia carga y predice. Al arrancar, `cargar_modelo()` usa `model_bundle/` si
existe y, si no, carga `MODEL_URI` desde `MLFLOW_TRACKING_URI`.

| Variable de entorno | Por defecto | Uso |
|---|---|---|
| `MODEL_PATH` | `<repo>/model_bundle` | Copia local del modelo; en la imagen, la ruta donde se copie |
| `MODEL_URI` | `models:/mibici-equipo16-p1-demanda@champion` | Modelo del registry si no hay copia local |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | Servidor de MLflow para exportar o cargar en línea |

Hallazgos verificados que justifican la estrategia:

- El registry guarda los artefactos con **rutas absolutas del equipo anfitrión**
  (`…/tc5061-proyecto-equipo16/mlruns/1/models/…`) y `make mlflow` no usa `--serve-artifacts`.
  Un contenedor que solo recibe `MLFLOW_TRACKING_URI=http://host.containers.internal:5000`
  resolvería la versión, pero no podría leer el modelo. Con la copia empaquetada, la inferencia no
  depende del servidor ni de esas rutas.
- La copia carga y predice lo mismo en un entorno mínimo **sin el paquete `mibici`**, con solo
  `mlflow-skinny==3.17.2`, `scikit-learn==1.9.1`, `skops==0.16.0`, `pandas==3.0.6`,
  `numpy==2.5.4` y `scipy==1.18.1` (Python 3.13). El módulo `model_loader` además usa `loguru` y
  `python-dotenv`, por medio de `mibici.config`.
- MLflow avisa que el modelo se registró con `mlflow==3.17.1` y el entorno tiene 3.17.2 (la
  corrección del CVE-2026-53995). Es una diferencia de versión de parche y no cambia las
  predicciones.
