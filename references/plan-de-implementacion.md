# Plan de implementación — Equipo 16

Cuatro Pull Requests, uno por integrante, integrados en orden. Cada PR deja `main` funcionando
(`make lint` y `make test` pasan) y respeta los contratos de abajo para que los siguientes puedan
avanzar en paralelo.

| Orden | Rama | Abre | Revisa | Resultado |
|---|---|---|---|---|
| 1 | `feat/base-del-proyecto` | CH1 | HH2 | Flujo Git, propuesta, ML Canvas, arquitectura, `config.py`, `tracking.py`, dependencias fijas |
| 2 | `feat/ingesta-datos` | LH3 | DH4 | `make data` descarga, normaliza y versiona los 24 meses en Parquet |
| 3 | `feat/features` | DH4 | CH1 | Notebook de exploración y `features.py` para P1, P2 y P3 |
| 4 | `feat/modelos-mlflow` | HH2 | LH3 | `make train` entrena baselines y candidatos y registra los runs en MLflow |

**Cómo trabajar antes de que se integre el PR anterior:** crear la rama desde `main` en cuanto el
PR previo esté integrado. Mientras tanto se puede programar contra los contratos con datos
sintéticos (DH4 y HH2) o probar con la rama del compañero (`git fetch` + `git switch <rama>`),
sin hacer merge de ella.

## Reglas comunes

- Rutas, fechas, semillas, URLs y parámetros **solo** en `mibici/config.py`; los demás módulos los importan.
- Cada función pública lleva docstring (qué hace y a qué pregunta sirve) y al menos una prueba en `tests/`
  con datos sintéticos. Las pruebas no descargan nada ni dependen de `data/`.
- Scripts con `typer` y `loguru`, como la plantilla. Nada de `print`.
- Ningún dato personal (`Usuario_Id`, `Genero`, `Año_de_nacimiento`) después de la ingesta.
- Si un PR necesita cambiar un contrato, se acuerda con el equipo y se documenta en este archivo en el mismo PR.

## Contratos

### Configuración (`mibici/config.py`, PR 1)

| Nombre | Valor / uso |
|---|---|
| `FECHA_INICIO`, `INICIO_VALIDACION`, `INICIO_PRUEBA`, `FECHA_FIN` | `2024-10-01`, `2026-04-01`, `2026-07-01`, `2026-09-30` |
| `SEMILLA` | `16` |
| `URLS_VIAJES`, `URL_NOMENCLATURA` | `{"2024_10": url, …}` (24 meses) y URL del catálogo de estaciones |
| `VIAJES_DIR`, `ESTACIONES_PARQUET` | `data/processed/viajes/`, `data/processed/estaciones.parquet` |
| `MANIFIESTO_DATOS` | `references/manifiesto_datos.json` (versionado en Git) |
| `ALFA_RIESGO`, `HORAS_FRANJA` | `0.5`, `3` |
| `EXPERIMENTO_MLFLOW`, `EQUIPO`, `MLFLOW_TRACKING_URI` | `mibici-equipo16`, `16`, `http://127.0.0.1:5000` (sobrescribible en `.env`) |

### Datos procesados (PR 2)

`data/processed/viajes/viajes_AAAA_MM.parquet`, un archivo por mes:

| Columna | Tipo | Origen |
|---|---|---|
| `viaje_id` | int64 | `Viaje_Id` |
| `inicio` | datetime64 | `Inicio_del_viaje` (acepta `AAAA-MM-DD HH:MM:SS` y `DD/MM/AAAA H:MM`) |
| `fin` | datetime64 | `Fin_del_viaje` |
| `origen_id` | int32 | `Origen_Id` |
| `destino_id` | int32 | `Destino_Id` |

`data/processed/estaciones.parquet`:

| Columna | Tipo | Origen |
|---|---|---|
| `estacion_id` | int32 | `id` |
| `nombre` | string | `name` sin el prefijo `ex-` |
| `municipio` | string | prefijo del nombre: `GDL`, `ZPN` o `TLQ` |
| `latitud`, `longitud` | float64 | `latitude`, `longitude` |
| `capacidad` | int32 | `dpcapacity` |
| `activa` | bool | `False` si el nombre empieza con `ex-` |

### Manifiesto de datos (PR 2)

`references/manifiesto_datos.json`, ordenado por archivo y **sin fechas de generación** (si no, la
versión cambiaría en cada ejecución aunque los datos fueran iguales):

```json
{
  "fuente": "https://mibici.net/es/datos-abiertos/",
  "fecha_inicio": "2024-10-01",
  "fecha_fin": "2026-09-30",
  "archivos": [
    {"archivo": "datos_abiertos_2024_10.csv", "sha256": "<hash del CSV descargado>",
     "filas": 0, "parquet": "viajes/viajes_2024_10.parquet"}
  ]
}
```

El hash se calcula sobre el CSV descargado (inmutable), no sobre el Parquet. `tracking.version_datos()`
toma los primeros 12 caracteres del SHA-256 de este archivo y lo registra en cada run.

### Lectura de datos (PR 2) y features (PR 3)

| Módulo | Función | Devuelve |
|---|---|---|
| `dataset.py` | `cargar_viajes(meses: list[str] \| None = None)` | DataFrame con el esquema de viajes |
| `dataset.py` | `cargar_estaciones()` | DataFrame con el esquema de estaciones |
| `features.py` | `limpiar_viajes(viajes, estaciones)` | Viajes sin duplicados, estaciones desconocidas ni duraciones negativas |
| `features.py` | `variables_calendario(fechas)` | Hora, día de la semana, mes, fin de semana, festivo |
| `features.py` | `demanda_diaria(viajes)` | `estacion_id`, `fecha`, `viajes` (incluye días con 0) — P1 |
| `features.py` | `agregar_rezagos(df, columna, rezagos, por="estacion_id")` | Columnas `<columna>_rezago_<n>` usando solo el pasado |
| `features.py` | `flujo_por_franja(viajes, horas_franja=HORAS_FRANJA)` | `estacion_id`, `franja_inicio`, `salidas`, `llegadas`, `flujo_neto` — P2 |
| `features.py` | `etiquetar_riesgo(flujo, estaciones, alfa=ALFA_RIESGO)` | Serie `sin_bicicletas` / `sin_anclajes` / `normal` — P2 |
| `features.py` | `perfiles_estacion(viajes, hasta=INICIO_VALIDACION)` | Una fila por estación con perfiles horarios normalizados — P3 |
| `features.py` | `particion_temporal(df, columna_fecha)` | `(entrenamiento, validacion, prueba)` con los cortes de `config.py` |

### Registro en MLflow (PR 1, usado en PR 4)

Todo run se abre con `mibici.tracking.iniciar_run(pregunta, nombre, params)` y sus métricas se
registran con `registrar_metricas()`, que exige los prefijos `train_`, `val_` o `test_`.
Nombres de run: `p<n>-<modelo>`, por ejemplo `p1-naive-estacional` o `p2-hgb-balanceado`.

## Detalle por PR

### PR 1 · Base del proyecto (CH1, revisa HH2)

- [x] README con integrantes, roles y flujo Git; plantilla de PR; `.gitignore` ampliado.
- [x] `references/propuesta-parte-a.md`, `references/ml-canvas.md`, `references/arquitectura.md` y este plan.
- [x] `config.py` con los contratos; `tracking.py`; pruebas `test_config.py` y `test_tracking.py`.
- [x] `requirements.txt` con versiones fijas; `.env.example`.

### PR 2 · Ingesta de datos (LH3, revisa DH4)

- `dataset.py` con `typer`: `python -m mibici.dataset` (y `make data`) descarga los meses de
  `URLS_VIAJES` y la nomenclatura a `data/raw/`, sin volver a descargar lo que ya existe.
- Lectura con UTF-8 y caída a latin-1 por archivo; ambos formatos de fecha (octubre 2024 usa `DD/MM/AAAA H:MM`).
- Elimina `Usuario_Id`, `Genero` y `Año_de_nacimiento` antes de escribir Parquet.
- Escribe los Parquet y el manifiesto con el formato acordado; `cargar_viajes()` y `cargar_estaciones()`.
- Pruebas con CSV sintéticos: uno UTF-8, uno latin-1 y uno con el formato de octubre de 2024;
  el manifiesto es idéntico si se ejecuta dos veces.
- Actualizar en el Makefile la regla `data` y documentar el comando en el README.
- Reportar en el PR las cifras reales: viajes por mes, total y estaciones con viajes.

### PR 3 · Exploración y features (DH4, revisa CH1)

- `notebooks/1.0-dh4-exploracion-viajes.ipynb`: tamaño real, estaciones sin viajes o fuera del
  catálogo, duraciones atípicas, distribución del flujo neto por franja (para confirmar `ALFA_RIESGO`
  y `HORAS_FRANJA`). Figuras en `reports/figures/`.
- `features.py` con las funciones del contrato.
- Pruebas de no fuga: un cambio en datos posteriores a una fecha no cambia las variables de esa fecha.
- Si la exploración cambia α, la franja o la limpieza, actualizar `config.py` y `propuesta-parte-a.md`.

### PR 4 · Modelos y experimento (HH2, revisa LH3)

- `modeling/p1_demanda.py`: naive estacional, promedio por estación × día de la semana y
  `HistGradientBoostingRegressor(loss="poisson")`; MAE, RMSE y desviación Poisson en validación y prueba.
- `modeling/p2_riesgo.py`: `DummyClassifier`, persistencia, `LogisticRegression(class_weight="balanced")`
  y `HistGradientBoostingClassifier`; F1 macro, recall por clase, average precision y balanced accuracy.
- `modeling/p3_clusters.py`: agrupación por municipio, `KMeans` y `AgglomerativeClustering(linkage="ward")`;
  silueta, Davies-Bouldin, Calinski-Harabasz y ARI entre semillas.
- `plots.py`: real contra predicho (P1), matriz de confusión (P2), mapa de clusters (P3), registrados como artefactos.
- `modeling/train.py` con `typer`: `python -m mibici.modeling.train --pregunta P1|P2|P3|todas` y regla `make train`.
- README completo: comandos de ingesta y experimento; el servidor de MLflow se levanta con `make mlflow`
  (`mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000`).
- Verificar desde un clon limpio: `make requirements`, `make data`, `make mlflow` (en otra terminal) y `make train`.
- Capturas para el PDF: lista de runs, detalle de un run, un artefacto y comparación de dos runs.
