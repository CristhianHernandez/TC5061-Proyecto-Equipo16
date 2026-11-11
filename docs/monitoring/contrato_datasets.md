# Conjuntos del monitoreo de drift y del A/B testing (Entregable 8)

Contrato entre la preparación de datos (`mibici/monitoring/datasets.py`, PR de CH1) y el reporte de
Evidently (LH3), el challenger (HH2) y la comparación A/B (DH4). Las cifras son las de
[`references/manifiesto_monitoreo.json`](../../references/manifiesto_monitoreo.json), generado el
2026-11-10 con `make monitoring-data`.

## Modelo monitoreado

| Campo | Valor |
|---|---|
| Modelo | `mibici-equipo16-p1-demanda` (P1, regresión de la demanda diaria por estación) |
| Champion (A) | Alias `champion` → **versión 2**, corrida `47933c5b0df749edaeb159289053f70e` |
| Versión de datos de entrenamiento | `4d8c2a1292fc` (`references/manifiesto_datos.json`) |
| Granularidad | Una fila por estación y día |
| Métrica principal | MAE (viajes por estación y día); RMSE como complemento |

Se eligió P1 porque es el modelo que sirve la API de la semana 8 y porque su objetivo, la demanda
observada, está disponible en el periodo nuevo para medir el desempeño de A y B. La versión se fija
por número: el alias es una referencia móvil y no se mueve antes de la decisión del A/B.

## Ventanas

| Ventana | Fechas | Días | Filas | Estaciones | Demanda media | Catálogo | Uso |
|---|---|---|---|---|---|---|---|
| `referencia` | 2024-10-01 a 2026-03-31 | 547 | 201,267 | 371 | 33.47 | `nomenclatura_2026_09` | Datos con los que se entrenó el champion; base del drift |
| `reciente` | 2026-04-01 a 2026-09-30 | 183 | 76,947 | 470 | 29.75 | `nomenclatura_2026_09` | Validación y prueba originales; el challenger puede entrenar con ellas |
| `actual` | 2026-10-01 a 2026-10-31 | 31 | 14,547 | 470 | 43.67 | `nomenclatura_2026_10` | "Producción": drift contra la referencia y evaluación común del A/B |

Motivos:

- **Referencia = train del champion.** El drift debe medir cuánto se alejan los datos de lo que el
  modelo realmente vio. El código comprueba que tiene las mismas filas que el train de P1.
- **Actual = octubre de 2026.** Es el primer mes publicado por MiBici (el 2 de noviembre de 2026)
  después de todos los datos del proyecto, así que ningún modelo, búsqueda ni decisión lo ha usado.
  Representa lo que la API recibiría en producción. Un mes completo da 31 días por estación y
  cubre todos los días de la semana.
- **Reciente = abril a septiembre de 2026.** Ya tiene etiquetas y es anterior a la evaluación, así que
  sirve para reentrenar el challenger con datos más cercanos a producción sin tocar octubre.
- **Catálogo vigente en la ventana actual.** En producción las variables de estación se toman del
  catálogo publicado en ese momento. Respecto al de septiembre, el de octubre da de baja 14
  estaciones que ya eran `ex-` (inactivas), cambia la capacidad de 11 y no agrega ninguna; las
  coordenadas solo cambian por redondeo (máximo 20 m). El detalle está en `cambios_catalogo`.

## Reglas sin fuga de información

1. Las ventanas son consecutivas y no se traslapan: `referencia < reciente < actual`
   (`validar_periodos` lo verifica y falla en caso contrario).
2. El challenger solo entrena y elige hiperparámetros con fechas anteriores al 2026-10-01:
   `datos_entrenamiento_challenger()` devuelve referencia + reciente y llama a `validar_sin_fuga`.
3. Octubre es la evaluación común: A y B predicen **exactamente las mismas 14,547 filas**.
4. Las variables de octubre se calculan con el mismo pipeline de P1 sobre el historial completo;
   los rezagos usan solo días previos (el 1 de octubre toma la demanda del 30 de septiembre, que
   ya era conocida). Hay una prueba que altera un día y comprueba que los días anteriores no
   cambian.
5. La comparación A/B fija versiones numéricas del registry. El alias `champion` no se mueve hasta
   que el análisis recomiende una decisión.

## Columnas

Las tres ventanas tienen las mismas columnas y tipos:

| Grupo | Columnas |
|---|---|
| Identificadores (no son variables del modelo) | `estacion_id`, `fecha` |
| Variables categóricas para Evidently | `dia_semana`, `mes`, `es_fin_semana`, `es_festivo` |
| Variables numéricas para Evidently | `dia_anio`, `demanda_lag_1`, `demanda_lag_7`, `demanda_media_7`, `capacidad`, `latitud`, `longitud` |
| Objetivo | `demanda` (viajes que inician en la estación ese día) |

Notas para la interpretación del drift (LH3):

- `mes` y `dia_anio` cambian por construcción, porque la ventana actual es un solo mes. Es
  estacionalidad del calendario, no un problema de datos; conviene reportarlo así o excluirlas
  del resumen de drift de entradas.
- `capacidad`, `latitud` y `longitud` cambian porque la red creció: la referencia tiene 371
  estaciones con viajes y la ventana actual 470. **99 estaciones empezaron a operar entre mayo y
  agosto de 2026** (12 en mayo, 41 en junio, 31 en julio y 15 en agosto), así que el champion nunca
  vio sus ubicaciones ni su historial. Es la causa más probable del drift en las variables de
  estación y un argumento directo para reentrenar.
- La demanda de octubre es mayor (43.7 viajes por estación y día) que en la referencia (33.5) y que
  en el periodo reciente (29.8). Eso desplaza también los rezagos (`demanda_lag_*`,
  `demanda_media_7`).
- Los nulos en los rezagos solo aparecen en la referencia (359 en `demanda_lag_1` y 2,519 en
  `demanda_lag_7` y `demanda_media_7`), al inicio del historial de cada estación. El modelo los
  admite.

## Calidad del mes actual

| Indicador | Octubre de 2026 |
|---|---|
| Viajes publicados | 635,532 |
| Filas duplicadas | 0 |
| Viajes con estación fuera del catálogo vigente | 1 |
| Viajes con duración ≤ 0 o > 24 h | 193 |
| Viajes que usa P1 después de la limpieza | 635,338 |

## Uso

```bash
make mlflow MLFLOW_PORT=5001                    # para anotar la versión del champion
make monitoring-data MLFLOW_PORT=5001           # o: python -m mibici.monitoring.datasets
python -m mibici.monitoring.datasets --sin-registry   # sin servidor de MLflow
```

Requiere los datos de entrenamiento procesados (`make data`). Descarga octubre y su catálogo a
`data/raw/`, los normaliza en `data/processed/monitoreo/` (fuera de Git y fuera de
`data/processed/viajes/`, porque P2 y P3 leen todos los archivos de esa carpeta) y guarda
`referencia.parquet`, `reciente.parquet` y `actual.parquet`. El manifiesto registra las fechas, el
SHA-256 de los archivos descargados, la versión del champion y una huella de cada conjunto. Volver
a ejecutar el comando produce las mismas huellas.

```python
from mibici.monitoring.datasets import (
    CATEGORICAS, NUMERICAS, OBJETIVO, VARIABLES,
    cargar_conjuntos, datos_entrenamiento_challenger,
)

conjuntos = cargar_conjuntos()            # {"referencia", "reciente", "actual"} -> DataFrame
conjuntos["referencia"][VARIABLES]        # LH3: entrada de Evidently (referencia vs actual)
datos_entrenamiento_challenger(conjuntos) # HH2: referencia + reciente, verificado sin fuga
conjuntos["actual"]                       # DH4: evaluación común de A y B
```

Comprobación de consumo (no sustituye el A/B de DH4): el champion versión 2 carga desde
`model_bundle/`, predice las 14,547 filas de octubre con el contrato de la API y su MAE es 8.12,
frente a 5.95 en la ventana reciente. Es una primera señal de que el cambio de octubre afecta el
desempeño.

## Limitación detectada

El portal de MiBici reemplazó `nomenclatura_2026_09.csv` por `nomenclatura_2026_10.csv`; el enlace
de `URL_NOMENCLATURA` en `mibici/config.py` responde 404 desde noviembre de 2026. `make data`
sigue funcionando en las máquinas que ya tienen el archivo en `data/raw/`, pero en un clon limpio
la ingesta del catálogo de entrenamiento falla. Corregirlo cambia el catálogo y la versión de los
datos del champion, así que se deja documentado para decidirlo en equipo.
