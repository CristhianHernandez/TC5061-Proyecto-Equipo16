# Model Card — Demanda diaria por estación (P1)

Tarjeta del modelo en producción del proyecto MiBici del Equipo 16, con la estructura de Mitchell et
al. (2019). Resume qué hace el modelo, con qué datos se entrenó, cómo se evaluó, dónde falla y cómo
se opera. Las cifras salen de la evidencia del repositorio; cada sección indica su fuente.

| Campo | Valor |
|---|---|
| Modelo registrado | `mibici-equipo16-p1-demanda` |
| Versión en producción | **3**, alias `champion` |
| Versión de rollback | 2, alias `challenger` |
| URI que carga la API | `models:/mibici-equipo16-p1-demanda@champion` |
| Algoritmo | `HistGradientBoostingRegressor` con pérdida Poisson (scikit-learn 1.9.1) |
| Serialización | MLflow `sklearn` con formato `skops`; solo se confían tipos de scikit-learn, numpy y `mibici` |
| Equipo responsable | Equipo 16, TC5061 MLOps (ver [Integrantes y roles](../README.md#integrantes-y-roles)) |
| Fecha de la decisión | 2026-11-13, A/B offline del Entregable 8 ([`docs/reportes/actividad10/`](../docs/reportes/actividad10/README.md)) |
| Fecha de la promoción | 2026-11-24, en el registry del equipo |

## 1. Detalles del modelo

- **Qué predice:** el número de viajes que se iniciarán en una estación de MiBici en un día, con
  información disponible hasta el día anterior (granularidad estación × día, horizonte D+1).
- **Receta:** `learning_rate` 0.05, `max_iter` 100, `max_leaf_nodes` 63, semilla 16. Los
  hiperparámetros se eligieron en la validación de abr–jun 2026 y ninguna versión posterior los
  reajustó.
- **Por qué este algoritmo:** la demanda es un conteo no negativo y asimétrico, y la pérdida Poisson
  lo modela directamente. Frente a Random Forest, HGB quedó a 0.89 % en `val_mae`, pero fue más
  estable mes a mes (desviación del MAE 0.742 contra 0.751) y mucho más barato de entrenar (1.4 s
  contra 275.8 s). La regla de selección está en el README, sección
  [Modelo en producción de cada pregunta](../README.md#modelo-en-producción-de-cada-pregunta).
- **Versión 3 frente a versión 2:** misma receta, variables, firma y semilla. Solo cambia la ventana
  de entrenamiento, que se amplía con abr–sep 2026 e incluye el historial de las estaciones nuevas.

## 2. Uso previsto

| | |
|---|---|
| Usuario principal | Operador del sistema MiBici: planeación de flota y del rebalanceo diario |
| Decisión que apoya | Cuántas bicicletas conviene tener disponibles en cada estación al día siguiente |
| Forma de consumo | API REST (`POST /predict`) o, de forma más barata, un lote diario ([FinOps](../docs/evidencias/operacion/finops/resumen.md)) |
| Fuera de alcance | Predicción por hora o de minutos; riesgo de quedarse sin bicicletas o anclajes (eso es P2); estaciones fuera del área metropolitana de Guadalajara; cualquier decisión sobre personas |

El modelo estima demanda agregada por estación. No identifica, perfila ni evalúa a usuarios.

## 3. Variables de entrada

Contrato completo en [`docs/serving/contrato_p1.md`](../docs/serving/contrato_p1.md). Las variables
se calculan en [`mibici/features.py`](../mibici/features.py) sin información posterior al día que se
predice.

| Variable | Significado |
|---|---|
| `dia_semana`, `mes`, `dia_anio`, `es_fin_semana` | Calendario del día a predecir |
| `es_festivo` | 1 si es festivo oficial en México (paquete `holidays`) |
| `demanda_lag_1` | Viajes del día calendario anterior; nulo si falta ese día |
| `demanda_lag_7` | Viajes de siete días antes |
| `demanda_media_7` | Media de los siete días anteriores, sin incluir el día actual |
| `capacidad`, `latitud`, `longitud` | Datos de la estación del catálogo de MiBici |

Los tres rezagos son opcionales: faltan al inicio del historial de una estación y el modelo maneja
los nulos de forma nativa. Las demás variables son obligatorias, y la API responde 422 si la entrada
no cumple la firma.

## 4. Datos

**Fuente:** [MiBici, datos abiertos](https://mibici.net/es/datos-abiertos/): viajes mensuales y
catálogo de estaciones. La ingesta ([`mibici/dataset.py`](../mibici/dataset.py)) descarga los
archivos, normaliza codificación, columnas y fechas, y guarda Parquet. El manifiesto
[`references/manifiesto_datos.json`](manifiesto_datos.json) registra el SHA-256 de cada archivo;
la versión de datos `4d8c2a1292fc` queda en cada corrida de MLflow.

| Conjunto | Fechas | Filas (estación × día) | Uso |
|---|---|---|---|
| Entrenamiento de la versión 3 | 2024-10-01 a 2026-09-30 | 278,214 (470 estaciones) | Ajuste del modelo |
| Entrenamiento de la versión 2 | 2024-10-01 a 2026-03-31 | 201,267 (371 estaciones) | Ajuste del champion anterior |
| Evaluación común del A/B | 2026-10-01 a 2026-10-31 | 14,547 (470 estaciones) | Comparación de versiones; nadie entrenó ni eligió nada con este mes |

Las ventanas no se traslapan; lo verifican `validar_periodos` y
[`tests/test_drift_datasets.py`](../tests/test_drift_datasets.py). El contrato de los conjuntos
está en [`docs/monitoring/contrato_datasets.md`](../docs/monitoring/contrato_datasets.md).

## 5. Métricas y evaluación

**Métricas:** el MAE es la principal (viajes por estación y día, fácil de explicar al operador). Se
reportan también RMSE (penaliza errores grandes en estaciones de alta demanda), desviación Poisson
(coherente con la pérdida), sesgo (media de ŷ − y) y MAE relativo a la demanda media.

**Desarrollo de la receta** (versión 2, partición temporal train oct-2024–mar-2026, validación
abr–jun 2026, prueba jul–sep 2026):

| Conjunto | MAE | RMSE | Desv. Poisson |
|---|---|---|---|
| Validación (abr–jun 2026) | 6.137 | 11.509 | 3.048 |
| Prueba (jul–sep 2026) | 5.795 | 11.025 | 2.333 |

En prueba, la receta reduce 21.2 % el MAE del baseline naive estacional.

**A/B offline sobre octubre de 2026** (mismas 14,547 filas para ambas versiones):

| Versión | MAE ↓ | RMSE ↓ | Desv. Poisson ↓ | Sesgo | MAE relativo |
|---|---|---|---|---|---|
| 2 (anterior) | 8.122 | 17.500 | 3.227 | −1.58 | 18.6 % |
| **3 (producción)** | **8.018** | **16.852** | **3.128** | **−1.33** | **18.4 %** |

- ΔMAE = −0.104 (−1.27 %), IC 95 % por bootstrap de bloques de días [−0.155, −0.054].
- La versión 3 gana 23 de 31 días (Wilcoxon p = 0.0004) y ningún segmento empeora más de 2 %.
- Antes de tocar octubre se verificó la receta reentrenada: con referencia + abr–jun, el MAE de
  jul–sep baja de 5.795 a 5.761.

Fuente: [`docs/reportes/actividad10/resultados_ab.json`](../docs/reportes/actividad10/resultados_ab.json).

## 6. Análisis por segmento (octubre de 2026)

| Segmento | N | Demanda media | MAE v2 | MAE v3 | Δ MAE |
|---|---|---|---|---|---|
| Estaciones conocidas | 11,470 | 50.6 | 9.230 | 9.101 | −1.40 % |
| Estaciones nuevas | 3,077 | 17.7 | 3.988 | 3.981 | −0.17 % |
| Laborable | 10,323 | 48.2 | 7.744 | 7.656 | −1.14 % |
| Fin de semana | 4,224 | 32.6 | 9.044 | 8.903 | −1.56 % |
| Conocidas, demanda alta | 5,766 | 79.8 | 13.599 | 13.347 | −1.86 % |
| Conocidas, demanda baja | 5,704 | 21.2 | 4.814 | 4.809 | −0.11 % |

El error absoluto crece con la demanda: las estaciones de alta demanda concentran los errores más
grandes y también la mejora de la versión 3.

## 7. Limitaciones conocidas

- **Subestima los cambios de nivel.** Ambas versiones subestiman octubre de 2026 (sesgo −1.33 en la
  versión 3) porque la demanda de las estaciones conocidas subió 37.9 % respecto a octubres previos y
  ese aumento no está en ningún dato de entrenamiento permitido.
- **Días atípicos.** Los días de demanda atípica (3, 10 y 12 de octubre de 2026) tienen el mismo
  error alto en las dos versiones: el calendario y los rezagos no anticipan eventos, clima ni cierres.
- **Estaciones nuevas.** Sin historial, los rezagos son nulos y la predicción depende de calendario,
  capacidad y ubicación. En octubre, el 21.2 % de las filas eran de estaciones que la versión 2 no
  vio.
- **No hay inventario.** El modelo predice viajes iniciados, no bicicletas disponibles; no sustituye
  un conteo de flota.
- **A/B offline.** La comparación fue pareada sobre datos históricos, no un experimento con tráfico
  aleatorizado ni con efecto de las predicciones sobre la operación.

## 8. Consideraciones éticas y datos personales

- **Minimización de datos:** la ingesta descarta `Usuario_Id`, `Genero` y `Año_de_nacimiento`. Los
  Parquet solo conservan identificador del viaje, horas y estaciones de origen y destino, y el modelo
  usa conteos agregados por estación y día.
- **Sin datos personales en MLflow:** parámetros, tags y artefactos no contienen información de
  usuarios ni credenciales (README, sección
  [Reproducibilidad y buenas prácticas del equipo](../README.md#reproducibilidad-y-buenas-prácticas-del-equipo)).
- **Condiciones de uso:** el portal de MiBici publica los datos como abiertos, pero no declara una
  licencia explícita; el uso es académico y se cita la fuente.
- **Riesgo de uso:** una subestimación sistemática en zonas de alta demanda podría dejar estaciones
  sin bicicletas. Por eso se monitorea el sesgo por segmento y la decisión final de rebalanceo es del
  operador.

## 9. Linaje en el Model Registry

Usamos aliases y no etapas (*stages*), que están obsoletas desde MLflow 2.9. `champion` es
producción; `challenger` conserva la versión anterior para un rollback inmediato.

| Versión | Alias | Corrida | Entrenamiento | Estado |
|---|---|---|---|---|
| 1 | — | `p1-hgb` | oct-2024 a mar-2026 | Primera candidata, reemplazada |
| 2 | `challenger` | `p1-hgb-featuresv1-learning_rate0.05-max_iter100-max_leaf_nodes63` (`47933c5b…`), commit `c30cd49`, datos `4d8c2a1292fc` | oct-2024 a mar-2026 | Rollback |
| 3 | `champion` | `p1-hgb-challenger-reentrenado-abr-sep-2026` (`c6349c2b…`), commit `544634d`, datos `4d8c2a1292fc` | oct-2024 a sep-2026 | **Producción** |

Cada versión guarda en tags su linaje: experimento, `run_id`, `git_commit`, `version_datos`,
ventana de entrenamiento y la versión del champion contra la que se comparó. La versión 3 tiene
además la tag `promocion`, que apunta a la corrida del A/B que la justificó
(`24fa3545f8664d9699e0d981605234a3`), y la versión 2 la tag `rol = rollback`.

El A/B del Entregable 8 se corrió en un registry reconstruido en otra máquina (corridas
`86cc548d…` y `658257aa…`). En el registry del equipo, la versión 3 se reentrenó con el mismo
comando y reprodujo exactamente las cifras: mismas huellas de datos (`827ffa8b5727`,
`4f52f6556fe4`, `b511db2054b2`), MAE de octubre 8.0180 y la misma decisión.

**Cómo se llegó a la versión 3:**

```bash
make monitoring-challenger    # registra la versión con alias challenger; champion no se mueve
make monitoring-ab            # A/B en octubre de 2026 → "promover_challenger"
# promoción: champion → versión 3 y challenger → versión 2 (rollback)
```

## 10. Despliegue y operación

| Aspecto | Valor | Fuente |
|---|---|---|
| Servicio | FastAPI: `POST /predict`, `GET /health`, `GET /docs` | [`docs/serving/contrato_p1.md`](../docs/serving/contrato_p1.md) |
| Orquestación | Docker Compose: `mlflow`, `model-init` y `api` con una sola imagen | [`docs/operacion/contrato_arranque.md`](../docs/operacion/contrato_arranque.md) |
| CI/CD | GitHub Actions: Ruff, pytest y prueba de humo del stack en cada push y PR | [`.github/workflows/operacion.yml`](../.github/workflows/operacion.yml) |
| Latencia | p95 de 28 ms con 5 usuarios y 37 ms con 20 (1 worker, 2 CPU) | [`docs/evidencias/operacion/carga/`](../docs/evidencias/operacion/carga/) |
| Capacidad | 114.9 pet/s con p95 de 140 ms fijando `OMP_NUM_THREADS=1` | [Locust](../docs/evidencias/operacion/carga/) |
| Costo | $0.00208 por 1,000 inferencias (5 usuarios) y $0.00052 (20 usuarios) en t4g.medium | [FinOps](../docs/evidencias/operacion/finops/resumen.md) |

Las cifras de carga y costo se midieron con la versión 2. La versión 3 tiene la misma receta y
firma, así que su costo de inferencia es equivalente. El stack de Compose sirve la versión 3:
en su registry propio aparece como versión 1 con `origen = bundle` y `version_origen = 3`.

## 11. Monitoreo y reentrenamiento

- **Drift** con Evidently ([`docs/evidencias/drift/`](../docs/evidencias/drift/README.md)): en
  octubre de 2026, 7 de 11 variables presentan drift (Wasserstein normalizada en numéricas y
  Jensen-Shannon en categóricas, umbral 0.1) y el objetivo tiene drift de 0.298.
- **Política de Continuous Training** (`politica_continuous_training`): mensual, al publicarse cada
  mes de MiBici. Se reentrena un candidato si hay drift del objetivo ≥ 0.1, alguna variable de
  historial de demanda con drift o ≥ 10 % de filas de estaciones no vistas. Los problemas de calidad
  bloquean el entrenamiento.
- **Degradación:** se mide con el MAE relativo a la demanda, no con el absoluto. En octubre, el MAE
  absoluto subió 40 % frente a jul–sep, pero el relativo bajó de 19.3 % a 18.6 %: la demanda creció,
  el modelo no se degradó.
- **Promoción gobernada:** solo la decisión `promover_challenger` del A/B habilita mover el alias,
  con aprobación humana. La versión anterior queda como `challenger` para rollback.

## 12. Cómo reproducir

```bash
make data                       # ingesta versionada
make train QUESTION=p1 MODEL=hgb
make register QUESTION=p1       # versión con alias champion y linaje
make monitoring-data            # ventanas referencia, reciente y actual
make monitoring-drift           # reporte de Evidently
make monitoring-challenger      # versión reentrenada con alias challenger
make monitoring-ab              # A/B y decisión
make export-model && make up    # stack de Compose sirviendo el champion
```

## Referencias

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji,
I. D., & Gebru, T. (2019). Model cards for model reporting. En *Proceedings of the Conference on
Fairness, Accountability, and Transparency* (pp. 220–229). ACM.
https://doi.org/10.1145/3287560.3287596

MiBici. (s.f.). *Datos abiertos* [Conjunto de datos]. Sistema de Bicicleta Pública del Área
Metropolitana de Guadalajara. https://mibici.net/es/datos-abiertos/
