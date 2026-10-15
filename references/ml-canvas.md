# ML Canvas — MiBici (Equipo 16)

> **Estado:** borrador para discusión, alineado con [`propuesta-parte-a.md`](propuesta-parte-a.md).
> Estructura basada en el Machine Learning Canvas de Louis Dorard (v1.1, 10 bloques).
> El orden sigue el del canvas: propósito (centro), lado de predicción (izquierda) y lado de aprendizaje (derecha).

## Propuesta de valor (centro)

**¿Quién usa el sistema y qué gana?**

- **Operador de MiBici (P1):** anticipar cuántos viajes iniciarán por estación al día siguiente para dimensionar
  flota y personal.
- **Equipo de rebalanceo (P2):** saber qué estaciones tienen riesgo de quedarse sin bicicletas o sin anclajes en
  las próximas 3 horas, para priorizar rutas antes de que el usuario encuentre la estación vacía o llena.
- **Planeación urbana (P3):** entender qué tipos de estaciones existen (residenciales, de empleo, de ocio) y dónde
  están, para decidir dónde ampliar la red o la infraestructura ciclista.

## Lado de predicción

### Tarea de predicción

| | P1 | P2 | P3 |
|---|---|---|---|
| Tipo | Regresión | Clasificación multiclase | Clustering (no supervisado) |
| Entidad | Estación × día | Estación × franja de 3 h | Estación |
| Salida | Número de viajes que inician | `sin_bicicletas` / `sin_anclajes` / `normal` | Grupo de patrón de uso |
| Horizonte | Día siguiente (D+1) | Siguiente franja de 3 h | Periodo de entrenamiento (perfil estable) |

### Decisiones

- **P1:** cuántas bicicletas y personal asignar por zona cada día.
- **P2:** qué estaciones visitar primero en cada ruta de rebalanceo y si se retiran o se llevan bicicletas.
- **P3:** dónde abrir, reubicar o ampliar estaciones; qué zonas requieren infraestructura según su patrón.

### Hacer predicciones

- **P1:** en lote, una vez al día (al cierre), para las ~470 estaciones activas.
- **P2:** en lote, al inicio de cada franja de 3 h, con los viajes registrados hasta ese momento.
- **P3:** bajo demanda o por trimestre; los clusters cambian lentamente.
- En esta entrega las predicciones se generan en lote desde el código del repositorio; no hay servicio en línea.

### Simulación de impacto

- **Antes de usar un modelo**, se compara contra su baseline en el conjunto de prueba (jul–sep 2026), que nunca
  se usa para ajustar.
- **P1:** MAE en viajes por estación y día; el modelo se justifica si reduce el MAE frente al naive estacional.
- **P2:** el costo de no detectar una estación en riesgo (falso negativo) es mayor que el de una visita innecesaria
  (falso positivo); por eso priorizamos *recall* de las clases de riesgo, F1 macro y PR-AUC.
- **P3:** silueta y estabilidad mayores que el baseline por municipio y perfiles interpretables por el usuario.

### Monitoreo

- Seguimiento de métricas por mes en datos nuevos (los CSV mensuales del portal) y comparación contra el baseline.
- Detección de cambios en la distribución: volumen de viajes, estaciones nuevas o dadas de baja (`ex-`) en la
  nomenclatura, cambios de formato o codificación en los CSV.
- Re-ejecutar los clusters de P3 y comparar con ARI para detectar cambios de patrón.

## Lado de aprendizaje

### Fuentes de datos

- Viajes mensuales de datos abiertos de MiBici (oct-2024 – sep-2026): <https://mibici.net/es/datos-abiertos/>
- Nomenclatura de estaciones (`nomenclatura_2026_09.csv`): ubicación y capacidad.
- Calendario de días festivos de México (variable derivada, sin fuente externa de datos personales).

### Recolección de datos

- Ingesta automatizada (`mibici/dataset.py`): descarga los meses configurados en `mibici/config.py`, normaliza la
  codificación (UTF-8 / latin-1) y el formato de fechas, elimina datos personales y guarda en Parquet.
- Versionado de datos: manifiesto con número de registros y hash por archivo; la versión se registra en MLflow.
- Las etiquetas de P1 y P2 se construyen a partir de los viajes (no requieren anotación manual).

### Variables (features)

- **Calendario:** hora, día de la semana, mes, festivo, fin de semana.
- **Historial (solo pasado):** rezagos y promedios móviles de salidas, llegadas y flujo neto por estación.
- **Estación:** capacidad, latitud, longitud, municipio.
- **P3:** perfiles horarios normalizados de salidas y llegadas (laboral y fin de semana), balance neto, volumen,
  duración media.
- Regla: ninguna variable usa información posterior al momento de predicción.

### Construcción de modelos

- Baselines primero: naive estacional (P1), clase más frecuente y persistencia (P2), agrupación por municipio (P3).
- Candidatos con scikit-learn: `HistGradientBoostingRegressor` (P1), `LogisticRegression` y
  `HistGradientBoostingClassifier` (P2), `KMeans` y `AgglomerativeClustering` (P3).
- Partición temporal: entrenamiento oct-2024 – mar-2026, validación abr–jun 2026, prueba jul–sep 2026.
- Cada run se registra en MLflow (`mibici-equipo16`) con parámetros, métricas con prefijo del conjunto,
  artefactos y tags de equipo, pregunta y commit de Git; semillas fijas desde `mibici/config.py`.
- Reentrenamiento propuesto: mensual, al publicarse un nuevo mes.

## Referencia

Dorard, L. (2018). *The Machine Learning Canvas*. <https://www.ownml.co/machine-learning-canvas>
