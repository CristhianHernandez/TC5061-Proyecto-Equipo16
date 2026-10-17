# ML Canvas — MiBici (Equipo 16)

> **Estado:** borrador para discusión, alineado con [`propuesta-parte-a.md`](propuesta-parte-a.md).
> Estructura: *The Machine Learning Canvas* v0.4 de Louis Dorard (10 bloques), según
> [From Data to AI with the Machine Learning Canvas (Part I)](https://medium.com/louis-dorard/from-data-to-ai-with-the-machine-learning-canvas-part-i-d171b867b047).
> El canvas se publica en [machinelearningcanvas.com](https://www.machinelearningcanvas.com) con licencia CC BY-SA 4.0.
> Cada bloque inicia con su pregunta guía original.

Distribución del canvas: a la izquierda, cómo se usan las predicciones (Decisions, ML task, Making
Predictions, Offline Evaluation); al centro, la propuesta de valor; a la derecha, cómo se aprende de
los datos (Data Sources, Collecting Data, Features, Building Models); abajo, Live Evaluation and Monitoring.

## Propuesta de valor (Value Propositions)

*¿Qué intentamos hacer por los usuarios finales del sistema predictivo? ¿A qué objetivos servimos?*

- **Operador de MiBici (P1):** anticipar cuántos viajes iniciarán por estación al día siguiente para dimensionar
  flota y personal.
- **Equipo de rebalanceo (P2):** saber qué estaciones tienen riesgo de quedarse sin bicicletas o sin anclajes en
  las próximas 3 horas, para priorizar rutas antes de que el usuario encuentre la estación vacía o llena.
- **Planeación urbana (P3):** entender qué tipos de estaciones existen (residenciales, de empleo, de ocio) y dónde
  están, para decidir dónde ampliar la red o la infraestructura ciclista.

## Lado izquierdo: uso de las predicciones

### Decisiones (Decisions)

*¿Cómo se usan las predicciones para tomar decisiones que den el valor propuesto al usuario final?*

- **P1:** cuántas bicicletas y personal asignar por zona cada día.
- **P2:** qué estaciones visitar primero en cada ruta de rebalanceo y si se retiran o se llevan bicicletas.
- **P3:** dónde abrir, reubicar o ampliar estaciones; qué zonas requieren infraestructura según su patrón.

### Tarea de ML (ML task)

*Entrada, salida a predecir y tipo de problema.*

| | P1 | P2 | P3 |
|---|---|---|---|
| Tipo de problema | Regresión | Clasificación multiclase | Clustering (no supervisado) |
| Entrada | Estación × día con calendario e historial | Estación × franja de 3 h con calendario y flujo neto previo | Perfil de uso de cada estación |
| Salida | Número de viajes que inician (D+1) | `sin_bicicletas` / `sin_anclajes` / `normal` en la siguiente franja | Grupo de patrón de uso |

### Hacer predicciones (Making Predictions)

*¿Cuándo hacemos predicciones sobre entradas nuevas? ¿Cuánto tiempo tenemos para calcular las variables
de una entrada nueva y predecir?*

- **P1:** en lote, una vez al día al cierre, para las ~470 estaciones activas; hay horas disponibles antes
  de que inicie el día siguiente.
- **P2:** en lote, al inicio de cada franja de 3 h, con los viajes registrados hasta ese momento; la
  predicción debe estar lista en minutos para que sirva a la ruta.
- **P3:** bajo demanda o por trimestre; los clusters cambian lentamente y no hay restricción de tiempo.
- En esta entrega las predicciones se generan en lote desde el código del repositorio; no hay servicio en línea.

### Evaluación fuera de línea (Offline Evaluation)

*Métodos y métricas para evaluar el sistema antes del despliegue.*

- **Método:** partición temporal (entrenamiento oct-2024 – mar-2026, validación abr–jun 2026, prueba
  jul–sep 2026). Modelos e hiperparámetros se eligen con validación; prueba se usa una sola vez.
- **P1:** MAE (principal), RMSE y desviación Poisson contra el baseline naive estacional.
- **P2:** F1 macro (principal), recall por clase de riesgo, average precision y balanced accuracy contra
  `DummyClassifier` y persistencia; no detectar una estación en riesgo cuesta más que una visita innecesaria.
- **P3:** silueta, Davies-Bouldin, Calinski-Harabasz y estabilidad (ARI) contra la agrupación por municipio.

## Lado derecho: aprendizaje a partir de los datos

### Fuentes de datos (Data Sources)

*¿Qué fuentes de datos crudos podemos usar (internas y externas)?*

- Viajes mensuales de datos abiertos de MiBici (oct-2024 – sep-2026): <https://mibici.net/es/datos-abiertos/>
- Nomenclatura de estaciones (`nomenclatura_2026_09.csv`): ubicación y capacidad.
- Calendario de días festivos de México (variable derivada, sin datos personales).

### Recolección de datos (Collecting Data)

*¿Cómo obtenemos datos nuevos para aprender (entradas y salidas)?*

- MiBici publica un CSV nuevo cada mes; la ingesta (`mibici/dataset.py`) lo descarga, normaliza la
  codificación (UTF-8 / latin-1) y el formato de fechas, elimina datos personales y guarda en Parquet.
- Las salidas (etiquetas) de P1 y P2 se construyen a partir de los mismos viajes; no requieren anotación manual.
- Versionado: manifiesto con número de registros y hash por archivo; la versión se registra en MLflow.

### Variables (Features)

*Representaciones de las entradas extraídas de las fuentes de datos crudos.*

- **Calendario:** hora, día de la semana, mes, festivo, fin de semana.
- **Historial (solo pasado):** rezagos y promedios móviles de salidas, llegadas y flujo neto por estación.
- **Estación:** capacidad, latitud, longitud, municipio.
- **P3:** perfiles horarios normalizados de salidas y llegadas (laboral y fin de semana), balance neto,
  volumen, duración media.
- Regla: ninguna variable usa información posterior al momento de predicción.

### Construcción de modelos (Building Models)

*¿Cuándo creamos o actualizamos modelos con datos de entrenamiento nuevos? ¿Cuánto tiempo tenemos para
calcular las variables de entrenamiento y crear un modelo?*

- Reentrenamiento propuesto: mensual, al publicarse un nuevo mes; hay días disponibles entre publicaciones.
- Baselines primero, luego candidatos con scikit-learn: `HistGradientBoostingRegressor` (P1),
  `LogisticRegression` y `HistGradientBoostingClassifier` (P2), `KMeans` y `AgglomerativeClustering` (P3).
- Cada entrenamiento se registra en MLflow (`mibici-equipo16`) con parámetros, métricas con prefijo del
  conjunto, artefactos y tags de equipo, pregunta y commit; semillas fijas desde `mibici/config.py`.

## Abajo: evaluación en vivo y monitoreo (Live Evaluation and Monitoring)

*Métodos y métricas para evaluar el sistema después del despliegue y cuantificar la creación de valor.*

- Con cada mes nuevo, comparar las predicciones contra lo ocurrido con las mismas métricas y contra el baseline.
- Valor: P1, error en viajes por estación frente al plan actual; P2, porcentaje de estaciones que quedaron
  vacías o llenas sin haber sido señaladas.
- Detectar cambios: volumen de viajes, estaciones nuevas o dadas de baja (`ex-`) en la nomenclatura,
  cambios de formato o codificación en los CSV.
- Re-ejecutar P3 y comparar con ARI para detectar cambios de patrón.

## Referencias

Dorard, L. (s.f.). *From data to AI with the Machine Learning Canvas (Part I)*. Medium.
https://medium.com/louis-dorard/from-data-to-ai-with-the-machine-learning-canvas-part-i-d171b867b047

Dorard, L. (s.f.). *The Machine Learning Canvas* (v0.4) [Plantilla]. https://www.machinelearningcanvas.com
