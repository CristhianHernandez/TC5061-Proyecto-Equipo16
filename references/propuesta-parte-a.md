# Propuesta — Parte A (borrador para discusión del equipo)

> **Estado:** borrador. La sección de datos se basa en lo verificado en el portal el 2026-10-14;
> las decisiones de P1, P2 y P3 son una **propuesta para discutir** y se ajustarán con la exploración.
> Las cifras marcadas como *estimadas* se reemplazarán por las exactas que produzca la ingesta.

## 1. Datos

### 1.1 Fuente

- Portal de datos abiertos de MiBici: <https://mibici.net/es/datos-abiertos/>
- Viajes mensuales: `https://mibici.net/site/assets/files/<id>/datos_abiertos_AAAA_MM.csv`
  (el `<id>` cambia por mes; se listarán en `mibici/config.py`).
- Estaciones: `https://mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv`
- Opera BKT Bici Pública S.A. de C.V.; administra la Agencia Metropolitana de Servicios de
  Infraestructura para la Movilidad del AMG (AMIM).

### 1.2 Rango de fechas: octubre 2024 – septiembre 2026 (24 meses)

El portal publica desde diciembre de 2014 hasta septiembre de 2026. Elegimos los 24 meses más recientes porque:

1. **Evita discontinuidades:** abril y mayo de 2023 no tienen archivo descargable, y 2020–2021 reflejan
   la pandemia, con patrones de movilidad que ya no representan la operación actual.
2. **Es consistente con las estaciones vigentes:** la nomenclatura disponible es de septiembre de 2026; con
   datos recientes, los viajes corresponden a la red actual.
3. **Cubre dos ciclos anuales completos**, necesarios para capturar la estacionalidad (vacaciones, lluvias,
   periodo escolar).
4. **Tamaño manejable** para ejecutar la ingesta y los experimentos en equipos personales.

### 1.3 Tamaño

| Concepto | Valor |
|---|---|
| Meses | 24 |
| Viajes | ~10 millones (*estimado* por el rango de `Viaje_Id`: 35.8 M → 45.7 M) |
| Volumen en CSV | ~770 MB (27–43 MB por mes) |
| Estaciones en la nomenclatura | 484 (14 marcadas como `ex-`, dadas de baja; 4 de ellas con capacidad 0) |
| Anclajes totales | 8,171 (capacidad mediana 15, máxima 39) |

### 1.4 Variables

**Viajes** (las 8 columnas son iguales en los 24 meses):

| Columna | Descripción | Uso |
|---|---|---|
| `Viaje_Id` | Identificador del viaje | Control de duplicados |
| `Usuario_Id` | Identificador del usuario | **Se descarta** (dato personal, no se requiere) |
| `Genero` | Género del usuario | **Se descarta** (dato personal, no se requiere) |
| `Año_de_nacimiento` | Año de nacimiento | **Se descarta** (dato personal, no se requiere) |
| `Inicio_del_viaje` | Fecha y hora de inicio | Objetivos de P1/P2, perfiles de P3 |
| `Fin_del_viaje` | Fecha y hora de fin | Llegadas (P2, P3) y duración |
| `Origen_Id` | Estación de origen | Salidas por estación |
| `Destino_Id` | Estación de destino | Llegadas por estación |

**Estaciones:** `id`, `name`, `latitude`, `longitude`, `dpcapacity` (capacidad en anclajes).

**Diferencias de formato detectadas** (las normaliza `mibici/dataset.py`):

- Codificación **latin-1** en 2024-10, 2024-11, 2024-12, 2025-01, 2025-02, 2025-07 y 2026-01; UTF-8 en el resto.
- **2024-10** usa otro formato: sin comillas, fin de línea CRLF y fechas `01/10/2024 0:00` (día/mes/año, sin
  segundos). El resto usa `AAAA-MM-DD HH:MM:SS`.

### 1.5 Condiciones de uso y datos personales

- El portal invita a "explorar, experimentar y compartir" los datos, pero **no declara una licencia explícita**;
  los Términos y condiciones (<https://mibici.net/es/termino-y-condiciones/>) no regulan su reutilización.
  Se citará la fuente y se usará con fines académicos.
- El Aviso de privacidad (<https://mibici.net/es/aviso-de-privacidad/>), a cargo de AMIM, no menciona la
  publicación de viajes como datos abiertos.
- **Tratamiento:** aplicamos minimización de datos. `Usuario_Id`, `Genero` y `Año_de_nacimiento` se eliminan
  en la ingesta, antes de guardar en Parquet, porque ninguna de las tres preguntas los necesita. Ningún dato
  personal se registra en params, tags ni artefactos de MLflow; solo se publican agregados por estación.

### 1.6 Partición sin fuga del futuro

| Conjunto | Periodo | Uso |
|---|---|---|
| Entrenamiento | oct-2024 – mar-2026 (18 meses) | Ajuste de modelos |
| Validación | abr-2026 – jun-2026 (3 meses) | Selección de modelo e hiperparámetros |
| Prueba | jul-2026 – sep-2026 (3 meses) | Evaluación final, una sola vez |

- La partición es **temporal**, nunca aleatoria: no se entrena con información posterior a lo que se evalúa.
- Las variables de rezago y promedios móviles usan solo información disponible **antes** del momento de predicción.
- Las estadísticas de normalización y codificación se calculan únicamente con entrenamiento.
- Las fechas de corte viven solo en `mibici/config.py`.

## 2. Preguntas (propuesta)

### P1 — Regresión: demanda de viajes por estación

| Aspecto | Propuesta |
|---|---|
| Qué se predice | Número de viajes que **inician** en cada estación |
| Granularidad | Estación × día |
| Horizonte | Día siguiente (D+1) |
| Usuario | Operador del sistema (planeación de flota y personal) |
| Relevancia | Anticipar la demanda permite asignar bicicletas y personal por estación |
| Objetivo | Conteo de viajes con `Origen_Id = s` e `Inicio_del_viaje` en el día d |
| Variables | Día de la semana, mes, festivos, rezagos (d-1, d-7), promedio móvil de 7 y 28 días, capacidad, ubicación |
| Baseline | *Naive estacional*: mismo día de la semana anterior; promedio histórico por estación × día de la semana |
| Candidato | `HistGradientBoostingRegressor` (scikit-learn), con pérdida Poisson por tratarse de conteos |
| Métricas | **MAE** (en viajes, interpretable para el operador), RMSE (penaliza errores grandes), desviación Poisson media; MAE por rebanadas (estaciones de alta y baja demanda). No usamos MAPE por los días con cero viajes |

### P2 — Clasificación: riesgo de quedarse sin bicicletas o sin anclajes

No hay inventario por estación, así que **aproximamos el riesgo con el flujo neto y la capacidad**:

- Flujo neto de la estación s en la franja t: `salidas − llegadas`.
- Suponiendo que la estación inicia la franja a media ocupación (≈ 50 % de su capacidad), un flujo neto de
  salida mayor o igual a `α · capacidad` la vaciaría, y uno de llegada mayor o igual a `α · capacidad` la llenaría.
- Proponemos `α = 0.5` (consistente con el supuesto de media ocupación) y analizar la sensibilidad con 0.4 y 0.6.

| Aspecto | Propuesta |
|---|---|
| Qué se predice | Clase de riesgo de la estación en la siguiente franja: `sin_bicicletas`, `sin_anclajes`, `normal` |
| Granularidad | Estación × franja de 3 horas |
| Horizonte | Siguiente franja (3 h), con información hasta el inicio de la franja |
| Usuario | Equipo de rebalanceo (rutas de camiones) |
| Relevancia | Priorizar qué estaciones atender antes de que fallen |
| Objetivo | Construido con la regla anterior sobre salidas y llegadas observadas en la franja |
| Variables | Hora, día de la semana, flujo neto rezagado (franja previa, mismo bloque de la semana anterior), capacidad, ubicación |
| Baseline | `DummyClassifier` (clase más frecuente); regla de persistencia (misma clase que el mismo bloque de la semana anterior) |
| Candidato | `LogisticRegression` con `class_weight="balanced"` y `HistGradientBoostingClassifier` |
| Métricas | Clases muy desbalanceadas: **F1 macro**, *recall* por clase de riesgo (fallar una estación en riesgo es lo más costoso), *average precision* (PR-AUC) y *balanced accuracy*. La exactitud simple no se usa como métrica principal |

**Limitaciones a reportar:** el supuesto de ocupación inicial, que no se observa el rebalanceo real del operador y
que la regla no considera la acumulación entre franjas.

### P3 — Clustering: patrones de uso de estaciones

| Aspecto | Propuesta |
|---|---|
| Qué se agrupa | Estaciones activas según su patrón de uso |
| Granularidad | Una fila por estación, con perfiles calculados en el periodo de entrenamiento |
| Usuario | Planeación urbana y movilidad (ubicación de nuevas estaciones, infraestructura ciclista) |
| Relevancia | Identificar zonas residenciales (salen en la mañana), de empleo (llegan en la mañana) y de ocio |
| Variables | Perfil horario normalizado de salidas y llegadas (laboral y fin de semana), balance neto, volumen total, duración media del viaje |
| Baseline | Agrupación por municipio (prefijo GDL / ZPN / TLQ de la nomenclatura) como regla sin aprendizaje |
| Candidato | `KMeans` (k elegido por silueta y codo) y `AgglomerativeClustering` (Ward) como alternativa |
| Métricas | Silueta, Davies-Bouldin, Calinski-Harabasz; estabilidad (ARI entre semillas y contra el periodo de validación) e interpretabilidad de los perfiles |
| Artefacto | Mapa de estaciones coloreado por cluster y perfiles horarios promedio por cluster |

## 3. Pendientes

- [ ] Validar con el equipo granularidad y horizonte de P1 y P2.
- [ ] Confirmar `α` y la definición de franjas de P2 tras la exploración.
- [ ] Reemplazar cifras estimadas con las de la ingesta (viajes exactos, estaciones con viajes).
- [ ] ML Canvas, pipeline por pregunta y diagrama de arquitectura.
- [ ] Referencias en formato APA 7.
