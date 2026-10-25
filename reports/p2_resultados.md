# P2 — Clasificación del riesgo operativo de estaciones MiBici

**Proyecto:** TC5061 — Equipo 16  
**Responsable:** HH2  
**Fecha:** 24 de octubre de 2026  
**Experimento MLflow:** `mibici-equipo16`

## 1. Objetivo

Desarrollar y evaluar modelos de clasificación multiclase para estimar el riesgo operativo de las estaciones del sistema MiBici, utilizando información histórica de viajes, características de las estaciones y variables temporales.

El objetivo es identificar tres posibles condiciones operativas:

- **Normal:** comportamiento del flujo neto dentro de los umbrales establecidos.
- **Sin bicicletas:** flujo neto asociado con un posible agotamiento de bicicletas disponibles.
- **Sin anclajes:** flujo neto asociado con una posible saturación de los espacios disponibles.

La unidad de análisis corresponde a una estación durante una franja temporal de tres horas.

Los resultados buscan aportar información para anticipar desequilibrios operativos y apoyar decisiones relacionadas con la redistribución de bicicletas.

## 2. Metodología

### 2.1. Fuente y periodo de los datos

Se utilizaron datos históricos del sistema MiBici correspondientes al periodo comprendido entre el 1 de octubre de 2024 y el 30 de septiembre de 2026.

La versión de los datos utilizada en los experimentos se identifica mediante:

`324489249bb4`

Los conjuntos se dividieron cronológicamente para evitar utilizar observaciones futuras durante el entrenamiento.

| Conjunto | Periodo | Observaciones |
|---|---|---:|
| Entrenamiento | Octubre 2024 – marzo 2026 | 2,056,720 |
| Validación | Abril – junio 2026 | 342,160 |
| Prueba | Julio – septiembre 2026 | 345,920 |

### 2.2. Construcción de la variable objetivo

La variable objetivo corresponde a una clasificación aproximada del riesgo operativo de cada estación.

Las etiquetas se construyeron utilizando el flujo neto de bicicletas, la capacidad de la estación y un parámetro de sensibilidad α = 0.5.

Es importante señalar que las categorías representan **etiquetas proxy**, no mediciones directas del inventario disponible en las estaciones.

Por tanto, las predicciones deben interpretarse como indicadores aproximados de desequilibrio operativo y no como confirmaciones de que una estación se encuentra físicamente vacía o saturada.

### 2.3. Variables predictoras

Se utilizaron siete variables:

| Variable | Descripción |
|---|---|
| `flujo_lag_1` | Flujo neto registrado en la franja anterior |
| `flujo_lag_56` | Flujo neto registrado 56 franjas antes |
| `capacidad` | Capacidad de la estación |
| `hora` | Hora de inicio de la franja |
| `dia_semana` | Día de la semana |
| `latitud` | Coordenada geográfica de la estación |
| `longitud` | Coordenada geográfica de la estación |

### 2.4. Estrategia de evaluación

Se comparó un baseline de persistencia frente a tres candidatos de aprendizaje automático:

1. Regresión logística.
2. HistGradientBoostingClassifier.
3. RandomForestClassifier.

La métrica principal de selección fue **F1 macro**, debido a que permite evaluar el desempeño promedio entre las tres categorías sin privilegiar exclusivamente a la clase mayoritaria.

La selección de hiperparámetros se realizó utilizando el conjunto de validación, mientras que el conjunto de prueba se reservó para la evaluación final.

Todos los experimentos utilizaron la semilla aleatoria `16`.

## 3. Experimentos y comparación de modelos

### 3.1. Modelos evaluados

Se evaluó un baseline de persistencia y tres algoritmos de aprendizaje automático:

- **Persistencia:** referencia basada en el comportamiento previo de la estación.
- **Regresión logística:** modelo lineal utilizado como alternativa interpretable.
- **HistGradientBoostingClassifier (HGB):** algoritmo de ensamble basado en árboles de decisión.
- **RandomForestClassifier:** ensamble de árboles de decisión con selección aleatoria de observaciones y variables.

Los experimentos se ejecutaron mediante MLflow, registrando métricas, hiperparámetros, artefactos y configuraciones.

Para los candidatos de aprendizaje automático se realizaron búsquedas de hiperparámetros, utilizando corridas anidadas (*nested runs*) para conservar la trazabilidad de las configuraciones evaluadas.

### 3.2. Hiperparámetros seleccionados

La selección se realizó maximizando la métrica `val_f1_macro`.

| Modelo | Hiperparámetros seleccionados |
|---|---|
| Regresión logística | `C=0.1` |
| HistGradientBoosting | `learning_rate=0.1`, `class_weight=None`, `max_iter=300` |
| Random Forest | `n_estimators=200`, `max_depth=15`, `class_weight=None` |

En HistGradientBoosting se utilizó `early_stopping=False` y `random_state=16`.

### 3.3. Resultados comparativos

La siguiente tabla presenta el desempeño de los modelos en los conjuntos de validación y prueba.

| Modelo | F1 macro validación | F1 macro prueba |
|---|---:|---:|
| **Persistencia** | **0.546124** | **0.567355** |
| HistGradientBoosting | 0.530071 | 0.535265 |
| Random Forest | 0.498602 | 0.501764 |
| Regresión logística | 0.359038 | 0.360418 |

El baseline de persistencia obtuvo el mejor desempeño en ambos conjuntos.

Entre los modelos de aprendizaje automático, HistGradientBoosting presentó el mayor F1 macro de validación, seguido de Random Forest y regresión logística.

### 3.4. Comparación con el baseline

Para cuantificar las diferencias de desempeño, se calculó la variación absoluta del F1 macro en el conjunto de prueba respecto al baseline.

| Modelo | Diferencia absoluta de F1 macro |
|---|---:|
| HistGradientBoosting | −0.032090 |
| Random Forest | −0.065591 |
| Regresión logística | −0.206937 |

Los tres candidatos presentaron un desempeño inferior al baseline.

En particular, HistGradientBoosting obtuvo un F1 macro de prueba de 0.535265, frente a 0.567355 de persistencia.

Por tanto, los resultados no aportan evidencia de una mejora predictiva respecto al método de referencia.

### 3.5. Selección del mejor candidato

HistGradientBoosting fue seleccionado como el mejor **candidato de aprendizaje automático**, utilizando exclusivamente el F1 macro de validación.

Su selección se fundamentó en:

1. Mayor F1 macro de validación entre los candidatos ML.
2. Desempeño superior al de Random Forest y regresión logística.
3. Disponibilidad de artefactos y configuración reproducible en MLflow.
4. Compatibilidad con el proceso de registro y recuperación de modelos.

Sin embargo, esta selección debe distinguirse de una recomendación de implementación operativa.

**El baseline de persistencia continúa siendo la alternativa con mejor desempeño observado.**

Por ello, HistGradientBoosting se considera un candidato experimental para futuras iteraciones, pero no se recomienda sustituir el baseline sin demostrar una mejora consistente mediante nuevas evaluaciones.

### 3.6. Identificadores de las corridas

Las corridas principales de MLflow permiten recuperar los resultados de cada experimento.

| Modelo | Run ID |
|---|---|
| Persistencia | `16635b47d53746858e1775ceda2a9a9e` |
| HistGradientBoosting | `e09ab3e67b414b6dbbd4c4bab57afe0b` |
| Random Forest | `c8114b46ce86448d995950a4c76fa34c` |
| Regresión logística | `a2fa179b969c4465b961bc1b2e0167b4` |

Estos identificadores facilitan la consulta de métricas, parámetros y artefactos de los experimentos registrados.

## 4. Registro del modelo y reproducibilidad

### 4.1. Registro en MLflow Model Registry

El modelo HistGradientBoosting, seleccionado como el mejor candidato de aprendizaje automático para P2, fue registrado en MLflow Model Registry.

| Propiedad | Valor |
|---|---|
| Nombre del modelo | `mibici-equipo16-p2-riesgo` |
| Versión registrada | `1` |
| Alias | `champion` |
| Algoritmo | `HistGradientBoostingClassifier` |
| Run ID de origen | `e09ab3e67b414b6dbbd4c4bab57afe0b` |
| Versión de datos | `324489249bb4` |

El modelo puede recuperarse utilizando la siguiente referencia:

`models:/mibici-equipo16-p2-riesgo@champion`

Se verificó su recuperación mediante `mlflow.sklearn.load_model()`, obteniendo correctamente un objeto de tipo `HistGradientBoostingClassifier`.

El alias `champion` identifica al mejor candidato ML registrado, pero no implica que el modelo supere al baseline ni que se encuentre desplegado en producción.

### 4.2. Firma del modelo

Se verificó que el modelo registrado contiene una firma de entrada y salida.

| Variable de entrada | Tipo | Obligatoria |
|---|---|---|
| `flujo_lag_1` | double | No |
| `flujo_lag_56` | double | No |
| `capacidad` | integer | Sí |
| `hora` | integer | Sí |
| `dia_semana` | integer | Sí |
| `latitud` | double | Sí |
| `longitud` | double | Sí |

La salida se representa mediante un tensor de objetos con dimensión variable:

`Tensor('object', (-1,))`

Esta firma permite documentar el esquema esperado para realizar predicciones.

### 4.3. Artefactos registrados

Se verificó que la corrida ganadora contiene cinco directorios de artefactos.

| Directorio | Archivos principales |
|---|---|
| `config/` | `MLproject`, `config.py`, `config_efectiva.json`, `python_env.yaml`, `requirements.txt` |
| `data/` | `manifiesto_datos.json` |
| `metrics/` | `busqueda.csv`, `metricas.json` |
| `model/` | `MLmodel`, `model.skops`, dependencias y ejemplos de entrada |
| `plots/` | `confusion_val.png`, `confusion_test.png` |

Estos artefactos permiten consultar la configuración del experimento, identificar la versión de los datos y recuperar los resultados de evaluación.

### 4.4. Configuración experimental

Los experimentos utilizaron la semilla aleatoria `16` y particiones temporales diferenciadas para entrenamiento, validación y prueba.

La configuración efectiva del modelo HistGradientBoosting fue:

| Parámetro | Valor |
|---|---|
| `learning_rate` | `0.1` |
| `class_weight` | `None` |
| `max_iter` | `300` |
| `early_stopping` | `False` |
| `random_state` | `16` |
| `alpha` | `0.5` |

Los metadatos de la corrida indican que el ajuste del modelo se realizó exclusivamente con el conjunto de entrenamiento, sin reajustarlo posteriormente utilizando validación.

La versión de datos `324489249bb4` permite identificar el conjunto utilizado en el experimento.

### 4.5. Verificaciones técnicas realizadas

Se ejecutaron las siguientes verificaciones:

| Verificación | Resultado |
|---|---|
| Pruebas automatizadas de modelado | 17 aprobadas |
| Ruff sobre los archivos modificados | Sin errores |
| Ruff sobre el repositorio completo | Sin errores |
| Recuperación del modelo mediante alias | Exitosa |
| Consulta de la firma del modelo | Exitosa |
| Consulta de parámetros y versión de datos | Exitosa |
| Consulta de directorios y archivos de artefactos | Exitosa |

Estas verificaciones respaldan la trazabilidad y recuperación del modelo registrado.

No obstante, la reproducción completa de los experimentos desde un entorno limpio queda pendiente de verificación independiente.

## 5. Análisis crítico, limitaciones y conclusiones

### 5.1. Interpretación de los resultados

Los experimentos permitieron comparar tres algoritmos de aprendizaje automático con un baseline de persistencia para clasificar el riesgo operativo de las estaciones MiBici.

El baseline obtuvo el mejor F1 macro tanto en validación (0.546124) como en prueba (0.567355).

Entre los candidatos de aprendizaje automático, HistGradientBoosting presentó el mejor desempeño, con un F1 macro de 0.530071 en validación y 0.535265 en prueba.

La diferencia de F1 macro en prueba respecto al baseline fue de −0.032090, equivalente a una disminución aproximada de 3.21 puntos porcentuales en la escala de F1 macro.

Por tanto, los resultados no justifican sustituir el baseline de persistencia por alguno de los modelos entrenados.

No obstante, los experimentos permitieron implementar y verificar un flujo reproducible de búsqueda de hiperparámetros, evaluación, registro de artefactos y recuperación de modelos mediante MLflow.

### 5.2. Limitaciones metodológicas

**Etiquetas proxy**

La principal limitación corresponde a la construcción de la variable objetivo.

Las categorías `normal`, `sin_bicicletas` y `sin_anclajes` se derivan del flujo neto de viajes respecto a la capacidad de cada estación, utilizando α = 0.5.

Estas etiquetas no representan mediciones directas del inventario real de bicicletas o anclajes.

Por consiguiente, un modelo con buen desempeño respecto a estas etiquetas no necesariamente identificaría con la misma precisión situaciones reales de estaciones vacías o saturadas.

**Sensibilidad del parámetro α**

La definición de las categorías depende del valor de α.

En esta evaluación se utilizó α = 0.5. No se realizó un análisis de sensibilidad con valores alternativos, como 0.4 y 0.6.

Por ello, todavía no conocemos la estabilidad de las clasificaciones y del desempeño de los modelos ante modificaciones de los umbrales.

**Distribución de las clases**

La métrica F1 macro permite considerar las tres categorías sin ponderarlas únicamente por su frecuencia.

Sin embargo, una evaluación operativa más completa requiere examinar el recall y la precisión por clase, especialmente para las categorías `sin_bicicletas` y `sin_anclajes`.

También es necesario revisar las matrices de confusión para identificar los tipos de errores más frecuentes.

**Generalización temporal**

La evaluación utilizó una partición cronológica, reservando el periodo de julio a septiembre de 2026 como conjunto de prueba.

Aunque esta estrategia permite evaluar el comportamiento en un periodo posterior al entrenamiento, no garantiza que el desempeño se mantenga ante cambios estacionales, modificaciones de infraestructura o alteraciones de los patrones de movilidad.

### 5.3. Consideraciones operativas y computacionales

La selección de un modelo para apoyar decisiones de redistribución de bicicletas debe considerar tanto su desempeño predictivo como su costo computacional y facilidad de mantenimiento.

El baseline de persistencia constituye una referencia sencilla, con menores requisitos de entrenamiento que los modelos de ensamble.

HistGradientBoosting y Random Forest incorporan mayor complejidad y requieren procesos adicionales de entrenamiento, evaluación y actualización.

En los experimentos realizados no se documentó una comparación sistemática de tiempos de entrenamiento, latencia de inferencia ni consumo de memoria.

Por tanto, no es posible establecer conclusiones cuantitativas sobre la eficiencia computacional relativa de los candidatos.

Dado que ninguno superó al baseline en F1 macro, actualmente no existe evidencia suficiente para justificar una mayor complejidad operativa exclusivamente por una mejora predictiva.

### 5.4. Recomendaciones

Para futuras iteraciones se recomienda:

1. **Evaluar la sensibilidad de α:** repetir los experimentos con valores de 0.4 y 0.6 para determinar cómo cambian las etiquetas y el desempeño.
2. **Analizar los errores por clase:** revisar las matrices de confusión, el recall y la precisión de las categorías operativamente críticas.
3. **Validar las etiquetas con inventarios reales:** incorporar información sobre bicicletas y anclajes disponibles, si existe una fuente confiable.
4. **Ampliar la evaluación temporal:** comparar el desempeño en diferentes periodos para identificar posibles variaciones estacionales.
5. **Medir costos computacionales:** registrar tiempos de entrenamiento e inferencia y consumo de memoria.
6. **Conservar el baseline como referencia:** exigir mejoras consistentes antes de recomendar un modelo de mayor complejidad.

### 5.5. Conclusión

Se implementó y evaluó un proceso experimental para la clasificación multiclase del riesgo operativo de las estaciones MiBici.

Se comparó un baseline de persistencia con regresión logística, HistGradientBoosting y Random Forest, utilizando F1 macro como métrica principal.

HistGradientBoosting fue el mejor candidato de aprendizaje automático y se registró en MLflow Model Registry como versión 1, con el alias `champion`.

La recuperación del modelo mediante su alias, la disponibilidad de una firma de entrada y salida y el registro de configuraciones, métricas y artefactos respaldan la trazabilidad del proceso experimental.

Sin embargo, el baseline de persistencia obtuvo el mejor desempeño predictivo en validación y prueba.

En consecuencia, **la principal aportación de esta etapa es la construcción de un proceso experimental trazable y reproducible, no la demostración de una mejora predictiva frente al baseline**.

Antes de utilizar los resultados para apoyar decisiones operativas, será necesario validar las etiquetas de riesgo, analizar los errores por clase y comprobar la estabilidad temporal del desempeño.