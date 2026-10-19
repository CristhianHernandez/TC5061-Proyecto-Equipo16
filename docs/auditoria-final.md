# Auditoría final — Primera entrega MiBici (Equipo 16)

Revisión hecha el 2026-10-18 sobre la rama `feat/modelos-mlflow` (a partir de `main` en `29e296d`), contra
la actividad *Construye | Proyecto MiBici: propuesta y base reproducible | Colaborativo* y su rúbrica.
Cada estado se basa en evidencia verificada: comandos ejecutados, archivos del repositorio, el
experimento de MLflow o la API pública de GitHub. Lo que no se pudo comprobar está marcado como
**No verificable** y requiere revisión manual.

Estados: **Cumple** · **Parcial** · **No cumple** · **No verificable**.

## Resumen por criterio

| Criterio | Peso | Estado | Comentario |
|---|---:|---|---|
| Formulación de las preguntas | 15% | Cumple | P1–P3 con tipo, granularidad, horizonte, usuario, decisión y métricas |
| Datos e ingesta reproducible | 15% | Cumple | `make data` reproduce el manifiesto versionado byte a byte (tras dos correcciones) |
| Pipeline, modelos, métricas y arquitectura | 15% | Cumple | Baseline y candidatos por pregunta, métricas justificadas, diagrama |
| Experimento base en MLflow | 15% | Cumple | 10 runs con params, métricas, tags y artefactos; 4 capturas reales |
| Estructura del repositorio | 15% | Cumple | Cookiecutter, `config.py` central, dependencias fijas; detalle menor en `predict.py` |
| Flujo de trabajo Git/GitHub | 10% | **Parcial** | Faltan revisiones de PR y un PR propio de LH3; el ruleset no exige aprobación |
| Feature engineering | 10% | Cumple | Variables sin fuga, con pruebas; dos variables propuestas no implementadas |
| README y claridad de la entrega | 5% | Cumple | README completo y verificado contra los comandos reales |

Esta tabla no es una calificación: solo resume qué está respaldado por evidencia.

## Matriz de cumplimiento

### 1. Formulación de las preguntas

| Requisito | Estado | Evidencia | Acción |
|---|---|---|---|
| ML Canvas completo (10 bloques) | Cumple | `references/ml-canvas.md` | — |
| P1, P2 y P3 formuladas (qué, tipo, salida) | Cumple | `references/propuesta-parte-a.md` §2 | — |
| Granularidad y horizonte por pregunta | Cumple | Propuesta §2 (estación × día D+1; estación × franja 3 h; una fila por estación) | — |
| Usuario y decisión que habilita | Cumple | Propuesta §2 y Canvas, bloque *Decisiones* | — |
| Proxy de riesgo de P2 justificado y con limitaciones | Cumple | Propuesta §2 (P2); param `etiqueta_riesgo` en cada run P2 | — |
| Análisis de sensibilidad de α (0.4 y 0.6) | Parcial | Propuesto en §2; no ejecutado | Ejecutar `preparar_datos_p2(..., alpha=0.4/0.6)` en la siguiente entrega |
| Propuesta sin marcas de borrador | Parcial | El encabezado aún dice "borrador" | CH1 decide si retira la marca en la versión final |

### 2. Datos e ingesta reproducible

| Requisito | Estado | Evidencia | Acción |
|---|---|---|---|
| Fuente y condiciones de uso | Cumple | Propuesta §1.1 y §1.5 | — |
| Rango temporal y número de meses | Cumple | `config.py` (2024-10-01 a 2026-09-30); 24 archivos en el manifiesto | — |
| Viajes y estaciones reales | Cumple | Manifiesto: 9,043,925 viajes; nomenclatura: 484 estaciones (470 activas, 8,171 anclajes) | — |
| `make data` ejecutable desde cero | Cumple (corregido) | El portal respondía 406 al User-Agent del script; corregido en `c6573c7` (cabeceras en `config.py`). Ejecución completa el 2026-10-18 | — |
| Versionado de datos | Cumple | Con los mismos CSV, el manifiesto regenerado es idéntico al versionado (`git diff` vacío); `version_datos=4d8c2a1292fc` en cada run | — |
| Municipio de cada estación | Cumple (corregido) | Era nulo en 483 de 484 estaciones; corregido en `c6573c7`: 357 GDL, 110 ZPN, 17 TLQ | — |
| Eliminación de datos personales | Cumple | Los Parquet solo tienen `viaje_id, inicio, fin, origen_id, destino_id`; `grep` sin coincidencias de `usuario_id/genero/nacimiento` en `mlruns/` | — |
| Pruebas de ingesta | Cumple | `tests/test_dataset.py` (7 pruebas), con un caso del formato real `(TLQ-003) …` | — |

### 3. Pipeline, modelos, métricas y arquitectura

| Requisito | Estado | Evidencia | Acción |
|---|---|---|---|
| Pipeline por pregunta | Cumple | `mibici/modeling/train.py` → `features.py` → `p1_demanda.py`, `p2_riesgo.py`, `p3_clusters.py` | — |
| Baseline y candidato por pregunta | Cumple | P1: naive estacional y promedio estación × día frente a HGB Poisson; P2: Dummy y persistencia frente a regresión logística y HGB; P3: municipio frente a KMeans y Ward | — |
| Métricas justificadas | Cumple | Propuesta §2; P1 MAE/RMSE/desviación Poisson; P2 F1 macro/recall/AP/balanced accuracy; P3 silueta/DB/CH/ARI | — |
| Selección en validación, prueba una vez | Cumple | `comun.seleccionar_modelo` no recibe prueba; `tests/test_modeling.py::test_seleccion_no_recibe_prueba_y_ajustes_no_aprenden_validacion` | — |
| Diagrama de arquitectura y stack | Cumple | `references/arquitectura.md` (Mermaid y tabla de herramientas) | — |

### 4. Experimento base en MLflow

| Requisito | Estado | Evidencia | Acción |
|---|---|---|---|
| Experimento `mibici-equipo16` | Cumple | Servidor `make mlflow MLFLOW_PORT=5001`, experimento id 1 | — |
| Runs con nombre descriptivo | Cumple | 10 runs `p1-*`, `p2-*`, `p3-*`, todos `FINISHED` | — |
| Tags equipo, pregunta y commit | Cumple | `equipo=16`, `pregunta`, `git_commit=400a5ad…` (sin sufijo `-sucio`) | — |
| Params: rango de fechas, versión de datos, semilla, hiperparámetros | Cumple | `fecha_inicio`, `inicio_validacion`, `inicio_prueba`, `fecha_fin`, `version_datos`, `semilla=16`, hiperparámetros reales elegidos | — |
| Métricas con prefijo de conjunto | Cumple | `registrar_metricas()` lo exige; prueba `test_diez_runs_completos_metricas_tags_artefactos` | — |
| Al menos un artefacto por run | Cumple | Figuras PNG, `metricas.json`, `rejilla.csv` / `busqueda_k.csv` | — |
| Reproducibilidad | Cumple | Dos ejecuciones completas (Codex en SQLite temporal y la real en el servidor) dieron las mismas métricas; `test_misma_semilla_metricas_identicas` | — |
| Cuatro capturas de la interfaz | Cumple | Lista de runs, detalle de `p1-hgb-poisson`, artefacto `confusion_test.png` de `p2-hgb` y comparación de los tres runs de P1, tomadas del servidor real (fuera del repositorio, en la carpeta del entregable) | — |

### 5. Estructura del repositorio

| Requisito | Estado | Evidencia | Acción |
|---|---|---|---|
| Estructura Cookiecutter Data Science | Cumple | `data/`, `docs/`, `models/`, `notebooks/`, `references/`, `reports/figures/`, `mibici/` | — |
| `config.py` centralizado | Cumple | Rutas, fechas, semillas, URLs, cabeceras, rejillas y `α` solo en `mibici/config.py` | — |
| `dataset.py` ejecutable | Cumple | `python -m mibici.dataset` / `make data` | — |
| `features.py` reutilizable y documentado | Cumple (corregido) | El merge `e988e06` había duplicado funciones (`make lint` fallaba) y perdido pruebas; restaurado en `9879920` desde `3c2eb47` | — |
| Notebook de exploración | Cumple | `notebooks/1.0-hh2-exploracion-viajes.ipynb` (restaurado en `9879920`) | — |
| Dependencias con versión fija | Cumple | `requirements.txt` (todas con `==`) | — |
| ML Canvas y diagrama en `references/` | Cumple | `ml-canvas.md`, `arquitectura.md` | — |
| `.gitignore` correcto | Cumple | Excluye `data/`, Parquet/CSV, `mlruns/`, `mlartifacts/`, `mlflow.db`, `.env`, modelos y ahora las figuras generadas | — |
| Sin archivos prohibidos en el historial | Cumple | `git log --all --name-only` sobre todas las ramas: ningún CSV, Parquet, `.env`, modelo ni carpeta de MLflow | — |
| Código plantilla sin usar | Parcial | `mibici/modeling/predict.py` conserva el ejemplo de Cookiecutter | Implementarlo cuando exista inferencia o eliminarlo (decisión de CH1) |

### 6. Flujo de trabajo Git/GitHub

Verificado con la API pública de GitHub el 2026-10-18 (sin autenticación; no se pueden ver configuraciones privadas).

| Requisito | Estado | Evidencia | Acción |
|---|---|---|---|
| `main` protegida | Parcial | Ruleset `protect_main` activo: exige PR, bloquea borrado y force push, pero `required_approving_review_count = 0` y no exige resolver conversaciones | CH1: subir a 1 aprobación y activar *Require conversation resolution* |
| Commits de los cuatro integrantes | Cumple | Autores: Cristhian Hernandez, Luispiriu, Horacio007, Derick Zohet | — |
| Un PR abierto por integrante | **No cumple** | PR #1 CristhianHernandez, PR #2 y #3 HHSavenger; LH3 no tiene PR propio (sus commits entraron por la rama de HH2); el PR de DH4 está pendiente de abrir | DH4 abre el PR 4; LH3 abre un PR propio, por ejemplo con el ajuste del ruleset o la documentación de datos |
| Una revisión por integrante | **No cumple** | PR #1: un comentario de Luispiriu (sin aprobación); PR #2 y #3: ninguna revisión; ambos los integró `zohet` | DH4 deja su revisión del PR #3 (de HH2); CH1 revisa y aprueba el PR 4; HH2 revisa el PR de LH3 |
| Convenciones de ramas | Cumple | `feat/base-del-proyecto`, `feat/ingesta-datos`, `feat/features`, `feat/p1-demanda-diaria`, `feat/modelos-mlflow` | — |
| Convenciones de commits | Parcial | La mayoría son Conventional Commits; el título del PR #1 (`Feat/base del proyecto`) no; PR #2 y #3 se integraron con merge commit, no con *squash* | Respetar el título y el *squash* en los PR que faltan |
| Integración mediante PR | Cumple | Todo en `main` entró por PR #1–#3 | — |

### 7. Feature engineering

| Requisito | Estado | Evidencia | Acción |
|---|---|---|---|
| Variables por pregunta | Cumple | P1: calendario, festivos, rezagos 1 y 7, media móvil 7; P2: hora, día, rezagos de flujo 1 y 56 franjas, capacidad; P3: 96 perfiles horarios, volumen, balance y duración | — |
| Sin fuga del futuro | Cumple | Rezagos y medias solo con el pasado; escalas, imputadores y promedios ajustados en train; pruebas de no fuga en `test_features.py` y `test_modeling.py` | — |
| Media móvil de 28 días y ubicación en P1 | Parcial | La propuesta menciona la media de 28 días; no está en `features.py` (sí se usan capacidad, latitud y longitud) | Agregar en la siguiente entrega o ajustar la propuesta |
| Filtro de primera actividad | Parcial (limitación) | `calcular_primera_actividad` usa todo el rango para decidir desde qué fecha entra cada estación; selecciona filas, pero no aporta valores futuros a las variables | Documentado como limitación en el entregable |

### 8. README y claridad

| Requisito | Estado | Evidencia | Acción |
|---|---|---|---|
| Propósito, preguntas, datos y estructura | Cumple | `README.md` | — |
| Instalación, ingesta, MLflow (puerto 5001 en macOS) y entrenamiento | Cumple | Comandos ejecutados el 2026-10-18 | — |
| Convenciones Git, roles y buenas prácticas | Cumple | `README.md`; usuario de HH2 corregido a `@HHSavenger` (autor de los PR #2 y #3) | — |

## Incidencias corregidas en esta revisión

1. **Merge `e988e06`:** `mibici/features.py` quedó con seis funciones y una importación duplicadas (8 errores de
   `ruff check`) y `make lint` fallaba en `main`; además se perdieron 640 líneas de pruebas de HH2 y la versión final del notebook. Se restauraron
   los tres archivos desde `3c2eb47` sin cambios (`9879920`).
2. **Descarga bloqueada:** el portal responde HTTP 406 a los User-Agent `python-requests/*` y
   `Mozilla/5.0`. Las cabeceras pasan a `config.py` con un User-Agent descriptivo (`c6573c7`).
3. **Municipio nulo:** el regex esperaba `GDL…`, pero los nombres reales son `(GDL-001) …`; el baseline
   de P3 no habría funcionado (`c6573c7`).
4. **Makefile:** la regla `mlflow` estaba duplicada (avisos de *overriding recipe*); se dejó una sola y se
   agregó `make train`.
5. **README:** secciones duplicadas y una descripción de la protección de `main` que no coincidía con el
   ruleset real.

## Pendientes que requieren intervención humana

1. Abrir el PR 4 (`feat/modelos-mlflow` → `main`) y asignarlo a CH1; CH1 lo revisa, aprueba e integra.
2. DH4 deja una revisión (aunque sea posterior a la integración) en el PR #3 de HH2.
3. LH3 abre un PR propio y otro integrante lo revisa.
4. CH1 ajusta el ruleset a 1 aprobación y conversaciones resueltas.
5. Agregar la portada al documento académico, revisar el texto final y exportarlo a PDF.
6. Verificar en Canvas el formato exacto de la entrega (no hay copia local de las instrucciones).
