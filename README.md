# TC5061-Proyecto-Equipo16

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Predicción de demanda, riesgo de desabasto y clustering de estaciones de MiBici, el sistema de bicicleta
pública del Área Metropolitana de Guadalajara. Este repositorio es la base reproducible del proyecto
integrador de TC5061: ingesta versionada, variables sin fuga del futuro, baselines y candidatos por
pregunta, y un experimento en MLflow.

## Preguntas

| | Pregunta | Tipo | Granularidad y horizonte | Usuario |
|---|---|---|---|---|
| **P1** | ¿Cuántos viajes iniciarán en cada estación mañana? | Regresión de conteos | Estación × día, D+1 | Operador del sistema |
| **P2** | ¿Qué estaciones corren riesgo de quedarse sin bicicletas o sin anclajes en la siguiente franja de 3 h? | Clasificación multiclase | Estación × franja de 3 h, siguiente franja | Equipo de rebalanceo |
| **P3** | ¿Qué tipos de estaciones existen según su patrón de uso? | Clustering | Una fila por estación | Planeación urbana |

MiBici no publica el inventario por estación, así que el riesgo de P2 es un **proxy**: una franja se
etiqueta `sin_bicicletas` si `salidas − llegadas ≥ α · capacidad` y `sin_anclajes` si
`salidas − llegadas ≤ −α · capacidad` (α = 0.5, franjas de 3 h). No es disponibilidad observada.

Detalle: [`references/propuesta-parte-a.md`](references/propuesta-parte-a.md),
[`references/ml-canvas.md`](references/ml-canvas.md) y [`references/arquitectura.md`](references/arquitectura.md).

## Datos

- **Fuente:** [datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/) (viajes mensuales y
  nomenclatura de estaciones `nomenclatura_2026_09.csv`).
- **Rango:** octubre de 2024 a septiembre de 2026, 24 archivos mensuales, **9,043,925 viajes**
  (cifra del manifiesto) y 484 estaciones en la nomenclatura (470 activas; 357 GDL, 110 ZPN, 17 TLQ).
- **Partición temporal:** entrenamiento oct-2024 – mar-2026, validación abr – jun 2026 y prueba
  jul – sep 2026. Los cortes viven solo en [`mibici/config.py`](mibici/config.py).
- **Datos personales:** `Usuario_Id`, `Genero` y `Año_de_nacimiento` se eliminan durante la ingesta, antes
  de escribir Parquet. Solo se trabaja con agregados por estación.
- **Versión de los datos:** [`references/manifiesto_datos.json`](references/manifiesto_datos.json) guarda el
  SHA-256 y el número de filas de cada CSV; cada run de MLflow registra los primeros 12 caracteres del hash
  de ese manifiesto como `version_datos`.

## Estructura del repositorio

```
├── Makefile               <- make data, make mlflow, make train, make lint, make test
├── README.md
├── data/                  <- fuera de Git (raw: CSV descargados; processed: Parquet)
├── docs/                  <- auditoría de la entrega
├── models/                <- reservado para modelos serializados (fuera de Git)
├── notebooks/             <- 1.0-hh2-exploracion-viajes.ipynb
├── references/            <- propuesta, ML Canvas, arquitectura, plan y manifiesto de datos
├── reports/figures/       <- figuras generadas (fuera de Git, salvo .gitkeep)
├── requirements.txt       <- dependencias con versión fija
├── tests/                 <- pruebas con datos sintéticos (pytest)
└── mibici/
    ├── config.py          <- única fuente de rutas, fechas, semillas, URLs y parámetros
    ├── dataset.py         <- descarga, normalización, eliminación de datos personales y manifiesto
    ├── features.py        <- limpieza, objetivos y variables de P1, P2 y P3; partición temporal
    ├── tracking.py        <- apertura de runs de MLflow con tags y parámetros obligatorios
    ├── plots.py           <- figuras que se registran como artefactos
    └── modeling/
        ├── p1_demanda.py  <- baselines y HistGradientBoostingRegressor (Poisson)
        ├── p2_riesgo.py   <- baselines, LogisticRegression y HistGradientBoostingClassifier
        ├── p3_clusters.py <- baseline por municipio, KMeans y Agglomerative (Ward)
        └── train.py       <- punto de entrada: python -m mibici.modeling.train
```

## Instalación

Requiere Python 3.13 y `make`.

```bash
git clone https://github.com/CristhianHernandez/TC5061-Proyecto-Equipo16.git
cd TC5061-Proyecto-Equipo16
python3.13 -m venv .venv          # Windows: py -3.13 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
make requirements                 # pip install -r requirements.txt (incluye el paquete mibici en modo editable)
cp .env.example .env              # Windows: copy .env.example .env
```

`.env` solo contiene la URI de MLflow y nunca se sube a Git.

## Ejecución en tres comandos

```bash
make data      # 1. descarga los 24 meses y la nomenclatura, genera Parquet y manifiesto (~770 MB de CSV)
make mlflow    # 2. en otra terminal: servidor de MLflow en http://127.0.0.1:5000
make train     # 3. entrena los 10 modelos (P1, P2 y P3) y registra los runs en mibici-equipo16
```

- `make data` no vuelve a descargar ni a procesar lo que ya existe. Con los mismos CSV, el manifiesto
  generado es idéntico al versionado.
- `make mlflow` guarda los runs en `mlflow.db` y los artefactos en `./mlruns/` (ambos fuera de Git).
  Abre <http://127.0.0.1:5000> y elige el experimento `mibici-equipo16`.
- Para entrenar una sola pregunta: `make train PREGUNTA=P1` (o `P2`, `P3`; por defecto `todas`), o bien
  `python -m mibici.modeling.train --pregunta P1`.

### macOS: puerto 5000 ocupado por AirPlay

En macOS el puerto 5000 lo usa AirPlay Receiver (el navegador recibe un 403). Usa otro puerto en **ambos**
comandos:

```bash
make mlflow MLFLOW_PORT=5001
make train MLFLOW_PORT=5001
```

Cuando `MLFLOW_PORT` se pasa en la línea de comandos, `make train` apunta a `http://127.0.0.1:<puerto>`.
Si prefieres no escribirlo cada vez, descomenta `MLFLOW_TRACKING_URI=http://127.0.0.1:5001` en `.env`
y levanta el servidor con `make mlflow MLFLOW_PORT=5001`.

## Qué registra cada run

Todos los runs se abren con `mibici.tracking.iniciar_run()`, que fija el experimento `mibici-equipo16` y
registra:

- **Tags:** `equipo`, `pregunta` (P1/P2/P3) y `git_commit` (con sufijo `-sucio` si había cambios sin commit).
- **Parámetros comunes:** fechas de inicio, cortes de validación y prueba, fecha de fin, `version_datos` y
  `semilla` (16, desde `config.py`), más los hiperparámetros reales de cada modelo.
- **Métricas** con prefijo de conjunto (`train_`, `val_`, `test_`), exigido por `registrar_metricas()`.
- **Artefactos:** gráficas (real contra predicho, matriz de confusión, mapa y perfiles de clusters) y
  tablas de métricas o búsqueda de hiperparámetros.

| Pregunta | Runs | Métrica principal |
|---|---|---|
| P1 | `p1-naive-estacional`, `p1-promedio-estacion-dia`, `p1-hgb-poisson` | MAE |
| P2 | `p2-dummy`, `p2-persistencia`, `p2-logistic-regression`, `p2-hgb` | F1 macro |
| P3 | `p3-municipio`, `p3-kmeans`, `p3-agglomerative-ward` | Silueta |

Los hiperparámetros se eligen solo con validación y la prueba se evalúa una vez con el modelo elegido.
En P3, `k` se elige con la silueta del periodo de entrenamiento; validación y prueba solo miden la
estabilidad de los grupos en periodos posteriores.

## Pruebas y estilo

```bash
make lint      # ruff format --check y ruff check
make test      # pytest con datos sintéticos; no descarga nada ni usa data/
```

Las pruebas de modelado usan un MLflow temporal (SQLite en un directorio temporal), así que no tocan el
experimento real.

## Reproducibilidad y buenas prácticas del equipo

- Rutas, fechas, semillas, URLs y parámetros solo en `mibici/config.py`; los demás módulos los importan.
- Partición temporal; rezagos y promedios usan solo información anterior al momento de predicción; imputadores,
  escaladores y promedios históricos se ajustan solo con entrenamiento.
- Dependencias con versión fija en `requirements.txt`.
- Ningún dato personal después de la ingesta y ninguno en params, tags o artefactos de MLflow.
- Nunca se suben datos, modelos serializados, `mlruns/`, `mlartifacts/`, `mlflow.db` ni `.env`;
  el `.gitignore` los excluye.
- Cada función pública tiene docstring y pruebas con datos sintéticos.

## Integrantes y roles

| Integrante | Usuario de GitHub | Rol | Responsabilidades principales |
|---|---|---|---|
| Cristhian Hernandez (CH1) | `@CristhianHernandez` | Líder técnico / MLOps | Repositorio, flujo Git, `mibici/config.py`, `mibici/tracking.py`, integración final y reporte |
| Luis Hernandez (LH3) | `@Luispiriu` | Ingeniería de datos | Ingesta (`mibici/dataset.py`), Parquet, versionado de datos y tratamiento de datos personales |
| Horacio Hernandez (HH2) | `@HHSavenger` | Exploración y features | Notebook de exploración y `mibici/features.py` para P1, P2 y P3 |
| Derick Hernandez (DH4) | `@zohet` | Modelado y experimentación | `mibici/modeling/`, `mibici/plots.py` y experimento en MLflow |

Trabajamos en cadena: cada integrante revisa el PR anterior y construye su parte encima
(CH1 → LH3 → HH2 → DH4 → cierre de CH1). Las tareas y los contratos entre módulos están en
[`references/plan-de-implementacion.md`](references/plan-de-implementacion.md).

## Flujo de trabajo en Git

Seguimos [GitHub Flow](https://docs.github.com/es/get-started/using-github/github-flow):

1. Actualizar `main` local: `git switch main && git pull`.
2. Crear una rama desde `main` con la convención de nombres (ver abajo).
3. Hacer commits pequeños con [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/).
4. Subir la rama y abrir un Pull Request hacia `main` usando la plantilla del repositorio.
5. Otro integrante revisa y aprueba; se resuelven todos los comentarios antes de integrar.
6. Se integra con **squash merge** y la rama se borra.

### Protección de `main`

`main` está protegida con el ruleset `protect_main`: no se permite push directo (todo cambio entra por
Pull Request), ni borrar la rama ni hacer force push. El acuerdo del equipo es integrar solo con una
aprobación de otro integrante; a la fecha de esta revisión (2026-10-18) el ruleset todavía no exige
aprobaciones (`required_approving_review_count = 0`), por lo que el acuerdo depende de la disciplina del
equipo hasta que CH1 lo ajuste.

### Convención de ramas

Formato `tipo/descripcion-corta`, en minúsculas y con guiones:

| Tipo | Uso | Ejemplo |
|---|---|---|
| `feat/` | Nueva funcionalidad | `feat/ingesta-datos` |
| `fix/` | Corrección de errores | `fix/codificacion-latin1` |
| `docs/` | Documentación | `docs/flujo-de-trabajo` |
| `refactor/` | Reestructura sin cambiar comportamiento | `refactor/config-centralizado` |
| `test/` | Pruebas | `test/ingesta` |
| `chore/` | Mantenimiento (dependencias, configuración) | `chore/fijar-dependencias` |

### Convención de commits

Formato `tipo(alcance opcional): descripción en imperativo`, por ejemplo:

```
feat(dataset): descargar meses y guardar en Parquet
fix(dataset): leer archivos con codificación latin-1
docs: documentar flujo de trabajo en Git
chore: fijar versiones en requirements.txt
```

Como se integra con squash merge, el **título del PR** también debe seguir esta convención, porque se
convierte en el commit final en `main`.

### Reglas del equipo

- Cada integrante abre al menos un PR y revisa al menos un PR de otro integrante.
- Nadie aprueba ni integra su propio PR.

## Propuesta académica

La propuesta de la primera entrega (ML Canvas, preguntas, datos, arquitectura, modelos y experimento base)
parte de los documentos de [`references/`](references/). El documento final en Word/PDF se entrega en Canvas
y no se versiona en este repositorio. La auditoría contra la rúbrica está en
[`docs/auditoria-final.md`](docs/auditoria-final.md).
