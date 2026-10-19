
# TC5061-Proyecto-Equipo16

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Predicción de demanda, riesgo de desabasto y clustering de estaciones de MiBici.

## Integrantes y roles

| Integrante | Usuario de GitHub | Rol | Responsabilidades principales |
|---|---|---|---|
| Cristhian Hernandez CH1 | `@CristhianHernandez` | Líder técnico / MLOps | Repositorio, flujo Git, `mibici/config.py`, `mibici/tracking.py`, integración final y reporte |
| Luis Hernandez LH3 | `@Luispiriu` | Ingeniería de datos | Ingesta (`mibici/dataset.py`), Parquet, versionado de datos y tratamiento de datos personales |
| Horacio Hernandez HH2 | `@usuario` | Exploración y features | Notebook de exploración y `mibici/features.py` para P1, P2 y P3 |
| Derick Hernandez DH4 | `@zohet` | Modelado y experimentación | `mibici/modeling/`, gráficas y experimento en MLflow |

Trabajamos en cadena: cada integrante revisa el PR anterior y construye su parte encima
(CH1 → LH3 → HH2 → DH4 → cierre de CH1). Las tareas de cada uno y los contratos entre módulos están en
[`references/plan-de-implementacion.md`](references/plan-de-implementacion.md).

## Flujo de trabajo en Git

Seguimos [GitHub Flow](https://docs.github.com/es/get-started/using-github/github-flow):

1. Actualizar `main` local: `git switch main && git pull`.
2. Crear una rama desde `main` con la convención de nombres (ver abajo).
3. Hacer commits pequeños con [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/).
4. Subir la rama y abrir un Pull Request hacia `main` usando la plantilla del repositorio.
5. Otro integrante revisa; se resuelven todos los comentarios antes de integrar.
6. Se integra con **squash merge** y la rama se borra automáticamente.

### Protección de `main`

`main` está protegida con un ruleset (`proteger-main`), sin excepciones para nadie:

- No se permite push directo: todo cambio entra por Pull Request.
- Se requiere 1 aprobación; un push nuevo invalida aprobaciones previas.
- Todas las conversaciones del PR deben estar resueltas.
- No se permite borrar la rama ni hacer force push.

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

Los tipos son los mismos que los de las ramas. Como se integra con squash merge, el **título del PR**
también debe seguir esta convención, porque se convierte en el commit final en `main`.

### Reglas del equipo

- Cada integrante abre al menos un PR y revisa al menos un PR de otro integrante.
- Nunca se suben datos, modelos serializados, carpetas de MLflow (`mlruns/`, `mlartifacts/`)
  ni credenciales (`.env`); el `.gitignore` los excluye.

## Ingesta y procesamiento de datos

La ingesta se realiza desde `mibici/dataset.py` y utiliza 24 meses de datos abiertos de MiBici, de octubre de 2024 a septiembre de 2026, además del catálogo de estaciones.

### Ejecución

Con Python 3.13 y las dependencias instaladas:

```bash
make requirements
make data
```

La ingesta descarga los CSV faltantes, normaliza columnas y fechas, elimina los datos personales (`Usuario_Id`, `Genero` y `Año_de_nacimiento`) y genera:

- 24 archivos Parquet de viajes en `data/processed/viajes/`.
- Un archivo `data/processed/estaciones.parquet` con el catálogo de estaciones.
- El manifiesto `references/manifiesto_datos.json` con los hashes SHA-256 y el número de viajes por mes.

En la ejecución realizada se procesaron los 24 meses y se registraron 484 estaciones.

### Lectura de datos

Los archivos procesados se pueden consultar desde Python:

```python
from mibici.dataset import cargar_viajes, cargar_estaciones

viajes = cargar_viajes(["2024_10", "2024_11"])
estaciones = cargar_estaciones()
```

Si se ejecuta `cargar_viajes()` sin indicar meses, se cargan todos los disponibles.

### Pruebas

Las pruebas de ingesta se encuentran en `tests/test_dataset.py`. Utilizan datos sintéticos para comprobar la lectura de CSV, transformación de fechas, exclusión de datos personales, procesamiento de estaciones y generación del manifiesto.

```bash
python -m pytest tests/test_dataset.py -v
```

Las 7 pruebas de ingesta pasaron correctamente con Python 3.13.

**Nota:** durante el desarrollo se encontraron respuestas HTTP 406 al descargar algunos archivos desde Docker. Los CSV pudieron descargarse mediante PowerShell y procesarse correctamente. Queda pendiente resolver la descarga automática desde un entorno limpio.

## Integrantes y roles

| Integrante | Usuario de GitHub | Rol | Responsabilidades principales |
|---|---|---|---|
| Cristhian Hernandez CH1| `@CristhianHernandez` | Líder técnico / MLOps | Repositorio, flujo Git, `mibici/config.py`, `mibici/tracking.py`, integración final y reporte |
| Luis Hernandez LH3| `@Luispiriu` | Ingeniería de datos | Ingesta (`mibici/dataset.py`), Parquet, versionado de datos y tratamiento de datos personales |
| Horacio Hernandez HH2| `@usuario` | Exploración y features | Notebook de exploración y `mibici/features.py` para P1, P2 y P3 |
| Derick Hernandez DH4| `@zohet` | Modelado y experimentación | `mibici/modeling/`, gráficas y experimento en MLflow |

Trabajamos en cadena: cada integrante revisa el PR anterior y construye su parte encima
(CH1 → LH3 → HH2 → DH4 → cierre de CH1). Las tareas de cada uno y los contratos entre módulos están en
[`references/plan-de-implementacion.md`](references/plan-de-implementacion.md).

## Flujo de trabajo en Git

Seguimos [GitHub Flow](https://docs.github.com/es/get-started/using-github/github-flow):

1. Actualizar `main` local: `git switch main && git pull`.
2. Crear una rama desde `main` con la convención de nombres (ver abajo).
3. Hacer commits pequeños con [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/).
4. Subir la rama y abrir un Pull Request hacia `main` usando la plantilla del repositorio.
5. Otro integrante revisa; se resuelven todos los comentarios antes de integrar.
6. Se integra con **squash merge** y la rama se borra automáticamente.

### Protección de `main`

`main` está protegida con un ruleset (`proteger-main`), sin excepciones para nadie:

- No se permite push directo: todo cambio entra por Pull Request.
- Se requiere 1 aprobación; un push nuevo invalida aprobaciones previas.
- Todas las conversaciones del PR deben estar resueltas.
- No se permite borrar la rama ni hacer force push.

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

Los tipos son los mismos que los de las ramas. Como se integra con squash merge, el **título del PR**
también debe seguir esta convención, porque se convierte en el commit final en `main`.

### Reglas del equipo

- Cada integrante abre al menos un PR y revisa al menos un PR de otro integrante.
- Nunca se suben datos, modelos serializados, carpetas de MLflow (`mlruns/`, `mlartifacts/`)
  ni credenciales (`.env`); el `.gitignore` los excluye.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- Parquet de viajes y estaciones para modelado.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         mibici and configuration for tools like black
│
├── references         <- Documentación y manifiesto SHA-256 de los datos.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
├── tests
│   ├── test_config.py
│   ├── test_dataset.py <- Pruebas de ingesta y versionado de datos
│   └── test_tracking.py
│
└── mibici             <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes mibici a Python module
    │
    ├── config.py      <- Store useful variables and configuration
    │
    ├── dataset.py     <- Descarga, normalización y versionado de datos MiBici
    │
    ├── features.py    <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py <- Code to run model inference with trained models
    │   └── train.py   <- Code to train models
    │
    └── plots.py       <- Code to create visualizations
```

--------
