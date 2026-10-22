"""Registro de experimentos en MLflow con las convenciones del equipo.

Todo run del proyecto se abre con `iniciar_run`, que garantiza:
- experimento `mibici-equipo16` y nombre de run descriptivo;
- tags obligatorias: equipo, pregunta (P1/P2/P3) y commit de Git, más las de filtrado
  `question`, `model_family`, `stage_of_work` y `version_datos`;
- parámetros de reproducibilidad: rango de fechas, versión de los datos y semilla;
- ningún parámetro con nombre de dato personal y métricas con prefijo del conjunto.

Cuando el código corre con `mlflow run`, MLflow crea el run antes de ejecutarlo y pasa su
identificador en `MLFLOW_RUN_ID`; `iniciar_run` reanuda ese run en lugar de crear otro.
"""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import subprocess

from loguru import logger
import mlflow
from mlflow import MlflowClient

from mibici.config import (
    EQUIPO,
    EXPERIMENTO_MLFLOW,
    FECHA_FIN,
    FECHA_INICIO,
    INICIO_PRUEBA,
    INICIO_VALIDACION,
    MANIFIESTO_DATOS,
    MLFLOW_TRACKING_URI,
    PROJ_ROOT,
    SEMILLA,
)

PREGUNTAS = ("P1", "P2", "P3")
PREFIJOS_METRICAS = ("train_", "val_", "test_")
ETAPAS_TRABAJO = ("baseline", "search", "candidate", "analysis")
# Fragmentos que no pueden aparecer en nombres de parámetros (datos personales del usuario)
TERMINOS_PERSONALES = ("usuario", "genero", "nacimiento", "edad")


def obtener_commit(repo: Path = PROJ_ROOT) -> str:
    """Devuelve el hash del commit actual; si hay cambios sin commit, agrega el sufijo `-sucio`.

    El sufijo deja evidencia en MLflow de que el run no corresponde exactamente a un commit.
    """
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    cambios = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{sha}-sucio" if cambios else sha


def version_datos(manifiesto: Path = MANIFIESTO_DATOS) -> str:
    """Devuelve los primeros 12 caracteres del SHA-256 del manifiesto de datos.

    El manifiesto lista cada archivo con su número de registros y su hash, así que cualquier
    cambio en los datos cambia esta versión.
    """
    if not manifiesto.exists():
        raise FileNotFoundError(f"No existe {manifiesto}; ejecuta primero `make data`.")
    return hashlib.sha256(manifiesto.read_bytes()).hexdigest()[:12]


def validar_params(params: dict) -> None:
    """Rechaza parámetros cuyo nombre sugiera un dato personal."""
    prohibidos = [k for k in params if any(t in k.lower() for t in TERMINOS_PERSONALES)]
    if prohibidos:
        raise ValueError(f"Parámetros con datos personales no permitidos: {prohibidos}")


def registrar_metricas(metricas: dict[str, float]) -> None:
    """Registra métricas exigiendo el prefijo del conjunto (`train_`, `val_` o `test_`)."""
    sin_prefijo = [k for k in metricas if not k.startswith(PREFIJOS_METRICAS)]
    if sin_prefijo:
        raise ValueError(f"Métricas sin prefijo de conjunto: {sin_prefijo}")
    mlflow.log_metrics(metricas)


def _etiquetas(pregunta: str, etiquetas: dict | None) -> dict:
    """Combina las tags obligatorias del equipo con las de filtrado de la corrida."""
    etiquetas = dict(etiquetas or {})
    etapa = etiquetas.get("stage_of_work")
    if etapa is not None and etapa not in ETAPAS_TRABAJO:
        raise ValueError(f"stage_of_work inválido: {etapa!r}; usa uno de {ETAPAS_TRABAJO}")
    return {
        "equipo": EQUIPO,
        "pregunta": pregunta,
        "question": pregunta.lower(),
        "git_commit": obtener_commit(),
        "version_datos": version_datos(MANIFIESTO_DATOS),
        **etiquetas,
    }


def _abrir_run(nombre: str):
    """Reanuda el run creado por `mlflow run` o abre uno nuevo en el experimento del equipo.

    `mlflow run` deja el identificador en MLFLOW_RUN_ID. Se retira de las variables de entorno
    después de reanudar: si se quedara, cualquier run anidado reutilizaría el run padre.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    run_id = os.environ.pop("MLFLOW_RUN_ID", None)
    if run_id is None:
        mlflow.set_experiment(EXPERIMENTO_MLFLOW)
        return mlflow.start_run(run_name=nombre)
    experimento = MlflowClient().get_experiment(MlflowClient().get_run(run_id).info.experiment_id)
    if experimento.name != EXPERIMENTO_MLFLOW:
        raise ValueError(
            f"El run {run_id} pertenece al experimento {experimento.name!r}; ejecuta mlflow run "
            f"con --experiment-name {EXPERIMENTO_MLFLOW}"
        )
    return mlflow.start_run(run_id=run_id)


@contextmanager
def iniciar_run(
    pregunta: str, nombre: str, params: dict | None = None, etiquetas: dict | None = None
):
    """Abre (o reanuda, bajo `mlflow run`) un run con las tags y parámetros obligatorios.

    Uso:
        with iniciar_run("P1", "p1-naive", {"modelo": "naive"},
                         {"stage_of_work": "baseline", "model_family": "naive"}) as run:
            registrar_metricas({"val_mae": 3.2})
    """
    pregunta = pregunta.upper()
    if pregunta not in PREGUNTAS:
        raise ValueError(f"Pregunta inválida: {pregunta!r}; usa una de {PREGUNTAS}")
    params = params or {}
    validar_params(params)
    tags = _etiquetas(pregunta, etiquetas)

    with _abrir_run(nombre) as run:
        mlflow.set_tags({"mlflow.runName": nombre, **tags})
        mlflow.log_params(
            {
                "fecha_inicio": FECHA_INICIO,
                "inicio_validacion": INICIO_VALIDACION,
                "inicio_prueba": INICIO_PRUEBA,
                "fecha_fin": FECHA_FIN,
                "version_datos": tags["version_datos"],
                "semilla": SEMILLA,
                **params,
            }
        )
        logger.info(f"Run {nombre!r} ({pregunta}) iniciado: {run.info.run_id}")
        yield run


@contextmanager
def iniciar_run_anidado(nombre: str, params: dict | None = None, etiquetas: dict | None = None):
    """Abre un run hijo del run activo (una combinación de una búsqueda de hiperparámetros).

    Hereda las tags de equipo, pregunta, commit y versión de datos del run padre.
    """
    padre = mlflow.active_run()
    if padre is None:
        raise RuntimeError("Un run anidado necesita un run padre activo")
    params = params or {}
    validar_params(params)
    heredadas = {
        k: v
        for k, v in MlflowClient().get_run(padre.info.run_id).data.tags.items()
        if k in ("equipo", "pregunta", "question", "git_commit", "version_datos", "model_family")
    }
    with mlflow.start_run(run_name=nombre, nested=True) as run:
        mlflow.set_tags({**heredadas, "stage_of_work": "search", **(etiquetas or {})})
        mlflow.log_params(params)
        yield run
