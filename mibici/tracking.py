"""Registro de experimentos en MLflow con las convenciones del equipo.

Todo run del proyecto se abre con `iniciar_run`, que garantiza:
- experimento `mibici-equipo16` y nombre de run descriptivo;
- tags obligatorias: equipo, pregunta (P1/P2/P3) y commit de Git;
- parámetros de reproducibilidad: rango de fechas, versión de los datos y semilla;
- ningún parámetro con nombre de dato personal y métricas con prefijo del conjunto.
"""

from contextlib import contextmanager
import hashlib
from pathlib import Path
import subprocess

from loguru import logger
import mlflow

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


@contextmanager
def iniciar_run(pregunta: str, nombre: str, params: dict | None = None):
    """Abre un run de MLflow con las tags y parámetros obligatorios del equipo.

    Uso:
        with iniciar_run("P1", "p1-naive-estacional", {"modelo": "naive"}) as run:
            registrar_metricas({"val_mae": 3.2})
    """
    if pregunta not in PREGUNTAS:
        raise ValueError(f"Pregunta inválida: {pregunta!r}; usa una de {PREGUNTAS}")
    params = params or {}
    validar_params(params)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENTO_MLFLOW)
    with mlflow.start_run(run_name=nombre) as run:
        mlflow.set_tags({"equipo": EQUIPO, "pregunta": pregunta, "git_commit": obtener_commit()})
        mlflow.log_params(
            {
                "fecha_inicio": FECHA_INICIO,
                "inicio_validacion": INICIO_VALIDACION,
                "inicio_prueba": INICIO_PRUEBA,
                "fecha_fin": FECHA_FIN,
                "version_datos": version_datos(MANIFIESTO_DATOS),
                "semilla": SEMILLA,
                **params,
            }
        )
        logger.info(f"Run {nombre!r} ({pregunta}) iniciado: {run.info.run_id}")
        yield run
