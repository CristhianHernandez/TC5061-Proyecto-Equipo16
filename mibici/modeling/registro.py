"""Entrada `register` del MLflow Project: lleva el modelo elegido de una pregunta a producción.

Esquema de ciclo de vida: **aliases** del Model Registry (las etapas Staging/Production están
obsoletas desde MLflow 2.9). La versión en producción tiene el alias `champion` y la versión que
reemplaza conserva `challenger`, de modo que un rollback es solo reasignar el alias. El código que
consume el modelo siempre carga `models:/<nombre>@champion` y no cambia al promover otra versión.

Uso:

    mlflow run . -e register --env-manager uv --experiment-name mibici-equipo16 -P question=p1
    mlflow run . -e register ... -P question=p1 -P run_id=<id de la corrida elegida>

Sin `run_id` se propone la corrida registrable con mejor métrica principal de **validación**;
la decisión final se justifica en el reporte con desempeño, estabilidad y costo.
"""

import os

from loguru import logger
import mlflow
from mlflow import MlflowClient
import typer

from mibici.config import (
    ALIAS_PRODUCCION,
    ALIAS_RETADOR,
    DESCRIPCIONES_MODELOS,
    EQUIPO,
    EXPERIMENTO_MLFLOW,
    METRICA_PRINCIPAL,
    MINIMIZAR,
    MLFLOW_TRACKING_URI,
    MODELOS_REGISTRADOS,
)
from mibici.modeling.comun import es_mejor, metrica_baseline, modulo_pregunta

app = typer.Typer()


def mejor_corrida(question: str):
    """Corrida candidata o de búsqueda con modelo servible y mejor métrica de validación."""
    cliente = MlflowClient()
    experimento = cliente.get_experiment_by_name(EXPERIMENTO_MLFLOW)
    if experimento is None:
        raise ValueError(f"No existe el experimento {EXPERIMENTO_MLFLOW}")
    metrica = f"val_{METRICA_PRINCIPAL[question]}"
    runs = cliente.search_runs(
        [experimento.experiment_id],
        filter_string=(
            f"tags.question = '{question}' and tags.modelo_registrable = 'true' "
            "and tags.stage_of_work != 'baseline' and attributes.status = 'FINISHED'"
        ),
        order_by=[f"metrics.{metrica} {'ASC' if MINIMIZAR[question] else 'DESC'}"],
    )
    runs = [r for r in runs if metrica in r.data.metrics]
    if not runs:
        raise ValueError(f"No hay corridas registrables de {question} con {metrica}")
    return runs[0]


def supera_baseline(question: str, run) -> str:
    """Compara la métrica principal de prueba con la del baseline de referencia (sí/no/sin dato).

    El registro no se bloquea: la decisión se justifica en el reporte, pero queda constancia en la
    versión y una advertencia si el modelo no mejora al baseline.
    """
    referencia = metrica_baseline(question)
    clave = f"test_{METRICA_PRINCIPAL[question]}"
    if referencia is None or clave not in run.data.metrics:
        return "sin dato"
    if es_mejor(question, run.data.metrics[clave], referencia[1]):
        return "sí"
    logger.warning(
        f"{run.info.run_name}: {clave}={run.data.metrics[clave]:.4f} no supera al baseline "
        f"({referencia[1]:.4f}); justificar la elección en el reporte"
    )
    return "no"


def version_de_corrida(nombre: str, run_id: str):
    """Versión ya registrada de una corrida, para no duplicar versiones al repetir el comando."""
    for version in MlflowClient().search_model_versions(f"name = '{nombre}'"):
        if version.run_id == run_id:
            return version
    return None


def promover(question: str, run_id: str | None = None) -> dict:
    """Registra el modelo de una corrida y le asigna `champion`; el anterior pasa a `challenger`.

    Agrega descripción al modelo y a la versión, y etiquetas de linaje en la versión
    (experimento, corrida, commit y versión de datos). Termina verificando que el modelo en
    producción se carga por nombre y alias y que predice sobre su ejemplo de entrada.
    """
    question = question.lower()
    modulo_pregunta(question)  # valida la pregunta
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    cliente = MlflowClient()
    run = cliente.get_run(run_id) if run_id else mejor_corrida(question)
    etiquetas = run.data.tags
    if etiquetas.get("question") != question or etiquetas.get("modelo_registrable") != "true":
        raise ValueError(
            f"La corrida {run.info.run_id} no tiene un modelo registrable de {question}"
        )
    nombre = MODELOS_REGISTRADOS[question]
    try:
        cliente.get_registered_model(nombre)
    except mlflow.exceptions.MlflowException:
        cliente.create_registered_model(nombre)
    cliente.update_registered_model(nombre, description=DESCRIPCIONES_MODELOS[question])
    cliente.set_registered_model_tag(nombre, "equipo", EQUIPO)
    cliente.set_registered_model_tag(nombre, "question", question)
    cliente.set_registered_model_tag(nombre, "esquema_ciclo_vida", "aliases champion/challenger")

    version = version_de_corrida(nombre, run.info.run_id)
    if version is None:
        version = mlflow.register_model(etiquetas["model_uri"], nombre)
    metrica = f"val_{METRICA_PRINCIPAL[question]}"
    experimento = cliente.get_experiment(run.info.experiment_id).name
    supera = supera_baseline(question, run)
    linaje = {
        "experimento": experimento,
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "git_commit": etiquetas.get("git_commit", ""),
        "version_datos": etiquetas.get("version_datos", ""),
        "model_family": etiquetas.get("model_family", ""),
        "stage_of_work": etiquetas.get("stage_of_work", ""),
        metrica: str(run.data.metrics.get(metrica)),
        "supera_baseline": supera,
    }
    for clave, valor in linaje.items():
        cliente.set_model_version_tag(nombre, version.version, clave, valor)
    cliente.update_model_version(
        nombre,
        version.version,
        description=(
            f"Corrida {run.info.run_name} ({run.info.run_id}) del experimento {experimento}; "
            f"commit {linaje['git_commit']}; versión de datos {linaje['version_datos']}; "
            f"{metrica}={linaje[metrica]}; supera al baseline en prueba: {supera}."
        ),
    )

    try:
        anterior = cliente.get_model_version_by_alias(nombre, ALIAS_PRODUCCION)
    except mlflow.exceptions.MlflowException:
        anterior = None
    retador = anterior.version if anterior and anterior.version != version.version else None
    if retador is not None:
        cliente.set_registered_model_alias(nombre, ALIAS_RETADOR, retador)
    cliente.set_registered_model_alias(nombre, ALIAS_PRODUCCION, version.version)

    uri = f"models:/{nombre}@{ALIAS_PRODUCCION}"
    modelo = mlflow.pyfunc.load_model(uri)
    ejemplo = modelo.input_example
    if ejemplo is None:
        raise ValueError(f"{uri} no tiene ejemplo de entrada para verificarlo")
    modelo.predict(ejemplo)
    return {
        "nombre": nombre,
        "version": version.version,
        "run_id": run.info.run_id,
        "uri": uri,
        "challenger": retador,
    }


@app.command()
def main(
    question: str = typer.Option(..., help="Pregunta: p1, p2 o p3"),
    run_id: str = typer.Option("", help="Corrida elegida; vacío propone la mejor en validación"),
):
    """Registra y promueve a producción el modelo elegido de una pregunta."""
    # `mlflow run` crea una corrida para este comando: se reanuda al final para documentarlo.
    propio = os.environ.pop("MLFLOW_RUN_ID", None)
    resultado = promover(question, run_id.strip() or None)
    if propio is not None:
        with mlflow.start_run(run_id=propio):
            mlflow.set_tags(
                {
                    "mlflow.runName": f"{question.lower()}-registro-{ALIAS_PRODUCCION}",
                    "question": question.lower(),
                    "uso": "registro",
                }
            )
            mlflow.log_params(
                {
                    "modelo_registrado": resultado["nombre"],
                    "version": resultado["version"],
                    "run_origen": resultado["run_id"],
                    "challenger": str(resultado["challenger"]),
                }
            )
    typer.echo(
        f"{resultado['uri']} -> versión {resultado['version']} (corrida {resultado['run_id']}); "
        f"challenger: {resultado['challenger']}"
    )


if __name__ == "__main__":
    app()
