"""Entrada `predict` del MLflow Project: usa el modelo en producción de una pregunta.

Carga `models:/<nombre>@champion` del Model Registry (no un archivo local), predice sobre el
conjunto de prueba y, si la pregunta tiene objetivo, reporta su métrica principal. Las
predicciones se guardan en data/processed/, que está fuera de Git.

    mlflow run . -e predict --env-manager uv --experiment-name mibici-equipo16 -P question=p1
"""

import os

from loguru import logger
import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
import typer

from mibici.config import (
    ALIAS_PRODUCCION,
    MLFLOW_TRACKING_URI,
    MODELOS_REGISTRADOS,
    PROCESSED_DATA_DIR,
)
from mibici.modeling.comun import modulo_pregunta

app = typer.Typer()


def cargar_produccion(question: str):
    """Carga el modelo con alias champion de la pregunta y devuelve (modelo, versión, uri)."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    nombre = MODELOS_REGISTRADOS[question]
    version = MlflowClient().get_model_version_by_alias(nombre, ALIAS_PRODUCCION)
    uri = f"models:/{nombre}@{ALIAS_PRODUCCION}"
    return mlflow.pyfunc.load_model(uri), version, uri


def predecir(question: str, conjuntos: dict | None = None) -> dict:
    """Predice el conjunto de prueba con el modelo en producción y resume el resultado."""
    question = question.lower()
    modulo = modulo_pregunta(question)
    modelo, version, uri = cargar_produccion(question)
    conjuntos = conjuntos if conjuntos is not None else modulo.preparar_conjuntos()
    prueba, objetivo = modulo.datos_prediccion(conjuntos)
    columnas = modelo.metadata.get_input_schema().input_names()
    prediccion = np.asarray(modelo.predict(prueba[columnas]))
    resumen = {"uri": uri, "version": version.version, "filas": len(prueba)}
    if objetivo is not None:
        if question == "p1":
            resumen["test_mae"] = float(
                modulo.metricas_demanda(prueba[objetivo], prediccion)["mae"]
            )
        else:
            resumen["test_f1_macro"] = float(modulo.f1_macro(prueba[objetivo], prediccion))
    else:
        resumen["grupos"] = len(np.unique(prediccion))
    salida = PROCESSED_DATA_DIR / f"predicciones_{question}.parquet"
    salida.parent.mkdir(parents=True, exist_ok=True)
    columnas_id = [c for c in ("estacion_id", "fecha", "fecha_franja", "hora") if c in prueba]
    pd.DataFrame(prueba[columnas_id]).assign(prediccion=prediccion).to_parquet(salida, index=False)
    resumen["salida"] = str(salida)
    return resumen


@app.command()
def main(question: str = typer.Option(..., help="Pregunta: p1, p2 o p3")):
    """Predice con el modelo en producción (alias champion) de una pregunta."""
    resumen = predecir(question)
    run_id = os.environ.pop("MLFLOW_RUN_ID", None)
    if run_id is not None:
        # Bajo `mlflow run` deja constancia de qué versión se usó (sin datos ni predicciones).
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tags(
                {
                    "mlflow.runName": f"{question}-prediccion-{ALIAS_PRODUCCION}",
                    "question": question.lower(),
                    "uso": "prediccion",
                }
            )
            mlflow.log_params({"modelo_uri": resumen["uri"], "version": resumen["version"]})
            mlflow.log_metrics({k: v for k, v in resumen.items() if k.startswith("test_")})
    logger.success(f"{resumen}")


if __name__ == "__main__":
    app()
