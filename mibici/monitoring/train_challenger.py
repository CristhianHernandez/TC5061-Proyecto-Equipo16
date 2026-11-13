"""Entrenamiento y registro del Challenger P2 sin modificar el Champion."""

import mlflow
from mlflow import MlflowClient

from mibici.config import (
    ALIAS_PRODUCCION,
    ALIAS_RETADOR,
    MLFLOW_TRACKING_URI,
    MODELOS_REGISTRADOS,
)
from mibici.modeling.comun import ejecutar
from mibici.modeling.registro import version_de_corrida


HIPERPARAMETROS = {
    "n_estimators": 200,
    "max_depth": 15,
    "class_weight": None,
}


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    cliente = MlflowClient()
    nombre = MODELOS_REGISTRADOS["p2"]

    champion_antes = cliente.get_model_version_by_alias(
        nombre, ALIAS_PRODUCCION
    )

    run_id = ejecutar(
        question="p2",
        model="rf",
        stage="candidate",
        hiperparametros=HIPERPARAMETROS,
        search=False,
    )

    corrida = cliente.get_run(run_id)
    uri = corrida.data.tags.get("model_uri")

    if not uri or corrida.data.tags.get("modelo_registrable") != "true":
        raise RuntimeError("La corrida no produjo un modelo registrable")

    version = version_de_corrida(nombre, run_id)

    if version is None:
        version = mlflow.register_model(uri, nombre)

    if str(version.version) == str(champion_antes.version):
        raise RuntimeError("La version Challenger coincide con el Champion")

    cliente.set_model_version_tag(
        nombre, version.version, "rol", "challenger"
    )
    cliente.set_model_version_tag(
        nombre, version.version, "modelo", "random_forest"
    )
    cliente.set_model_version_tag(
        nombre, version.version, "run_id", run_id
    )

    champion_despues = cliente.get_model_version_by_alias(
        nombre, ALIAS_PRODUCCION
    )

    if str(champion_antes.version) != str(champion_despues.version):
        raise RuntimeError("El alias Champion cambio inesperadamente")

    if version.status != "READY":
        raise RuntimeError("La version Challenger no esta lista")

    cliente.set_registered_model_alias(
        nombre, ALIAS_RETADOR, version.version
    )

    print(f"Run ID: {run_id}")
    print(f"Champion conservado: v{champion_despues.version}")
    print(f"Challenger registrado: v{version.version}")


if __name__ == "__main__":
    main()

