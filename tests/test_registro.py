"""Model Registry: registro con descripción y linaje, aliases champion/challenger y carga."""

import mlflow
from mlflow import MlflowClient
import pytest

from mibici.config import ALIAS_PRODUCCION, ALIAS_RETADOR, MODELOS_REGISTRADOS
from mibici.modeling import comun, predict, registro


@pytest.fixture(scope="module")
def candidatas(mlflow_sesion, conjuntos):
    """Baseline y dos candidatas de P2 con distinta regularización."""
    comun.ejecutar("p2", None, None, {}, False, conjuntos["p2"])
    ids = [
        comun.ejecutar("p2", "logistica", None, {"C": c}, False, conjuntos["p2"])
        for c in (0.01, 1.0)
    ]
    return ids


def test_propone_la_mejor_en_validacion(candidatas):
    propuesta = registro.mejor_corrida("p2")
    corridas = [MlflowClient().get_run(r) for r in candidatas]
    assert propuesta.data.metrics["val_f1_macro"] == max(
        r.data.metrics["val_f1_macro"] for r in corridas
    )
    assert propuesta.data.tags["stage_of_work"] != "baseline"


def test_promover_registra_linaje_descripcion_y_alias(candidatas, conjuntos):
    cliente = MlflowClient()
    nombre = MODELOS_REGISTRADOS["p2"]
    primero = registro.promover("p2", candidatas[0])
    version = cliente.get_model_version_by_alias(nombre, ALIAS_PRODUCCION)
    assert version.version == primero["version"]
    assert version.run_id == candidatas[0]
    assert cliente.get_registered_model(nombre).description
    assert candidatas[0] in version.description
    run = cliente.get_run(candidatas[0])
    for clave in ("experimento", "run_id", "git_commit", "version_datos", "val_f1_macro"):
        assert version.tags[clave]
    assert version.tags["supera_baseline"] in ("sí", "no")
    assert version.tags["experimento"] == "mibici-equipo16"
    assert version.tags["git_commit"] == run.data.tags["git_commit"]
    assert version.tags["version_datos"] == run.data.tags["version_datos"]

    segundo = registro.promover("p2", candidatas[1])
    assert segundo["challenger"] == primero["version"]
    assert cliente.get_model_version_by_alias(nombre, ALIAS_RETADOR).version == primero["version"]
    assert cliente.get_model_version_by_alias(nombre, ALIAS_PRODUCCION).run_id == candidatas[1]

    # Repetir la promoción de la misma corrida no crea otra versión
    repetido = registro.promover("p2", candidatas[1])
    assert repetido["version"] == segundo["version"]
    assert repetido["challenger"] is None

    modelo = mlflow.pyfunc.load_model(f"models:/{nombre}@{ALIAS_PRODUCCION}")
    columnas = modelo.metadata.get_input_schema().input_names()
    assert len(modelo.predict(conjuntos["p2"]["test"][columnas].head(4))) == 4


def test_no_promueve_baselines_sin_modelo(mlflow_sesion, conjuntos):
    run_id = comun.ejecutar("p2", "persistencia", None, {}, False, conjuntos["p2"])
    with pytest.raises(ValueError, match="registrable"):
        registro.promover("p2", run_id)


def test_predict_usa_el_modelo_en_produccion(candidatas, conjuntos, mlflow_sesion):
    registro.promover("p2", candidatas[1])
    resumen = predict.predecir("p2", conjuntos["p2"])
    assert resumen["uri"] == f"models:/{MODELOS_REGISTRADOS['p2']}@{ALIAS_PRODUCCION}"
    assert 0 <= resumen["test_f1_macro"] <= 1
    assert resumen["filas"] == len(conjuntos["p2"]["test"])
    assert (mlflow_sesion["ruta"] / "procesados" / "predicciones_p2.parquet").exists()
