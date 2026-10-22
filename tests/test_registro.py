"""Model Registry: registro con descripción y linaje, aliases champion/challenger y carga."""

from types import SimpleNamespace

import mlflow
from mlflow import MlflowClient
import pytest
from typer.testing import CliRunner

from mibici.config import ALIAS_PRODUCCION, ALIAS_RETADOR, MODELOS_REGISTRADOS
from mibici.modeling import comun, predict, registro


def test_p1_desempata_por_estabilidad_y_costo_sin_elegir_modelo_peor_que_naive(monkeypatch):
    def corrida(nombre, val, test, estabilidad, duracion, etapa="candidate"):
        return SimpleNamespace(
            info=SimpleNamespace(run_id=nombre),
            data=SimpleNamespace(
                metrics={
                    "val_mae": val,
                    "test_mae": test,
                    "test_mae_std_mensual": estabilidad,
                    "train_duracion_s": duracion,
                    "test_latencia_ms_por_1000": duracion,
                },
                tags={"stage_of_work": etapa},
            ),
        )

    bosque = corrida("rf", 6.08, 5.86, 0.75, 275)
    hgb = corrida("hgb", 6.14, 5.79, 0.74, 1.4)
    busqueda = corrida("search", 6.14, 5.79, 0.74, 1.4, "search")
    poisson = corrida("poisson", 5.9, 15.6, 0.9, 2)
    cliente = SimpleNamespace(
        get_experiment_by_name=lambda _: SimpleNamespace(experiment_id="1"),
        search_runs=lambda *_args, **_kwargs: [poisson, bosque, busqueda, hgb],
    )
    monkeypatch.setattr(registro, "MlflowClient", lambda: cliente)
    monkeypatch.setattr(registro, "metrica_baseline", lambda _: ("baseline", 7.35))
    assert registro.mejor_corrida("p1").info.run_id == "hgb"

    otro = corrida("otro", 6.12, 5.8, 0.74, 10)
    cliente.search_runs = lambda *_args, **_kwargs: [bosque, otro, hgb]
    assert registro.mejor_corrida("p1").info.run_id == "hgb"


def test_cli_register_normaliza_comillas_vacias_de_mlflow_en_windows(monkeypatch):
    llamadas = []

    def promover_falso(question, run_id):
        llamadas.append((question, run_id))
        return {
            "nombre": "mibici-equipo16-p1-demanda",
            "version": "1",
            "run_id": "corrida",
            "uri": "models:/mibici-equipo16-p1-demanda@champion",
            "challenger": None,
        }

    monkeypatch.setattr(registro, "promover", promover_falso)
    resultado = CliRunner().invoke(
        registro.app, ["--question", "p1", "--run-id", '""']
    )
    assert resultado.exit_code == 0, resultado.output
    assert llamadas == [("p1", None)]


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
