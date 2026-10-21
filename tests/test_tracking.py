import mlflow
import pytest

from mibici import tracking


@pytest.fixture
def mlflow_temporal(tmp_path, monkeypatch):
    """MLflow y manifiesto en un directorio temporal, sin tocar los del proyecto."""
    manifiesto = tmp_path / "manifiesto_datos.json"
    manifiesto.write_text('{"archivos": []}')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(tracking, "MANIFIESTO_DATOS", manifiesto)
    return manifiesto


def test_version_datos_cambia_con_el_manifiesto(tmp_path):
    manifiesto = tmp_path / "m.json"
    manifiesto.write_text("a")
    v1 = tracking.version_datos(manifiesto)
    manifiesto.write_text("b")
    assert v1 != tracking.version_datos(manifiesto)
    assert len(v1) == 12


def test_version_datos_sin_manifiesto(tmp_path):
    with pytest.raises(FileNotFoundError, match="make data"):
        tracking.version_datos(tmp_path / "no_existe.json")


def test_obtener_commit_es_hash_de_git():
    commit = tracking.obtener_commit().removesuffix("-sucio")
    assert len(commit) == 40
    int(commit, 16)


def test_rechaza_params_con_datos_personales():
    with pytest.raises(ValueError, match="personales"):
        tracking.validar_params({"modelo": "kmeans", "Usuario_Id": 1})


def test_rechaza_pregunta_invalida(mlflow_temporal):
    with pytest.raises(ValueError, match="Pregunta"), tracking.iniciar_run("P4", "run-invalido"):
        pass


def test_run_lleva_tags_params_y_metricas(mlflow_temporal):
    with tracking.iniciar_run("P1", "p1-prueba", {"modelo": "naive"}) as run:
        tracking.registrar_metricas({"val_mae": 1.5})
        with pytest.raises(ValueError, match="prefijo"):
            tracking.registrar_metricas({"mae": 1.5})

    datos = mlflow.get_run(run.info.run_id).data
    assert datos.tags["equipo"] == "16"
    assert datos.tags["pregunta"] == "P1"
    assert datos.tags["git_commit"]
    assert datos.params["modelo"] == "naive"
    assert datos.params["version_datos"] == tracking.version_datos(mlflow_temporal)
    assert datos.params["semilla"] == "16"
    assert datos.metrics == {"val_mae": 1.5}
    assert datos.tags["question"] == "p1"
    assert datos.tags["version_datos"] == datos.params["version_datos"]


def test_reanuda_el_run_creado_por_mlflow_run(mlflow_temporal, monkeypatch):
    """Bajo `mlflow run` se usa el run de MLFLOW_RUN_ID y se libera para los runs anidados."""
    mlflow.set_tracking_uri(tracking.MLFLOW_TRACKING_URI)
    experimento = mlflow.set_experiment("mibici-equipo16")
    creado = mlflow.MlflowClient().create_run(experimento.experiment_id).info.run_id
    monkeypatch.setenv("MLFLOW_RUN_ID", creado)
    with tracking.iniciar_run(
        "P2", "p2-hgb", {"modelo": "hgb"}, {"stage_of_work": "search", "model_family": "boosting"}
    ) as run:
        assert run.info.run_id == creado
        assert "MLFLOW_RUN_ID" not in tracking.os.environ
        with tracking.iniciar_run_anidado("p2-hgb-hijo", {"learning_rate": 0.1}) as hijo:
            assert hijo.info.run_id != creado
    padre = mlflow.get_run(creado)
    assert padre.info.run_name == "p2-hgb"
    assert padre.data.tags["stage_of_work"] == "search"
    hijo = mlflow.get_run(hijo.info.run_id)
    assert hijo.data.tags["mlflow.parentRunId"] == creado
    assert hijo.data.tags["question"] == "p2"
    assert hijo.data.tags["model_family"] == "boosting"


def test_rechaza_run_de_otro_experimento(mlflow_temporal, monkeypatch):
    mlflow.set_tracking_uri(tracking.MLFLOW_TRACKING_URI)
    otro = mlflow.set_experiment("Default-otro")
    creado = mlflow.MlflowClient().create_run(otro.experiment_id).info.run_id
    monkeypatch.setenv("MLFLOW_RUN_ID", creado)
    with pytest.raises(ValueError, match="--experiment-name"), tracking.iniciar_run("P1", "x"):
        pass


def test_rechaza_etapa_desconocida(mlflow_temporal):
    with (
        pytest.raises(ValueError, match="stage_of_work"),
        tracking.iniciar_run("P1", "x", etiquetas={"stage_of_work": "produccion"}),
    ):
        pass
