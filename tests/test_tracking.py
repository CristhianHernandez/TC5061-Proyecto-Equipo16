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
