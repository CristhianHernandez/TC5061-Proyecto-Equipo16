"""El MLflow Project declara entorno y entradas, y no repite valores de config.py."""

import re

from mlflow.projects._project_spec import load_project
import pytest
from typer.testing import CliRunner
import yaml

from mibici.config import PROJ_ROOT
from mibici.modeling import train

HIPERPARAMETROS_TRAIN = (
    "features",
    "learning_rate",
    "max_leaf_nodes",
    "max_iter",
    "n_estimators",
    "max_depth",
    "alpha",
    "C",
    "class_weight",
    "k",
    "n_components",
    "covariance_type",
)


@pytest.fixture(scope="module")
def proyecto():
    return load_project(str(PROJ_ROOT))


def test_entradas_ingest_train_register_predict_y_compare(proyecto):
    assert {"ingest", "train", "register", "predict", "compare"} <= set(proyecto._entry_points)
    assert "mibici.modeling.comparar" in proyecto.get_entry_point("compare").command
    assert "mibici.dataset" in proyecto.get_entry_point("ingest").command
    entrada = proyecto.get_entry_point("train")
    assert {"question", "model", "stage", "search", *HIPERPARAMETROS_TRAIN} <= set(
        entrada.parameters
    )
    for nombre in entrada.parameters:
        assert "{" + nombre + "}" in entrada.command


def test_valores_por_defecto_vienen_de_config(proyecto):
    """Los defaults del MLproject son vacíos: no repiten rutas, fechas, semillas ni hiperparámetros."""
    permitidos = {"", "p1", "false"}
    for nombre in ("train", "register", "predict"):
        for parametro in proyecto.get_entry_point(nombre).parameters.values():
            assert str(parametro.default) in permitidos, parametro.name
    lineas = (PROJ_ROOT / "MLproject").read_text(encoding="utf-8").splitlines()
    texto = "\n".join(linea for linea in lineas if not linea.lstrip().startswith("#"))
    assert not re.search(r"20\d\d-\d\d-\d\d|semilla|seed|data/", texto)


def test_entorno_con_las_versiones_fijas():
    entorno = yaml.safe_load((PROJ_ROOT / "python_env.yaml").read_text(encoding="utf-8"))
    assert str(entorno["python"]) == "3.13"
    assert "-r requirements.txt" in entorno["dependencies"]
    lineas = [
        linea.strip()
        for linea in (PROJ_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if linea.strip() and linea.strip() != "-e ."
    ]
    assert lineas and all("==" in linea for linea in lineas)


def test_gitignore_excluye_mlflow_datos_y_modelos():
    reglas = (PROJ_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for regla in ("mlflow.db", "mlruns/", "mlartifacts/", "/data/", "*.parquet", "/models/*"):
        assert regla in reglas


def test_cli_train_convierte_y_omite_vacios(monkeypatch):
    llamadas = []
    monkeypatch.setattr(
        train, "ejecutar", lambda *args, **kwargs: llamadas.append((args, kwargs)) or "run"
    )
    argumentos = ["--question", "p2", "--model", "hgb", "--class-weight", "none"]
    argumentos += [
        "--features",
        "v2",
        "--learning-rate",
        "0.05",
        "--max-leaf-nodes",
        "63",
        "--C",
        "",
    ]
    resultado = CliRunner().invoke(train.app, argumentos)
    assert resultado.exit_code == 0, resultado.output
    (args, kwargs) = llamadas[0]
    assert args[:3] == ("p2", "hgb", None)
    assert args[3] == {
        "class_weight": None,
        "features": "v2",
        "learning_rate": 0.05,
        "max_leaf_nodes": 63,
    }
    assert kwargs == {"search": False}
    assert CliRunner().invoke(train.app, ["--search", "tal vez"]).exit_code != 0


def test_cli_train_normaliza_comillas_vacias_de_mlflow_en_windows(monkeypatch):
    llamadas = []
    monkeypatch.setattr(
        train, "ejecutar", lambda *args, **kwargs: llamadas.append((args, kwargs)) or "run"
    )
    resultado = CliRunner().invoke(
        train.app,
        [
            "--question",
            "p1",
            "--model",
            "hgb",
            "--stage",
            '""',
            "--search",
            "true",
            "--features",
            "v1",
            "--learning-rate",
            '""',
            "--max-leaf-nodes",
            '""',
        ],
    )
    assert resultado.exit_code == 0, resultado.output
    (args, kwargs) = llamadas[0]
    assert args[:3] == ("p1", "hgb", None)
    assert args[3] == {"features": "v1"}
    assert kwargs == {"search": True}
