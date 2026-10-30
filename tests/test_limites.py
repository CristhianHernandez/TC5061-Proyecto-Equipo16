"""Casos límite de las utilidades del pipeline de entrenamiento (mibici.modeling.train y comun).

Cubre entradas vacías, parámetros inválidos y esquemas incompatibles: el pipeline debe fallar con
un error claro **antes** de registrar nada en MLflow, o rechazar la entrada en el modelo servido.
"""

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import numpy as np
import pandas as pd
import pytest

from mibici.modeling import comun, train

pytestmark = pytest.mark.limite


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("63", 63),
        ("  63 ", 63),
        ("0", 0),
        ("-1", -1),
        ("0.05", 0.05),
        ("1e-3", 0.001),
        ("none", None),
        ("NULL", None),
        ("balanced", "balanced"),
    ],
)
def test_convertir_tipos_de_los_parametros_de_mlflow_run(texto, esperado):
    assert train.convertir(texto) == esperado
    assert type(train.convertir(texto)) is type(esperado)


@pytest.mark.parametrize("vacio", ["", "   ", '""', "''"])
def test_parametros_vacios_se_omiten_y_se_toman_de_config(vacio):
    """MLflow pasa '' (o comillas literales en Windows) cuando el parámetro no se indicó."""
    assert train.hiperparametros_dados(learning_rate=vacio, k=vacio) == {}
    assert train.hiperparametros_dados(learning_rate="0.1", k=vacio) == {"learning_rate": 0.1}


def test_estabilidad_mensual_con_un_mes_o_sin_datos_es_cero():
    mae = lambda real, pred: float(np.mean(np.abs(np.asarray(real) - np.asarray(pred))))
    un_mes = pd.Series(pd.to_datetime(["2026-07-01", "2026-07-15"]))
    assert comun.estabilidad_mensual(un_mes, [1, 2], [1, 3], mae) == 0.0
    vacias = pd.Series(pd.to_datetime([]))
    assert comun.estabilidad_mensual(vacias, [], [], mae) == 0.0
    dos_meses = pd.Series(pd.to_datetime(["2026-07-01", "2026-08-01"]))
    # errores de 0 y 2 por mes: desviación estándar poblacional = 1
    assert comun.estabilidad_mensual(dos_meses, [1, 1], [1, 3], mae) == pytest.approx(1.0)


def test_nombre_de_corrida_es_estable_sin_importar_el_orden():
    assert comun.nombre_corrida("p1", "hgb", {}) == "p1-hgb"
    a = comun.nombre_corrida("p1", "hgb", {"max_iter": 100, "learning_rate": 0.05})
    b = comun.nombre_corrida("p1", "hgb", {"learning_rate": 0.05, "max_iter": 100})
    assert a == b == "p1-hgb-learning_rate0.05-max_iter100"


def test_pregunta_ignora_mayusculas_y_rechaza_desconocidas():
    assert comun.modulo_pregunta("P2") is comun.modulo_pregunta("p2")
    for invalida in ("p4", "", "demanda"):
        with pytest.raises(ValueError, match="Pregunta"):
            comun.modulo_pregunta(invalida)


def test_hiperparametros_desconocidos_o_etapa_incoherente_se_rechazan():
    receta = comun.modulo_pregunta("p1").MODELOS["hgb"]
    with pytest.raises(ValueError, match="no acepta"):
        comun.resolver_hiperparametros("p1", "hgb", receta, {"n_clusters": 3})
    base = comun.modulo_pregunta("p1").MODELOS["naive"]
    with pytest.raises(ValueError, match="no corresponde"):
        comun.resolver_etapa(base, "candidate", False)
    assert comun.resolver_etapa(base, "analysis", False) == "analysis"


def test_cli_train_rechaza_search_invalido_sin_entrenar(monkeypatch):
    from typer.testing import CliRunner

    llamadas = []
    monkeypatch.setattr(train, "ejecutar", lambda *a, **k: llamadas.append(a))
    resultado = CliRunner().invoke(train.app, ["--question", "p1", "--search", "quizas"])
    assert resultado.exit_code != 0
    assert llamadas == []


def test_entrenamiento_vacio_falla_sin_registrar_corridas(conjuntos, mlflow_aislado):
    """Con un conjunto de entrenamiento vacío el error ocurre antes de abrir la corrida."""
    vacios = {**conjuntos["p1"], "train": conjuntos["p1"]["train"].iloc[0:0]}
    with pytest.raises(ValueError):
        comun.ejecutar("p1", "hgb", None, {"max_iter": 10}, False, vacios)
    experimento = MlflowClient().get_experiment_by_name("mibici-equipo16")
    corridas = (
        [] if experimento is None else MlflowClient().search_runs([experimento.experiment_id])
    )
    assert corridas == []


def test_modelo_servido_rechaza_esquema_incompatible(conjuntos, mlflow_aislado):
    """La firma registrada hace que el modelo cargado rechace entradas sin una columna esperada."""
    run_id = comun.ejecutar("p1", "hgb", None, {"max_iter": 10}, False, conjuntos["p1"])
    modelo = mlflow.pyfunc.load_model(mlflow.get_run(run_id).data.tags["model_uri"])
    columnas = modelo.metadata.get_input_schema().input_names()
    prueba = conjuntos["p1"]["test"][columnas]

    assert len(modelo.predict(prueba.head(3))) == 3
    with pytest.raises(MlflowException):
        modelo.predict(prueba.drop(columns=columnas[0]).head(3))
