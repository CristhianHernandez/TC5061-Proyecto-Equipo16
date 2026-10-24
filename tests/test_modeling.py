"""Contratos de entrenamiento y registro de P1, P2 y P3 con datos sintéticos y MLflow temporal."""

import matplotlib.pyplot as plt
import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import ParameterGrid

from mibici.config import (
    BASELINES,
    CLASES_RIESGO,
    METRICA_PRINCIPAL,
    REJILLA_P1,
)
from mibici.modeling import comun
from mibici.modeling import p1_demanda as p1
from mibici.modeling import p2_riesgo as p2
from mibici.modeling import p3_clusters as p3

# Corridas que se ejecutan una vez para toda la sesión: baselines primero, luego candidatas
CORRIDAS = (
    ("p1", "naive", {}),
    ("p1", "promedio", {}),
    ("p1", "hgb", {"max_iter": 20}),
    ("p1", "rf", {"n_estimators": 10, "max_depth": 4}),
    ("p1", "poisson", {"max_iter": 100}),
    ("p2", "persistencia", {}),
    ("p2", "dummy", {}),
    ("p2", "logistica", {}),
    ("p2", "hgb", {"max_iter": 20}),
    ("p3", "municipio", {}),
    ("p3", "kmeans", {"k": 3}),
    ("p3", "ward", {"k": 3}),
)
SERVIBLES = {
    ("p1", "hgb"),
    ("p1", "rf"),
    ("p1", "poisson"),
    ("p2", "dummy"),
    ("p2", "logistica"),
    ("p2", "hgb"),
    ("p3", "kmeans"),
}


@pytest.fixture(scope="session")
def corridas(mlflow_sesion, conjuntos):
    """Ejecuta cada receta como lo hace `mlflow run . -e train` y devuelve sus corridas."""
    resultado = {}
    for question, model, hiperparametros in CORRIDAS:
        run_id = comun.ejecutar(question, model, None, hiperparametros, False, conjuntos[question])
        resultado[(question, model)] = mlflow.get_run(run_id)
    return resultado


def artefactos(run_id, carpeta=""):
    """Rutas de artefactos de una carpeta de la corrida (recorre un nivel de subcarpetas)."""
    cliente = MlflowClient()
    rutas = []
    for elemento in cliente.list_artifacts(run_id, carpeta or None):
        if elemento.is_dir:
            rutas += [a.path for a in cliente.list_artifacts(run_id, elemento.path)]
        else:
            rutas.append(elemento.path)
    return set(rutas)


def test_cada_corrida_lleva_etiquetas_parametros_y_metricas(corridas):
    cliente = MlflowClient()
    for (question, model), run in corridas.items():
        assert run.info.status == "FINISHED"
        assert cliente.get_experiment(run.info.experiment_id).name == "mibici-equipo16"
        etiquetas = run.data.tags
        assert etiquetas["equipo"] == "16"
        assert etiquetas["pregunta"] == question.upper()
        assert etiquetas["question"] == question
        assert etiquetas["git_commit"]
        assert etiquetas["version_datos"] == run.data.params["version_datos"]
        esperado = (
            "baseline"
            if comun.modulo_pregunta(question).MODELOS[model].baseline
            else ("candidate")
        )
        assert etiquetas["stage_of_work"] == esperado
        assert etiquetas["model_family"]
        assert etiquetas["mlflow.runName"].startswith(f"{question}-{model}")
        for clave in (
            "fecha_inicio",
            "inicio_validacion",
            "inicio_prueba",
            "fecha_fin",
            "semilla",
        ):
            assert clave in run.data.params
        assert all(k.startswith(("train_", "val_", "test_")) for k in run.data.metrics)
        principal = METRICA_PRINCIPAL[question]
        assert f"val_{principal}" in run.data.metrics
        assert f"test_{principal}" in run.data.metrics
        assert run.data.metrics["train_duracion_s"] >= 0


def test_artefactos_organizados_en_subcarpetas(corridas):
    for (question, model), run in corridas.items():
        rutas = artefactos(run.info.run_id)
        assert "data/manifiesto_datos.json" in rutas
        assert "config/config_efectiva.json" in rutas
        assert {"config/MLproject", "config/python_env.yaml", "config/requirements.txt"} <= rutas
        assert "metrics/metricas.json" in rutas
        assert any(r.startswith("plots/") and r.endswith(".png") for r in rutas)
        if question == "p1" and run.data.tags["stage_of_work"] == "candidate":
            assert {
                "plots/real_vs_predicho.png",
                "plots/residuales.png",
                "plots/error_por_estacion.png",
            } <= rutas
        entradas = run.inputs.dataset_inputs
        assert [i.dataset.name for i in entradas] == [f"{question}-entrenamiento"]
        assert [t.value for t in entradas[0].tags] == ["training"]
        registrable = run.data.tags["modelo_registrable"]
        if (question, model) in SERVIBLES:
            assert registrable == "true"
            assert {"model/MLmodel", "model/input_example.json"} <= rutas
        else:
            assert registrable == "false"
            assert not any(r.startswith("model/") for r in rutas)


def test_modelo_con_firma_ejemplo_y_carga(corridas, conjuntos):
    for clave in SERVIBLES:
        run = corridas[clave]
        info = mlflow.models.get_model_info(run.data.tags["model_uri"])
        assert info.signature is not None
        assert info.saved_input_example_info is not None
        modelo = mlflow.pyfunc.load_model(run.data.tags["model_uri"])
        columnas = modelo.metadata.get_input_schema().input_names()
        prueba = conjuntos[clave[0]]["test"]
        assert len(modelo.predict(prueba[columnas].head(3))) == 3
        assert run.data.metrics["test_latencia_ms_por_1000"] > 0


def test_configuracion_efectiva_sin_rutas_ni_credenciales(corridas):
    run = corridas[("p1", "hgb")]
    ruta = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path="config/config_efectiva.json"
    )
    configuracion = pd.read_json(ruta, typ="series")
    assert configuracion["model"] == "hgb"
    assert configuracion["hiperparametros"]["max_iter"] == 20
    assert configuracion["hiperparametros"]["learning_rate"] == 0.1  # tomado de config.py
    assert "tracking_uri" not in configuracion.index


def test_mejora_contra_baseline_de_referencia(corridas):
    for question in ("p1", "p2", "p3"):
        base = corridas[(question, BASELINES[question])]
        principal = METRICA_PRINCIPAL[question]
        for (q, model), run in corridas.items():
            if q != question or run.data.tags["stage_of_work"] == "baseline":
                continue
            assert run.data.tags["baseline_run_id"] == base.info.run_id
            b = base.data.metrics[f"test_{principal}"]
            v = run.data.metrics[f"test_{principal}"]
            esperada = 100 * ((b - v) if question == "p1" else (v - b)) / abs(b)
            assert run.data.metrics[f"test_mejora_{principal}_pct"] == pytest.approx(esperada)


def test_busqueda_como_nested_runs(corridas, conjuntos):
    padre_id = comun.ejecutar("p1", "hgb", None, {"max_iter": 20}, True, conjuntos["p1"])
    cliente = MlflowClient()
    padre = cliente.get_run(padre_id)
    hijos = cliente.search_runs(
        [padre.info.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{padre_id}'"
    )
    assert padre.data.tags["stage_of_work"] == "search"
    assert padre.data.tags["search_role"] == "parent"
    assert len(hijos) == len(ParameterGrid(REJILLA_P1))
    for hijo in hijos:
        assert hijo.data.tags["stage_of_work"] == "search"
        assert hijo.data.tags["question"] == "p1"
        assert hijo.data.tags["git_commit"] == padre.data.tags["git_commit"]
        assert not any(k.startswith("test_") for k in hijo.data.metrics)
    minimo = min(h.data.metrics["val_mae"] for h in hijos)
    assert padre.data.metrics["val_mae"] == pytest.approx(minimo)
    # La combinación elegida por la padre es una hija con el mínimo de validación (puede haber empates)
    elegida = {clave: padre.data.params[f"mejor_{clave}"] for clave in REJILLA_P1}
    assert any(
        {clave: h.data.params[f"hp_{clave}"] for clave in REJILLA_P1} == elegida
        and h.data.metrics["val_mae"] == pytest.approx(minimo)
        for h in hijos
    )
    rutas = artefactos(padre_id)
    assert {"metrics/busqueda.csv", "model/MLmodel", "data/manifiesto_datos.json"} <= rutas
    assert "test_mae" in padre.data.metrics


def test_busqueda_elige_sin_ver_prueba(mlflow_sesion, conjuntos):
    alterados = {k: v.copy() for k, v in conjuntos["p1"].items()}
    alterados["test"]["demanda"] = 999
    a = MlflowClient().get_run(
        comun.ejecutar("p1", "hgb", None, {"max_iter": 20}, True, conjuntos["p1"])
    )
    b = MlflowClient().get_run(
        comun.ejecutar("p1", "hgb", None, {"max_iter": 20}, True, alterados)
    )
    elegidos = {k: v for k, v in a.data.params.items() if k.startswith("mejor_")}
    assert elegidos == {k: v for k, v in b.data.params.items() if k.startswith("mejor_")}
    assert a.data.metrics["val_mae"] == b.data.metrics["val_mae"]


def test_misma_semilla_metricas_identicas(mlflow_sesion, conjuntos):
    corridas = [
        MlflowClient().get_run(
            comun.ejecutar("p2", "hgb", None, {"max_iter": 20}, False, conjuntos["p2"])
        )
        for _ in range(2)
    ]
    variables = [
        {k: v for k, v in r.data.metrics.items() if "duracion" not in k and "latencia" not in k}
        for r in corridas
    ]
    assert variables[0] == pytest.approx(variables[1], abs=1e-12)


def test_validaciones_de_modelo_etapa_e_hiperparametros(mlflow_sesion, conjuntos):
    with pytest.raises(ValueError, match="no acepta"):
        comun.ejecutar("p1", "hgb", None, {"C": 1.0}, False, conjuntos["p1"])
    with pytest.raises(ValueError, match="no corresponde"):
        comun.ejecutar("p1", "naive", "candidate", {}, False, conjuntos["p1"])
    with pytest.raises(ValueError, match="rejilla"):
        comun.ejecutar("p3", "ward", None, {}, True, conjuntos["p3"])
    with pytest.raises(ValueError, match="no tiene el modelo"):
        comun.ejecutar("p2", "svm", None, {}, False, conjuntos["p2"])
    with pytest.raises(ValueError, match="Pregunta"):
        comun.ejecutar("p4", None, None, {}, False, conjuntos["p1"])


def test_naive_manual_respaldo_y_promedio_solo_train():
    entrenamiento = pd.DataFrame(
        {"estacion_id": [1, 1, 2], "dia_semana": [0, 0, 1], "demanda": [2, 6, 10]}
    )
    futuro = pd.DataFrame(
        {
            "estacion_id": [1, 1, 99],
            "dia_semana": [0, 0, 0],
            "demanda_lag_7": [7, np.nan, np.nan],
            "demanda": [100] * 3,
        }
    )
    assert p1.predecir_naive(entrenamiento, futuro) == pytest.approx([7, 4, 6])
    assert p1.predecir_promedio(entrenamiento, futuro) == pytest.approx([4, 4, 6])
    cambiado = futuro.assign(demanda=-999)
    np.testing.assert_array_equal(
        p1.predecir_naive(entrenamiento, futuro), p1.predecir_naive(entrenamiento, cambiado)
    )


def test_persistencia_manual_y_ap_indicador():
    datos = pd.DataFrame({"flujo_lag_56": [10, -10, 9, -9, np.nan], "capacidad": [20] * 5})
    pred = p2.predecir_persistencia(datos)
    assert list(pred) == ["sin_bicicletas", "sin_anclajes", "normal", "normal", "normal"]
    puntuaciones = np.column_stack([pred == c for c in CLASES_RIESGO])
    metricas = p2.metricas_riesgo(pred, pred, puntuaciones, CLASES_RIESGO)
    assert metricas["f1_macro"] == 1
    assert metricas["ap_macro_riesgo"] == 1


def test_metricas_poisson_recorta_cero():
    metricas = p1.metricas_demanda([0, 1], [0, 0])
    assert metricas["mae"] == 0.5
    assert np.isfinite(metricas["desviacion_poisson"])


def test_hgb_v2_usa_media_movil_28(conjuntos):
    resultado = p1.entrenar_hgb(
        conjuntos["p1"],
        {"features": "v2", "learning_rate": 0.1, "max_leaf_nodes": 31, "max_iter": 20},
    )
    assert "demanda_media_28" in resultado.ejemplo.columns
    assert resultado.params["features"] == "v2"
    assert set(resultado.figuras) == {"real_vs_predicho", "residuales", "error_por_estacion"}


def test_candidatos_se_ajustan_solo_con_train(conjuntos):
    """Cambiar val o test no cambia lo que aprende un modelo: se ajusta solo con train."""
    for question, model, hiperparametros in (
        ("p1", "hgb", {"learning_rate": 0.1, "max_leaf_nodes": 31, "max_iter": 20}),
        ("p1", "rf", {"n_estimators": 10, "max_depth": 4}),
        ("p1", "poisson", {"alpha": 1.0, "max_iter": 100}),
        ("p2", "logistica", {"C": 1.0}),
        ("p3", "kmeans", {"k": 3}),
    ):
        receta = comun.modulo_pregunta(question).MODELOS[model]
        original = receta.entrenar(conjuntos[question], hiperparametros)
        alterados = dict(conjuntos[question])
        for periodo in ("val", "test"):
            cambio = alterados[periodo].copy()
            numericas = cambio.select_dtypes("number").columns.drop("estacion_id")
            cambio[numericas] = cambio[numericas] * 7 + 3
            alterados[periodo] = cambio
        nuevo = receta.entrenar(alterados, hiperparametros)
        ejemplo = original.ejemplo
        np.testing.assert_array_equal(
            original.modelo.predict(ejemplo), nuevo.modelo.predict(ejemplo)
        )
        for clave, valor in original.metricas.items():
            if clave.startswith("train_"):
                assert nuevo.metricas[clave] == pytest.approx(valor)
        for resultado in (original, nuevo):
            for figura in resultado.figuras.values():
                plt.close(figura)


def test_p3_kmeans_es_pipeline_servible_y_escala_de_train(conjuntos):
    resultado = p3.entrenar_kmeans(conjuntos["p3"], {"k": 3})
    perfiles = conjuntos["p3"]["train"]
    etiquetas = resultado.modelo.predict(p3.entradas(perfiles))
    assert len(np.unique(etiquetas)) == 3
    _, matriz = p3.ajustar_escala(perfiles)
    np.testing.assert_allclose(resultado.modelo[:-1].transform(p3.entradas(perfiles)), matriz)
    assert p3.MODELOS["ward"].hiperparametros == ("k",)
    assert p3.entrenar_ward(conjuntos["p3"], {"k": 3}).modelo is None


def test_bajo_mlflow_run_no_choca_con_los_parametros_del_mlproject(
    mlflow_sesion, conjuntos, monkeypatch
):
    """`mlflow run` registra antes todos los parámetros del MLproject (vacíos si no se pasaron)."""
    cliente = MlflowClient()
    experimento = cliente.get_experiment_by_name("mibici-equipo16").experiment_id
    for question, model, dados, search in (
        ("p3", "municipio", {}, False),
        ("p3", "kmeans", {"k": 3}, False),
        ("p1", "hgb", {"max_iter": 20}, True),
    ):
        run_id = cliente.create_run(experimento).info.run_id
        pedidos = {nombre: "" for nombre in comun.parametros_mlproject()}
        pedidos.update({"question": question, "model": model, "search": str(search).lower()})
        pedidos.update({k: str(v) for k, v in dados.items()})
        for clave, valor in pedidos.items():
            cliente.log_param(run_id, clave, valor)
        monkeypatch.setenv("MLFLOW_RUN_ID", run_id)
        assert comun.ejecutar(question, model, None, dados, search, conjuntos[question]) == run_id
        params = cliente.get_run(run_id).data.params
        assert params["model"] == model  # lo pedido en el comando
        if model == "kmeans":
            assert params["k"] == "3" and params["hp_k"] == "3"
        if model == "hgb":
            assert params["learning_rate"] == ""  # no se pidió
            assert params["hp_learning_rate"]  # valor efectivo tomado de config.py
