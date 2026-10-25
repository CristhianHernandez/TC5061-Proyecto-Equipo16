"""P3: variantes de variables, GMM, búsqueda de k, regla de selección y tablas comparativas."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
import pytest

from mibici.config import MODELOS_REGISTRADOS
from mibici.modeling import comparar, comun, registro
from mibici.modeling import p3_clusters as p3


def cerrar(resultado):
    for figura in resultado.figuras.values():
        plt.close(figura)


def test_tasas_diarias_hacen_comparables_periodos_de_distinta_duracion(sinteticos):
    perfiles = sinteticos[3]
    largo = perfiles.assign(**{c: perfiles[c] * 6 for c in p3.VOLUMENES})
    np.testing.assert_allclose(
        p3.matriz_perfiles(p3.a_tasas_diarias(largo, "2025-01-01", "2025-07-01")).to_numpy(),
        p3.matriz_perfiles(p3.a_tasas_diarias(perfiles, "2025-01-01", "2025-01-31")).to_numpy(),
        rtol=0.05,
    )


def test_v2_quita_el_volumen_y_conserva_el_patron(sinteticos):
    perfiles = sinteticos[3]
    v1, v2 = p3.matriz_perfiles(perfiles), p3.matriz_patron(perfiles)
    assert not any(c.startswith("log_") for c in v2)
    assert set(v1) - set(v2) == {"log_total_salidas", "log_total_llegadas", "log_volumen_total"}


@pytest.mark.parametrize(
    ("receta", "hiperparametros"),
    [
        ("kmeans", {"k": 3, "features": "v2"}),
        ("gmm", {"n_components": 3, "covariance_type": "diag"}),
        ("gmm", {"n_components": 3, "covariance_type": "spherical"}),
    ],
)
def test_candidatas_servibles_con_predict_estabilidad_y_silueta(
    conjuntos, receta, hiperparametros
):
    resultado = p3.MODELOS[receta].entrenar(conjuntos["p3"], hiperparametros)
    perfiles = conjuntos["p3"]["train"]
    etiquetas = resultado.modelo.predict(p3.entradas(perfiles))
    assert hasattr(resultado.modelo, "predict")
    assert 0 <= resultado.metricas["train_ari_semillas"] <= 1
    # Las métricas se miden en el espacio de referencia v1, comparable con el baseline
    _, referencia = p3.ajustar_escala(perfiles)
    assert resultado.metricas["train_silueta"] == pytest.approx(
        p3.metricas_clusters(referencia, etiquetas)["silueta"]
    )
    assert {"mapa", "perfiles", "silueta"} <= set(resultado.figuras)
    cerrar(resultado)


def test_hiperparametros_invalidos_de_p3(conjuntos):
    with pytest.raises(ValueError, match="features"):
        p3.entrenar_kmeans(conjuntos["p3"], {"k": 3, "features": "v9"})
    with pytest.raises(ValueError, match="covariance_type"):
        p3.entrenar_gmm(conjuntos["p3"], {"n_components": 3, "covariance_type": "otra"})


def test_kmeans_v2_se_ajusta_solo_con_train(conjuntos):
    original = p3.entrenar_kmeans(conjuntos["p3"], {"k": 3, "features": "v2"})
    alterados = dict(conjuntos["p3"])
    for periodo in ("val", "test"):
        cambio = alterados[periodo].copy()
        numericas = cambio.columns.drop("estacion_id")
        cambio[numericas] = cambio[numericas] * 7 + 3
        alterados[periodo] = cambio
    nuevo = p3.entrenar_kmeans(alterados, {"k": 3, "features": "v2"})
    np.testing.assert_array_equal(
        original.modelo.predict(original.ejemplo), nuevo.modelo.predict(original.ejemplo)
    )
    cerrar(original)
    cerrar(nuevo)


def test_busqueda_de_k_como_nested_runs(mlflow_sesion, conjuntos, monkeypatch):
    monkeypatch.setitem(comun.REJILLAS["p3"], "kmeans", {"k": [2, 3]})
    padre_id = comun.ejecutar("p3", "kmeans", None, {}, True, conjuntos["p3"])
    cliente = MlflowClient()
    padre = cliente.get_run(padre_id)
    hijos = cliente.search_runs(
        [padre.info.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{padre_id}'"
    )
    assert padre.data.tags["stage_of_work"] == "search"
    assert len(hijos) == 2
    for hijo in hijos:
        assert "train_ari_semillas" in hijo.data.metrics
        assert not any(k.startswith("test_") for k in hijo.data.metrics)
    mejor = max(h.data.metrics["val_silueta"] for h in hijos)
    assert padre.data.metrics["val_silueta"] == pytest.approx(mejor)
    assert "plots/silueta.png" in {a.path for a in cliente.list_artifacts(padre_id, "plots")}


def test_regla_p3_exige_estabilidad_y_candidata(monkeypatch):
    def corrida(nombre, silueta, ari, etapa="candidate"):
        return SimpleNamespace(
            info=SimpleNamespace(run_id=nombre),
            data=SimpleNamespace(
                metrics={"val_silueta": silueta, "train_ari_semillas": ari},
                tags={"stage_of_work": etapa},
            ),
        )

    # Ordenadas por silueta de validación, como las devuelve search_runs
    inestable = corrida("inestable", 0.30, 0.55)
    busqueda = corrida("busqueda", 0.25, 0.95, "search")
    estable = corrida("estable", 0.20, 0.92)
    cliente = SimpleNamespace(
        get_experiment_by_name=lambda _: SimpleNamespace(experiment_id="1"),
        search_runs=lambda *_args, **_kwargs: [inestable, busqueda, estable],
    )
    monkeypatch.setattr(registro, "MlflowClient", lambda: cliente)
    assert registro.mejor_corrida("p3").info.run_id == "estable"
    cliente.search_runs = lambda *_args, **_kwargs: [inestable]
    with pytest.raises(ValueError, match="ARI"):
        registro.mejor_corrida("p3")


@pytest.fixture(scope="module")
def corridas_p3(mlflow_sesion, conjuntos):
    """Baseline, dos candidatas servibles y Ward de P3."""
    ids = {
        "municipio": comun.ejecutar("p3", None, None, {}, False, conjuntos["p3"]),
        "kmeans": comun.ejecutar("p3", "kmeans", None, {"k": 3}, False, conjuntos["p3"]),
        "gmm": comun.ejecutar("p3", "gmm", None, {"n_components": 3}, False, conjuntos["p3"]),
        "ward": comun.ejecutar("p3", "ward", None, {"k": 3}, False, conjuntos["p3"]),
    }
    return ids


def test_champion_de_p3_tiene_predict(corridas_p3, conjuntos):
    with pytest.raises(ValueError, match="registrable"):
        registro.promover("p3", corridas_p3["ward"])
    resultado = registro.promover("p3", corridas_p3["kmeans"])
    modelo = mlflow.pyfunc.load_model(resultado["uri"])
    assert resultado["uri"] == f"models:/{MODELOS_REGISTRADOS['p3']}@champion"
    columnas = modelo.metadata.get_input_schema().input_names()
    grupos = modelo.predict(conjuntos["p3"]["test"][columnas].head(4))
    assert len(grupos) == 4


def test_compare_escribe_tablas_con_baseline_candidatas_y_mejora(
    corridas_p3, mlflow_sesion, monkeypatch
):
    rutas = {q: mlflow_sesion["ruta"] / f"tabla_comparativa_{q}.csv" for q in ("p1", "p2", "p3")}
    monkeypatch.setattr(comparar, "TABLAS_COMPARATIVAS", rutas)
    monkeypatch.setattr(comparar, "MLFLOW_TRACKING_URI", mlflow_sesion["uri"])
    comparar.comparar(MlflowClient().get_run(corridas_p3["kmeans"]).data.tags["version_datos"])
    tabla = pd.read_csv(rutas["p3"])
    assert rutas["p1"].exists() and rutas["p2"].exists()
    assert tabla["stage_of_work"].iloc[0] == "baseline"
    assert {"municipio", "kmeans", "gmm", "ward"} <= set(tabla["modelo"])
    assert tabla["corrida"].is_unique
    assert not (tabla["stage_of_work"] == "analysis").any()
    for columna in ("val_silueta", "test_silueta", "train_ari_semillas", "train_duracion_s"):
        assert columna in tabla
    base = tabla.loc[tabla["modelo"] == "municipio"].iloc[0]
    assert base["test_mejora_silueta_abs"] == pytest.approx(0)
    kmeans = tabla.loc[tabla["run_id"] == corridas_p3["kmeans"]].iloc[0]
    assert kmeans["alias"].startswith("champion")
    assert "k=3" in kmeans["hiperparametros"]
