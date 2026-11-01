"""Casos normales y límite de métricas y contratos de modelos de P1, P2 y P3.

Usa resultados calculados a mano y fixtures sintéticos compartidos. Comprueba tipos, dominios
y causalidad temporal sin datos reales, red ni registro de corridas en MLflow.
"""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mibici import features
from mibici.config import (
    CLASES_RIESGO,
    FECHA_FIN,
    FECHA_INICIO,
    INICIO_PRUEBA,
    INICIO_VALIDACION,
    MINIMO_POISSON,
)
from mibici.modeling import p1_demanda as p1
from mibici.modeling import p2_riesgo as p2
from mibici.modeling import p3_clusters as p3


def test_metricas_demanda_con_errores_manual():
    # Errores absolutos 1 y 2; Poisson: promedio de 2*(y*log(y/mu)-y+mu).
    resultado = p1.metricas_demanda([1, 4], [2, 2])
    assert resultado == pytest.approx(
        {"mae": 1.5, "rmse": np.sqrt(2.5), "desviacion_poisson": 3 * np.log(2) - 1},
        rel=0,
        abs=1e-12,
    )
    assert all(type(valor) is float for valor in resultado.values())


@pytest.mark.limite
@pytest.mark.parametrize("prediccion", [0.0, -2.0])
def test_metricas_demanda_aplica_piso_poisson_solo_a_la_desviacion(prediccion):
    resultado = p1.metricas_demanda([1], [prediccion])
    assert resultado == pytest.approx(
        {
            "mae": 1 - prediccion,
            "rmse": 1 - prediccion,
            "desviacion_poisson": 2 * (np.log(1 / MINIMO_POISSON) - 1 + MINIMO_POISSON),
        },
        rel=0,
        abs=1e-12,
    )


@pytest.mark.limite
@pytest.mark.parametrize("real,prediccion", [([], []), ([1, 2], [1])])
def test_metricas_demanda_rechaza_vacio_y_longitudes_distintas(real, prediccion):
    with pytest.raises(ValueError):
        p1.metricas_demanda(real, prediccion)


def test_metricas_riesgo_con_confusion_y_columnas_reordenadas():
    real = ["normal", "sin_anclajes", "sin_bicicletas", "sin_bicicletas"]
    prediccion = ["normal", "sin_anclajes", "sin_bicicletas", "normal"]
    clases = ["sin_bicicletas", "normal", "sin_anclajes"]
    puntuaciones = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.7, 0.1, 0.2], [0.2, 0.7, 0.1]])
    # F1: (2/3 + 1 + 2/3)/3; balanced: (1 + 1 + 1/2)/3, distinto de accuracy=3/4.
    # AP bicicletas: (1 + 2/3)/2; AP anclajes: 1.
    assert p2.metricas_riesgo(real, prediccion, puntuaciones, clases) == pytest.approx(
        {
            "f1_macro": 7 / 9,
            "balanced_accuracy": 5 / 6,
            "recall_normal": 1,
            "recall_sin_anclajes": 1,
            "recall_sin_bicicletas": 1 / 2,
            "ap_sin_anclajes": 1,
            "ap_sin_bicicletas": 5 / 6,
            "ap_macro_riesgo": 11 / 12,
        },
        rel=0,
        abs=1e-12,
    )


@pytest.mark.limite
def test_metricas_riesgo_una_clase_sin_positivos_de_riesgo():
    resultado = p2.metricas_riesgo(["normal"] * 2, ["normal"] * 2, np.ones((2, 1)), ["normal"])
    assert resultado == pytest.approx(
        {
            "f1_macro": 1 / 3,
            "balanced_accuracy": 1,
            "recall_normal": 1,
            "recall_sin_anclajes": 0,
            "recall_sin_bicicletas": 0,
            "ap_sin_anclajes": 0,
            "ap_sin_bicicletas": 0,
            "ap_macro_riesgo": 0,
        },
        rel=0,
        abs=1e-12,
    )


def test_metricas_clusters_separacion_manual():
    # Centros 1 y 11, radios medios 1; dispersión entre grupos 100 e interna 4.
    resultado = p3.metricas_clusters(np.array([[0], [2], [10], [12]]), [0, 0, 1, 1])
    assert resultado == pytest.approx(
        {"silueta": 79 / 99, "davies_bouldin": 1 / 5, "calinski_harabasz": 50},
        rel=0,
        abs=1e-12,
    )


@pytest.mark.limite
@pytest.mark.parametrize("etiquetas", [[0, 0, 0], [0, 1, 2], []])
def test_metricas_clusters_omite_indices_indefinidos(etiquetas):
    matriz = np.arange(len(etiquetas)).reshape(-1, 1)
    assert p3.metricas_clusters(matriz, etiquetas) == {}


@pytest.mark.integracion
@pytest.mark.parametrize(
    "pregunta,entrenar,parametros",
    [
        ("p1", p1.entrenar_hgb, {"max_iter": 10, "max_leaf_nodes": 7}),
        ("p2", p2.entrenar_logistica, {"C": 1.0}),
        ("p3", p3.entrenar_kmeans, {"k": 3}),
    ],
)
def test_contrato_de_predicciones_por_fila(conjuntos, pregunta, entrenar, parametros):
    resultado = entrenar(conjuntos[pregunta], parametros)
    try:
        if pregunta == "p3":
            assert resultado.modelo[-1].cluster_centers_.shape[0] == parametros["k"]
        for n in (1, 5, len(resultado.x_prueba)):
            prediccion = resultado.modelo.predict(resultado.x_prueba.iloc[:n])
            assert isinstance(prediccion, np.ndarray)
            assert prediccion.shape == (n,)
            if pregunta == "p1":
                assert np.issubdtype(prediccion.dtype, np.floating)
                assert np.isfinite(prediccion).all()
                assert (prediccion >= 0).all()
            elif pregunta == "p2":
                assert all(isinstance(valor, str) for valor in prediccion)
                assert set(prediccion) <= set(CLASES_RIESGO)
                assert set(resultado.modelo.classes_) == set(CLASES_RIESGO)
            else:
                assert np.issubdtype(prediccion.dtype, np.integer)
                assert ((0 <= prediccion) & (prediccion < parametros["k"])).all()
    finally:
        for figura in resultado.figuras.values():
            plt.close(figura)


@pytest.mark.parametrize("pregunta,columna", [("p1", "fecha"), ("p2", "fecha_franja")])
def test_conjuntos_compartidos_respetan_cortes_y_no_comparten_claves(
    conjuntos, sinteticos, pregunta, columna
):
    cortes = [pd.Timestamp(c) for c in (FECHA_INICIO, INICIO_VALIDACION, INICIO_PRUEBA)]
    cortes.append(pd.Timestamp(FECHA_FIN) + pd.Timedelta(days=1))
    claves = []
    for i, periodo in enumerate(("train", "val", "test")):
        filas = conjuntos[pregunta][periodo]
        assert not filas.empty
        assert filas[columna].ge(cortes[i]).all()
        assert filas[columna].lt(cortes[i + 1]).all()
        assert filas[columna].min() == cortes[i]
        clave = pd.MultiIndex.from_frame(filas[["estacion_id", columna]])
        assert clave.is_unique
        claves.append(set(clave))
    for izquierda, derecha in combinations(claves, 2):
        assert izquierda.isdisjoint(derecha)
    fuente = sinteticos[1 if pregunta == "p1" else 2]
    assert set.union(*claves) == set(pd.MultiIndex.from_frame(fuente[["estacion_id", columna]]))


@pytest.mark.parametrize("pregunta", ["p1", "p2"])
def test_rezagos_no_cambian_al_alterar_periodo_actual_y_futuro(pregunta):
    if pregunta == "p1":
        columna_fecha, objetivo = "fecha", "demanda"
        fechas = pd.to_datetime(["2025-01-01", "2025-01-07", "2025-01-08", "2025-01-09"])
        transformar = lambda df: features.agregar_lag_7(features.agregar_lag_demanda(df))
        rezagos = ["demanda_lag_1", "demanda_lag_7"]
    else:
        columna_fecha, objetivo = "fecha_franja", "flujo_neto"
        fechas = pd.to_datetime(
            ["2025-01-01 09:00", "2025-01-08 06:00", "2025-01-08 09:00", "2025-01-08 12:00"]
        )
        transformar = features.agregar_rezagos_flujo
        rezagos = ["flujo_lag_1", "flujo_lag_56"]
    datos = pd.DataFrame(
        {
            "estacion_id": [1] * 4 + [2] * 4,
            columna_fecha: list(fechas) * 2,
            objetivo: [3, 7, 100, 200, 30, 70, 1000, 2000],
        }
    ).iloc[::-1]
    original = transformar(datos)
    alterados = datos.copy()
    alterados.loc[alterados[columna_fecha] >= fechas[2], objetivo] = -9999
    nuevo = transformar(alterados)
    seleccion = original[columna_fecha].eq(fechas[2])
    np.testing.assert_array_equal(original.loc[seleccion, rezagos], [[7, 3], [70, 30]])
    pd.testing.assert_frame_equal(
        original.loc[original[columna_fecha] <= fechas[2], rezagos],
        nuevo.loc[nuevo[columna_fecha] <= fechas[2], rezagos],
    )
