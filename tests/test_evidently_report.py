"""Reporte de drift de Evidently: detecta drift intencional, no inventa drift y deja evidencia.

Usa conjuntos sintéticos con el contrato de `mibici.monitoring.datasets`; no lee datos reales ni
requiere MLflow.
"""

import json

import numpy as np
import pandas as pd
import pytest

from mibici.config import UMBRAL_DRIFT
from mibici.monitoring import evidently_report as rep
from mibici.monitoring.datasets import COLUMNAS, OBJETIVO, VARIABLES


def conjunto(fechas, estaciones=range(1, 81), semilla=16, **cambios) -> pd.DataFrame:
    """Filas estación × día con la distribución base; `cambios` desplaza columnas (+= valor)."""
    azar = np.random.default_rng(semilla)
    fechas = pd.to_datetime(pd.Series(fechas))
    filas = pd.MultiIndex.from_product([list(estaciones), fechas], names=["estacion_id", "fecha"])
    datos = filas.to_frame(index=False)
    n = len(datos)
    fecha = datos["fecha"]
    datos["dia_semana"] = fecha.dt.dayofweek
    datos["mes"] = fecha.dt.month
    datos["es_fin_semana"] = (datos["dia_semana"] >= 5).astype(int)
    datos["dia_anio"] = fecha.dt.dayofyear
    datos["es_festivo"] = (azar.random(n) < 0.03).astype(int)
    for columna in ["demanda_lag_1", "demanda_lag_7", "demanda_media_7"]:
        datos[columna] = azar.gamma(2.0, 15.0, n)
    # Variables de estación fijas por estación, como en el catálogo
    ids = datos["estacion_id"].to_numpy()
    datos["capacidad"] = (15 + (ids % 3) * 4).astype(float)
    datos["latitud"] = 20.6 + ids * 0.005
    datos["longitud"] = -103.4 + ids * 0.005
    datos[OBJETIVO] = azar.gamma(2.0, 15.0, n)
    for columna, desplazamiento in cambios.items():
        datos[columna] = datos[columna] + desplazamiento
    return datos[COLUMNAS]


OCTUBRE = pd.date_range("2025-10-01", "2025-10-31")  # mismo calendario en referencia y actual


@pytest.fixture(scope="module")
def sin_drift():
    referencia = conjunto(OCTUBRE, semilla=1)
    actual = conjunto(OCTUBRE, semilla=2)
    snapshot = rep.ejecutar_evidently(referencia, actual)
    return referencia, actual, snapshot


@pytest.fixture(scope="module")
def con_drift():
    referencia = conjunto(OCTUBRE, semilla=1)
    actual = conjunto(OCTUBRE, semilla=2, demanda_lag_1=20.0, capacidad=6.0)
    snapshot = rep.ejecutar_evidently(referencia, actual)
    return referencia, actual, snapshot


def conjuntos_sinteticos(desplazamiento=0.0):
    """Referencia, reciente y actual con fechas dentro de las ventanas del proyecto."""
    return {
        "referencia": conjunto(
            pd.date_range("2024-10-01", "2026-03-31", freq="5D"), estaciones=range(1, 11)
        ),
        "reciente": conjunto(
            pd.date_range("2026-04-01", "2026-09-30", freq="3D"),
            estaciones=range(1, 13),
            semilla=3,
        ),
        "actual": conjunto(
            pd.date_range("2026-10-01", "2026-10-31"),
            estaciones=range(1, 13),
            semilla=4,
            demanda=desplazamiento,
            demanda_lag_1=desplazamiento,
        ),
    }


# --- detección ---------------------------------------------------------------------------------


def test_sin_drift_no_marca_variables(sin_drift):
    referencia, actual, snapshot = sin_drift
    tabla = rep.tabla_drift(snapshot, referencia, actual, "prueba")
    resumen = rep.resumen_comparacion(tabla, snapshot, len(referencia), len(actual))
    assert resumen["variables_con_drift"] == 0
    assert not resumen["drift_del_conjunto"]
    assert not resumen["objetivo"]["drift"]
    assert (tabla["puntaje"] < UMBRAL_DRIFT).all()


def test_drift_intencional_se_detecta_solo_en_las_columnas_alteradas(con_drift):
    referencia, actual, snapshot = con_drift
    tabla = rep.tabla_drift(snapshot, referencia, actual, "prueba").set_index("columna")
    assert set(tabla.index[tabla["drift"]]) == {"demanda_lag_1", "capacidad"}
    assert tabla.loc[VARIABLES, "drift"].sum() == 2
    assert tabla.loc["demanda_lag_1", "cambio_media_pct"] > 50
    # Toda capacidad actual está fuera del rango visto en la referencia (15 a 23 -> 21 a 29)
    assert tabla.loc["capacidad", "fuera_de_referencia_pct"] > 0


def test_el_conteo_de_la_tabla_coincide_con_evidently(con_drift):
    referencia, actual, snapshot = con_drift
    tabla = rep.tabla_drift(snapshot, referencia, actual, "prueba")
    resumen = rep.resumen_comparacion(tabla, snapshot, len(referencia), len(actual))
    assert resumen["variables_con_drift"] == rep.columnas_con_drift_evidently(snapshot)["count"]
    assert resumen["variables_con_drift_por_puntaje"][0] in {"demanda_lag_1", "capacidad"}
    assert resumen["drift_por_grupo"]["estación"] == ["capacidad"]


def test_identificadores_fuera_y_objetivo_aparte():
    referencia = conjunto(OCTUBRE, semilla=1)
    actual = conjunto(OCTUBRE, semilla=2, **{OBJETIVO: 30.0})
    snapshot = rep.ejecutar_evidently(referencia, actual)
    assert set(rep.puntajes_drift(snapshot)) == {*VARIABLES, OBJETIVO}
    tabla = rep.tabla_drift(snapshot, referencia, actual, "prueba")
    resumen = rep.resumen_comparacion(tabla, snapshot, len(referencia), len(actual))
    # El objetivo cambia pero no cuenta en la proporción de variables de entrada con drift
    assert resumen["objetivo"]["drift"]
    assert resumen["variables_con_drift"] == 0
    assert resumen["variables_monitoreadas"] == len(VARIABLES)


def test_metodos_fijos_aun_con_pocas_filas():
    """Con < 1,000 filas Evidently cambiaría a p-values; aquí se mantienen las distancias."""
    referencia = conjunto(OCTUBRE[:10], estaciones=range(1, 21), semilla=1)
    actual = conjunto(OCTUBRE[:10], estaciones=range(1, 21), semilla=2)
    assert len(referencia) < 1000
    snapshot = rep.ejecutar_evidently(referencia, actual)
    metodos = {c: d["metodo"] for c, d in rep.puntajes_drift(snapshot).items()}
    assert {metodos[c] for c in rep.CATEGORICAS} == {rep.METODO_DRIFT_CATEGORICO}
    assert {metodos[c] for c in [*rep.NUMERICAS, OBJETIVO]} == {rep.METODO_DRIFT_NUMERICO}


# --- contrato y calidad ------------------------------------------------------------------------


def test_esquema_incompleto_o_con_tipos_invalidos_falla():
    datos = conjunto(OCTUBRE[:3])
    with pytest.raises(ValueError, match="Faltan columnas"):
        rep.validar_esquema(datos.drop(columns="capacidad"))
    with pytest.raises(TypeError, match="no numérico"):
        rep.validar_esquema(datos.astype({"mes": str}))
    with pytest.raises(ValueError, match="vacío"):
        rep.validar_esquema(datos.iloc[:0])


def test_fuera_de_referencia_por_rango_y_por_categoria():
    referencia = pd.Series([1.0, 2.0, 3.0])
    assert rep.fuera_de_referencia(referencia, pd.Series([0.0, 2.0, 4.0, np.nan]), False) == 2 / 3
    assert rep.fuera_de_referencia(pd.Series([1, 2]), pd.Series([1, 3, 3, 2]), True) == 0.5
    assert rep.fuera_de_referencia(referencia, pd.Series([np.nan]), False) == 0.0


def test_calidad_detecta_duplicados_nulos_constantes_y_tipos():
    referencia = conjunto(OCTUBRE[:5])
    # Dos filas repetidas: la primera idéntica y la segunda con la misma clave y otra demanda
    repetidas = referencia.iloc[:2].copy()
    repetidas.loc[repetidas.index[1], OBJETIVO] += 1
    datos = pd.concat([referencia, repetidas], ignore_index=True)
    datos.loc[5, "demanda_lag_7"] = np.nan
    datos.loc[6, OBJETIVO] = -1
    datos["capacidad"] = datos["capacidad"].astype("float32")
    resultado = rep.calidad(datos.drop(columns="longitud"), referencia)
    assert resultado["filas_duplicadas"] == 1
    assert resultado["claves_duplicadas"] == 2
    assert resultado["nulos"] == {"demanda_lag_7": 1}
    assert resultado["columnas_faltantes"] == ["longitud"]
    assert "mes" in resultado["variables_constantes"]  # una sola semana de octubre
    assert resultado["demanda_negativa"] == 1
    assert resultado["tipos_distintos_a_referencia"] == {"capacidad": ["float64", "float32"]}


def test_estaciones_nuevas_y_comparacion_solo_con_estaciones_conocidas():
    conjuntos = conjuntos_sinteticos()
    nuevas = rep.estaciones_nuevas(conjuntos["referencia"], conjuntos["actual"])
    assert nuevas["estaciones_conocidas"] == 10
    assert nuevas["estaciones_nuevas"] == 2
    assert nuevas["filas_estaciones_nuevas_pct"] == pytest.approx(100 * 2 / 12, abs=0.01)
    _, actual = rep.preparar_comparacion(conjuntos, "referencia_vs_actual_estaciones_conocidas")
    assert set(actual["estacion_id"]) == set(range(1, 11))


def test_demanda_mismo_mes_compara_contra_anios_previos():
    conjuntos = conjuntos_sinteticos(desplazamiento=30.0)
    resultado = rep.demanda_mismo_mes(conjuntos)
    assert resultado["meses"] == [10]
    assert set(resultado["demanda_media_por_anio"]) == {2024, 2025, 2026}
    assert resultado["cambio_vs_anios_previos_pct"] > 50


def test_variables_a_graficar_omite_calendario_y_objetivo():
    tabla = pd.DataFrame(
        {
            "comparacion": rep.COMPARACION_PRINCIPAL,
            "columna": ["dia_anio", OBJETIVO, "demanda_lag_1", "demanda_lag_7", "demanda_media_7"],
            "grupo": ["calendario", "objetivo", *["historial de demanda"] * 3],
            "tipo": "numerica",
            "puntaje": [1.0, 0.9, 0.3, 0.2, 0.25],
            "drift": True,
        }
    )
    assert rep.variables_a_graficar(tabla) == ["demanda_lag_1", "demanda_media_7"]


# --- evidencia ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def evidencia(tmp_path_factory):
    salida = tmp_path_factory.mktemp("drift")
    resumen = rep.generar(conjuntos_sinteticos(desplazamiento=25.0), salida)
    return salida, resumen


def test_generar_escribe_html_tablas_y_graficas(evidencia):
    salida, resumen = evidencia
    for nombre, (_, _, con_html, _) in rep.COMPARACIONES.items():
        assert (salida / f"reporte_{nombre}.html").exists() == con_html
    html = (salida / f"reporte_{rep.COMPARACION_PRINCIPAL}.html").read_text(encoding="utf-8")
    assert "demanda_lag_1" in html
    for archivo in [
        "drift_por_variable.csv",
        "resumen_drift.json",
        "grafica_drift_por_variable.png",
        "grafica_distribuciones.png",
        "grafica_demanda_diaria.png",
    ]:
        assert (salida / archivo).stat().st_size > 0
    tabla = pd.read_csv(salida / "drift_por_variable.csv")
    assert list(tabla.columns) == rep.COLUMNAS_TABLA
    assert len(tabla) == len(rep.COMPARACIONES) * (len(VARIABLES) + 1)
    assert set(resumen["comparaciones"]) == set(rep.COMPARACIONES)
    principal = resumen["comparaciones"][rep.COMPARACION_PRINCIPAL]
    assert "demanda_lag_1" in principal["variables_con_drift_por_puntaje"]
    assert principal["objetivo"]["drift"]
    assert resumen["configuracion"]["excluidas"] == ["estacion_id", "fecha"]


def test_generar_es_reproducible(evidencia, tmp_path):
    salida, _ = evidencia
    rep.generar(conjuntos_sinteticos(desplazamiento=25.0), tmp_path, html=False)
    for archivo in ["drift_por_variable.csv", "resumen_drift.json"]:
        assert (tmp_path / archivo).read_bytes() == (salida / archivo).read_bytes()
    assert json.loads((tmp_path / "resumen_drift.json").read_text(encoding="utf-8"))
    assert not list(tmp_path.glob("*.html"))
