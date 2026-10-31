"""LH3: detección real con copias de los fixtures compartidos, sin red ni datos reales."""

import json

import numpy as np
import pandas as pd
import pytest

from mibici import config, features
from mibici.validation.deepchecks import (
    evaluar_drift,
    resumir_resultados,
    validar_estaciones,
    validar_particiones,
    validar_viajes,
)

pytestmark = pytest.mark.deepchecks


def estados(resultado):
    return [c.category.name for c in resultado.conditions_results]


def assert_validos(resultados):
    assert resultados
    for nombre, resultado in resultados.items():
        assert estados(resultado) == ["PASS"], (nombre, resultado.value)


def test_viajes_validos_y_sin_mutacion(viajes_sinteticos, estaciones_sinteticas):
    originales = viajes_sinteticos.copy(deep=True)
    catalogo = estaciones_sinteticas.copy(deep=True)
    resultados = validar_viajes(viajes_sinteticos, estaciones_sinteticas)
    assert_validos(resultados)
    assert resultados["duplicados"].value == 0
    assert resultados["nulos"].value.to_numpy().max() == 0
    pd.testing.assert_frame_equal(viajes_sinteticos, originales)
    pd.testing.assert_frame_equal(estaciones_sinteticas, catalogo)


def test_catalogo_y_catalogo_filtrado_para_p2(estaciones_sinteticas):
    assert_validos(validar_estaciones(estaciones_sinteticas))
    p2 = estaciones_sinteticas.loc[estaciones_sinteticas.capacidad.gt(0)]
    assert_validos(validar_estaciones(p2, para_p2=True))
    resultado = validar_estaciones(estaciones_sinteticas, para_p2=True)["integridad"]
    assert resultado.value["capacidad_invalida"] == 1
    assert estados(resultado) == ["FAIL"]


@pytest.mark.parametrize("columna", ["viaje_id", "inicio", "fin", "origen_id", "destino_id"])
def test_nulos_detectados(viajes_sinteticos, estaciones_sinteticas, columna):
    datos = viajes_sinteticos.convert_dtypes()
    datos.loc[0, columna] = pd.NA
    resultado = validar_viajes(datos, estaciones_sinteticas)["nulos"]
    assert resultado.value.loc[columna].iloc[0] == pytest.approx(1 / len(datos))
    assert estados(resultado) == ["FAIL"]


def test_duplicado_exacto(viajes_sinteticos, estaciones_sinteticas):
    datos = pd.concat([viajes_sinteticos, viajes_sinteticos.iloc[:1]], ignore_index=True)
    resultados = validar_viajes(datos, estaciones_sinteticas)
    assert resultados["duplicados"].value == pytest.approx(1 / len(datos))
    assert estados(resultados["duplicados"]) == ["FAIL"]
    assert resultados["integridad"].value["id_repetido"] == 1


def test_id_repetido_en_viajes_distintos(viajes_sinteticos, estaciones_sinteticas):
    datos = viajes_sinteticos.copy()
    datos.loc[1, "viaje_id"] = datos.loc[0, "viaje_id"]
    resultados = validar_viajes(datos, estaciones_sinteticas)
    assert resultados["duplicados"].value == 0
    assert resultados["integridad"].value["id_repetido"] == 1
    assert estados(resultados["integridad"]) == ["FAIL"]


@pytest.mark.parametrize("columna", ["origen_id", "destino_id"])
def test_estacion_inexistente(viajes_sinteticos, estaciones_sinteticas, columna):
    datos = viajes_sinteticos.copy()
    datos.loc[0, columna] = estaciones_sinteticas.estacion_id.max() + 1
    resultado = validar_viajes(datos, estaciones_sinteticas)["integridad"]
    regla = "origen_desconocido" if columna == "origen_id" else "destino_desconocido"
    assert resultado.value[regla] == 1
    assert estados(resultado) == ["FAIL"]


@pytest.mark.parametrize("columna", ["viaje_id", "origen_id", "inicio"])
def test_tipo_invalido_uniforme(viajes_sinteticos, estaciones_sinteticas, columna):
    datos = viajes_sinteticos.astype({columna: "string"})
    resultado = validar_viajes(datos, estaciones_sinteticas)["esquema"]
    assert resultado.value[f"tipo:{columna}"] == 1
    assert estados(resultado) == ["FAIL"]


def test_tipo_mixto_detectado_por_deepchecks(viajes_sinteticos, estaciones_sinteticas):
    datos = viajes_sinteticos.astype({"destino_id": "object"})
    datos.loc[0, "destino_id"] = "invalido"
    resultado = validar_viajes(datos, estaciones_sinteticas)["tipos_mixtos"]
    assert "destino_id" in resultado.value
    assert estados(resultado) == ["FAIL"]


def test_fecha_con_texto_invalido_se_reporta(viajes_sinteticos, estaciones_sinteticas):
    datos = viajes_sinteticos.astype({"inicio": "object"})
    datos.loc[0, "inicio"] = "fecha_invalida_sintetica"
    resultados = validar_viajes(datos, estaciones_sinteticas)
    assert resultados["esquema"].value["tipo:inicio"] == 1
    assert resultados["integridad"].value["fecha_invalida"] == 1
    assert estados(resultados["integridad"]) == ["FAIL"]


def test_reporte_omite_muestras_y_es_json(viajes_sinteticos, estaciones_sinteticas):
    datos = viajes_sinteticos.astype({"destino_id": "object"})
    datos.loc[0, "destino_id"] = "valor_que_no_debe_exportarse"
    resultados = validar_viajes(datos, estaciones_sinteticas)
    reporte = resumir_resultados(resultados)
    texto = json.dumps(reporte, allow_nan=False)
    assert "valor_que_no_debe_exportarse" not in texto
    assert "_examples" not in texto
    assert reporte["tipos_mixtos"]["valor"]["destino_id"]["strings"] == pytest.approx(1 / 200)


def test_validaciones_sin_conexiones_de_red(viajes_sinteticos, estaciones_sinteticas, monkeypatch):
    import socket

    def rechazar(*args, **kwargs):
        pytest.fail("La validación intentó conectarse a la red")

    monkeypatch.setattr(socket.socket, "connect", rechazar)
    assert_validos(validar_viajes(viajes_sinteticos, estaciones_sinteticas))
    datos = pd.DataFrame({"capacidad": np.arange(20, dtype=float)})
    assert estados(evaluar_drift(datos, datos, ["capacidad"])) == ["PASS"]


@pytest.mark.parametrize("minutos", [-1, 0, 1441])
def test_duracion_invalida(viajes_sinteticos, estaciones_sinteticas, minutos):
    datos = viajes_sinteticos.copy()
    datos.loc[0, "fin"] = datos.loc[0, "inicio"] + pd.Timedelta(minutes=minutos)
    resultado = validar_viajes(datos, estaciones_sinteticas)["integridad"]
    assert resultado.value["duracion_invalida"] == 1
    assert estados(resultado) == ["FAIL"]


def test_duracion_limite_24_horas_valida(viajes_sinteticos, estaciones_sinteticas):
    datos = viajes_sinteticos.copy()
    datos.loc[0, "fin"] = datos.loc[0, "inicio"] + pd.Timedelta(hours=24)
    assert_validos(validar_viajes(datos, estaciones_sinteticas))


@pytest.mark.parametrize("fecha", ["antes", "despues", "invalida"])
def test_fecha_fuera_periodo_o_invalida(viajes_sinteticos, estaciones_sinteticas, fecha):
    datos = viajes_sinteticos.copy()
    dato = {
        "antes": pd.Timestamp(config.FECHA_INICIO) - pd.Timedelta(minutes=1),
        "despues": pd.Timestamp(config.FECHA_FIN) + pd.Timedelta(days=1),
        "invalida": pd.NaT,
    }[fecha]
    datos.loc[0, "inicio"] = dato
    datos.loc[0, "fin"] = dato + pd.Timedelta(minutes=10)
    resultado = validar_viajes(datos, estaciones_sinteticas)["integridad"]
    regla = "fecha_invalida" if fecha == "invalida" else "inicio_fuera_periodo"
    assert resultado.value[regla] == 1
    assert estados(resultado) == ["FAIL"]


@pytest.mark.parametrize("capacidad", [-1, np.nan, np.inf, 1.5])
def test_capacidad_invalida(estaciones_sinteticas, capacidad):
    datos = estaciones_sinteticas.astype({"capacidad": "float64"})
    datos.loc[0, "capacidad"] = capacidad
    resultado = validar_estaciones(datos)["integridad"]
    assert resultado.value["capacidad_invalida"] == 1
    assert estados(resultado) == ["FAIL"]


def test_id_estacion_repetido(estaciones_sinteticas):
    datos = estaciones_sinteticas.copy()
    datos.loc[1, "estacion_id"] = datos.loc[0, "estacion_id"]
    resultado = validar_estaciones(datos)["integridad"]
    assert resultado.value["id_repetido"] == 1
    assert estados(resultado) == ["FAIL"]


def test_columnas_ausentes_y_tabla_vacia(viajes_sinteticos, estaciones_sinteticas):
    resultado = validar_viajes(viajes_sinteticos.drop(columns="fin"), estaciones_sinteticas)
    assert resultado["esquema"].value["falta:fin"] == 1
    assert estados(resultado["esquema"]) == ["FAIL"]
    with pytest.raises(ValueError, match="observaciones"):
        validar_viajes(viajes_sinteticos.iloc[:0], estaciones_sinteticas)


@pytest.mark.parametrize("pregunta,columna", [("p1", "fecha"), ("p2", "fecha_franja")])
def test_particiones_produccion_sin_fuga(conjuntos, pregunta, columna):
    resultado = validar_particiones(conjuntos[pregunta], columna)
    assert estados(resultado) == ["PASS"]
    assert not any(resultado.value.values())


def test_cortes_exactos_y_ultimo_dia_completo():
    cortes = pd.to_datetime(
        [config.FECHA_INICIO, config.INICIO_VALIDACION, config.INICIO_PRUEBA, config.FECHA_FIN]
    )
    fechas = [
        cortes[0],
        cortes[1] - pd.Timedelta(seconds=1),
        cortes[1],
        cortes[2] - pd.Timedelta(seconds=1),
        cortes[2],
        cortes[3] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
    ]
    datos = pd.DataFrame({"estacion_id": 1, "fecha": fechas})
    partes = dict(zip(("train", "val", "test"), features.particion_temporal(datos)))
    assert [len(df) for df in partes.values()] == [2, 2, 2]
    assert estados(validar_particiones(partes)) == ["PASS"]


@pytest.mark.parametrize("anomalia", ["fuga", "nat", "vacia", "duplicada", "fuera_periodo"])
def test_problemas_temporales_detectados(conjuntos, anomalia):
    partes = {k: v.copy() for k, v in conjuntos["p1"].items()}
    if anomalia == "fuga":
        partes["train"] = pd.concat([partes["train"], partes["test"].iloc[:1]])
        regla = "fuera_corte:train"
    elif anomalia == "nat":
        partes["val"].loc[partes["val"].index[0], "fecha"] = pd.NaT
        regla = "fecha_invalida"
    elif anomalia == "vacia":
        partes["val"] = partes["val"].iloc[:0]
        regla = "vacia:val"
    elif anomalia == "fuera_periodo":
        partes["test"].loc[partes["test"].index[0], "fecha"] = pd.Timestamp(
            config.FECHA_FIN
        ) + pd.Timedelta(days=1)
        regla = "fuera_corte:test"
    else:
        partes["val"] = pd.concat([partes["val"], partes["val"].iloc[:1]])
        regla = "clave_repetida"
    resultado = validar_particiones(partes)
    assert resultado.value[regla] == 1
    assert estados(resultado) == ["FAIL"]


def test_viaje_que_termina_en_validacion_no_es_informacion_de_train(viajes_sinteticos):
    partes = dict(
        zip(("train", "val", "test"), features.particion_temporal(viajes_sinteticos, "inicio"))
    )
    assert estados(
        validar_particiones(partes, "inicio", claves=("viaje_id",), columna_fin="fin")
    ) == ["PASS"]
    train = partes["train"].copy()
    train.loc[train.index[0], "inicio"] = pd.Timestamp(config.INICIO_VALIDACION) - pd.Timedelta(
        minutes=1
    )
    train.loc[train.index[0], "fin"] = pd.Timestamp(config.INICIO_VALIDACION)
    resultado = validar_particiones(
        {**partes, "train": train}, "inicio", claves=("viaje_id",), columna_fin="fin"
    )
    assert resultado.value["fin_fuera_corte:train"] == 1
    assert estados(resultado) == ["FAIL"]


@pytest.mark.parametrize("columna", ["inicio", "fin"])
@pytest.mark.parametrize("tipo", ["entero", "texto", "con_zona"])
def test_particiones_rechazan_fechas_de_tipo_invalido(viajes_sinteticos, columna, tipo):
    partes = dict(
        zip(("train", "val", "test"), features.particion_temporal(viajes_sinteticos, "inicio"))
    )
    if tipo == "entero":
        # Un epoch válido no debe ocultar que el contrato requiere datetime local.
        partes["train"][columna] = partes["train"][columna].dt.as_unit("ns").astype("int64")
    elif tipo == "texto":
        partes["train"][columna] = partes["train"][columna].astype("string")
    else:
        partes["train"][columna] = partes["train"][columna].dt.tz_localize("UTC")
    with pytest.raises(ValueError, match="datetime|hora local sin zona"):
        validar_particiones(partes, "inicio", claves=("viaje_id",), columna_fin="fin")


def test_rezagos_de_train_no_dependen_de_datos_futuros(sinteticos):
    datos = sinteticos[1][["estacion_id", "fecha", "demanda"]].copy()
    alterados = datos.copy()
    alterados.loc[alterados.fecha.ge(pd.Timestamp(config.INICIO_VALIDACION)), "demanda"] += 10000
    calcular = lambda df: features.agregar_media_movil_7(features.agregar_lag_7(df))
    train_original = features.particion_temporal(calcular(datos))[0]
    train_alterado = features.particion_temporal(calcular(alterados))[0]
    pd.testing.assert_frame_equal(train_original, train_alterado)
    assert train_original.demanda_lag_7.notna().any()
    assert train_original.demanda_media_7.notna().any()


def test_drift_control_y_desplazamiento_con_alerta(sinteticos):
    train = features.particion_temporal(sinteticos[1])[0]
    columnas = ["demanda_lag_1", "demanda_media_7"]
    original = train.copy(deep=True)
    control = evaluar_drift(train, train.copy(), columnas)
    assert estados(control) == ["PASS"]
    assert all(v["Drift score"] == pytest.approx(0) for v in control.value.values())
    desplazado = train.copy()
    desplazado[columnas] += 10000
    alerta = evaluar_drift(train, desplazado, columnas)
    assert estados(alerta) == ["WARN"]
    assert all(v["Drift score"] > 0.2 for v in alerta.value.values())
    assert alerta.passed_conditions(fail_if_warning=False)  # WARN no bloquea integridad.
    pd.testing.assert_frame_equal(train, original)
    reporte = resumir_resultados({"control": control, "desplazamiento": alerta})
    assert json.loads(json.dumps(reporte))["desplazamiento"]["condiciones"][0]["estado"] == "WARN"


@pytest.mark.parametrize("anomalia", ["nulo", "infinito", "vacio", "texto"])
def test_drift_rechaza_entrada_invalida(sinteticos, anomalia):
    datos = sinteticos[1][["demanda_lag_1"]].astype(float)
    alterados = datos.copy()
    if anomalia == "vacio":
        alterados = alterados.iloc[:0]
    elif anomalia == "texto":
        alterados = alterados.astype(str)
    else:
        alterados.iloc[0, 0] = np.nan if anomalia == "nulo" else np.inf
    with pytest.raises(ValueError):
        evaluar_drift(datos, alterados, ["demanda_lag_1"])
