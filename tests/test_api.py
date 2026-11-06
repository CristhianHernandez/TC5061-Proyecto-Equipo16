"""Contrato HTTP LH3 con el loader real de CH1 y un predictor determinista, sin MLflow remoto."""

import json

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from mibici.serving import api
from mibici.serving.model_loader import ErrorEntrada, ModeloNoDisponible, ModeloServido, Variable


@pytest.fixture
def registro_p1():
    return {
        "dia_semana": 2,
        "mes": 9,
        "es_fin_semana": 0,
        "dia_anio": 273,
        "es_festivo": 0,
        "demanda_lag_1": 40.0,
        "demanda_lag_7": 29.0,
        "demanda_media_7": 37.7,
        "capacidad": 15,
        "latitud": 20.675,
        "longitud": -103.345,
    }


@pytest.fixture
def servido_api():
    class Predictor:
        def __init__(self):
            self.entradas = []

        def predict(self, entrada):
            self.entradas.append(entrada.copy(deep=True))
            # Regla sintética para verificar orden y contenido, no simula métricas del champion.
            return (entrada.capacidad * 2 + entrada.demanda_lag_1.fillna(0)).to_numpy()

    # Firma acordada por CH1, independiente de las constantes de la API.
    tipos = {
        "dia_semana": "integer",
        "mes": "integer",
        "es_fin_semana": "long",
        "dia_anio": "integer",
        "es_festivo": "long",
        "demanda_lag_1": "double",
        "demanda_lag_7": "double",
        "demanda_media_7": "double",
        "capacidad": "integer",
        "latitud": "double",
        "longitud": "double",
    }
    variables = [
        Variable(nombre, tipo, not nombre.startswith("demanda_")) for nombre, tipo in tipos.items()
    ]
    return ModeloServido(
        Predictor(),
        {
            "nombre": "modelo-sintetico-p1",
            "version": "prueba",
            "alias": "prueba",
            "run_id": "sintetico",
            "uri": "file:///ruta-privada",
            "linaje": {"token": "secreto-que-no-debe-salir"},
        },
        variables,
    )


@pytest.fixture
def cliente_api(servido_api, monkeypatch):
    def sin_red(*args, **kwargs):
        pytest.fail("Las pruebas API no deben llamar servicios HTTP externos")

    # TestClient usa un transporte ASGI propio. No se bloquea socketpair, que
    # asyncio necesita internamente en Windows para arrancar su event loop.
    monkeypatch.setattr("requests.sessions.Session.request", sin_red)
    monkeypatch.setattr("httpx.HTTPTransport.handle_request", sin_red)
    with TestClient(api.crear_app(lambda: servido_api)) as cliente:
        yield cliente


def test_predict_200_orden_y_json(cliente_api, registro_p1, servido_api):
    segundo = {**registro_p1, "capacidad": 20, "demanda_lag_1": 3.0}
    payload = {"registros": [dict(reversed(list(registro_p1.items()))), segundo]}
    respuesta = cliente_api.post("/predict", json=payload)
    assert respuesta.status_code == 200
    assert respuesta.json() == {
        "predicciones": [70.0, 43.0],
        "modelo": {
            "nombre": "modelo-sintetico-p1",
            "version": "prueba",
            "alias": "prueba",
            "run_id": "sintetico",
        },
    }
    assert list(servido_api.modelo.entradas[0]) == servido_api.columnas
    assert servido_api.modelo.entradas[0].capacidad.dtype == "int32"
    assert servido_api.modelo.entradas[0].es_festivo.dtype == "int64"
    assert "secreto" not in respuesta.text and "ruta-privada" not in respuesta.text
    json.dumps(respuesta.json(), allow_nan=False)


def test_rezagos_opcionales_y_null_equivalentes(cliente_api, registro_p1, servido_api):
    rezagos = ["demanda_lag_1", "demanda_lag_7", "demanda_media_7"]
    sin_rezagos = {k: v for k, v in registro_p1.items() if k not in rezagos}
    nulos = {**registro_p1, **dict.fromkeys(rezagos)}
    uno = cliente_api.post("/predict", json={"registros": [sin_rezagos]})
    dos = cliente_api.post("/predict", json={"registros": [nulos]})
    assert uno.status_code == dos.status_code == 200
    assert uno.json()["predicciones"] == dos.json()["predicciones"] == [30.0]
    assert servido_api.modelo.entradas[0][rezagos].isna().all().all()


@pytest.mark.parametrize(
    "cambio",
    [
        {"dia_semana": -1},
        {"dia_semana": 7},
        {"dia_semana": 2.0},
        {"mes": 0},
        {"mes": 13},
        {"dia_anio": 0},
        {"dia_anio": 367},
        {"es_fin_semana": 1},
        {"es_festivo": 2},
        {"es_festivo": True},
        {"capacidad": 0},
        {"capacidad": -2},
        {"capacidad": 2**31},
        {"capacidad": "15"},
        {"capacidad": None},
        {"latitud": None},
        {"latitud": 91},
        {"longitud": -181},
        {"latitud": "20.675"},
        {"demanda_lag_1": -1},
        {"demanda_lag_7": False},
        {"demanda_media_7": "37.7"},
        {"estacion_id": 55},
    ],
)
def test_entrada_invalida_422_no_invoca_modelo(cliente_api, registro_p1, servido_api, cambio):
    respuesta = cliente_api.post("/predict", json={"registros": [{**registro_p1, **cambio}]})
    assert respuesta.status_code == 422
    assert respuesta.json()["detail"]
    assert servido_api.modelo.entradas == []


@pytest.mark.parametrize("faltante", ["dia_semana", "capacidad", "longitud"])
def test_campos_obligatorios(cliente_api, registro_p1, servido_api, faltante):
    registro_p1.pop(faltante)
    respuesta = cliente_api.post("/predict", json={"registros": [registro_p1]})
    assert respuesta.status_code == 422
    assert faltante in respuesta.text
    assert servido_api.modelo.entradas == []


@pytest.mark.parametrize(
    "payload",
    [{}, {"registros": []}, {"registros": [42]}, [], {"registros": "texto"}, {"otro": 1}],
)
def test_cuerpo_invalido(cliente_api, servido_api, payload):
    assert cliente_api.post("/predict", json=payload).status_code == 422
    assert servido_api.modelo.entradas == []


@pytest.mark.parametrize("valor", [float("nan"), float("inf"), -float("inf")])
def test_no_finitos_en_peticion_son_422_json_valido(cliente_api, registro_p1, servido_api, valor):
    registro_p1["demanda_lag_1"] = valor
    respuesta = cliente_api.post(
        "/predict",
        content=json.dumps({"registros": [registro_p1]}),
        headers={"Content-Type": "application/json"},
    )
    assert respuesta.status_code == 422
    json.dumps(respuesta.json(), allow_nan=False)
    assert servido_api.modelo.entradas == []


def test_json_malformado_422(cliente_api):
    respuesta = cliente_api.post(
        "/predict", content="{", headers={"Content-Type": "application/json"}
    )
    assert respuesta.status_code == 422


def test_health_swagger_y_openapi(cliente_api):
    salud = cliente_api.get("/health")
    assert salud.status_code == 200 and salud.json()["estado"] == "listo"
    assert "ruta-privada" not in salud.text and "secreto" not in salud.text
    assert cliente_api.get("/docs").status_code == 200
    esquema = cliente_api.get("/openapi.json").json()
    assert "post" in esquema["paths"]["/predict"]
    assert "RespuestaPrediccion" in esquema["components"]["schemas"]
    assert esquema["components"]["schemas"]["RegistroP1"]["additionalProperties"] is False


def test_modelo_se_carga_una_vez_y_se_libera(servido_api, registro_p1):
    llamadas = []

    def cargar():
        llamadas.append(1)
        return servido_api

    aplicacion = api.crear_app(cargar)
    assert llamadas == []  # Tampoco se carga al importar el módulo.
    with TestClient(aplicacion) as cliente:
        assert cliente.get("/health").status_code == 200
        for _ in range(2):
            assert cliente.post("/predict", json={"registros": [registro_p1]}).status_code == 200
        assert llamadas == [1]
    assert aplicacion.state.modelo is None


def test_loader_predeterminado_es_el_de_ch1(monkeypatch, servido_api):
    llamadas = []
    monkeypatch.setattr(api, "cargar_modelo", lambda: llamadas.append(1) or servido_api)
    with TestClient(api.crear_app()) as cliente:
        assert cliente.get("/health").status_code == 200
    assert llamadas == [1]


@pytest.mark.parametrize("error", [ModeloNoDisponible, OSError, RuntimeError])
def test_carga_fallida_health_y_predict_503_sin_filtraciones(registro_p1, error):
    def cargar():
        raise error("file:///ruta-interna?token=secreto-interno")

    with TestClient(api.crear_app(cargar)) as cliente:
        salud = cliente.get("/health")
        prediccion = cliente.post("/predict", json={"registros": [registro_p1]})
        assert salud.status_code == prediccion.status_code == 503
        assert salud.json() == {"estado": "no_disponible", "modelo": None}
        assert "ruta-interna" not in salud.text + prediccion.text
        assert "secreto-interno" not in salud.text + prediccion.text
        assert cliente.get("/docs").status_code == 200


def test_firma_incompatible_no_reporta_listo(servido_api):
    servido_api.variables.append(Variable("demanda_media_28", "double", True))
    with TestClient(api.crear_app(lambda: servido_api)) as cliente:
        assert cliente.get("/health").status_code == 503


@pytest.mark.parametrize(
    "error,estado",
    [(ErrorEntrada, 422), (ModeloNoDisponible, 503), (RuntimeError, 500), (ValueError, 500)],
)
def test_errores_del_modelo_controlados(
    cliente_api, registro_p1, servido_api, monkeypatch, error, estado
):
    def fallar(_):
        raise error("token=secreto-interno; /ruta-interna/modelo")

    monkeypatch.setattr(servido_api, "predecir", fallar)
    respuesta = cliente_api.post("/predict", json={"registros": [registro_p1]})
    assert respuesta.status_code == estado
    assert "secreto-interno" not in respuesta.text and "ruta-interna" not in respuesta.text
    if estado == 503:
        assert cliente_api.get("/health").status_code == 503


@pytest.mark.parametrize(
    "salida",
    [[np.nan], [np.inf], [-np.inf], [0.0], [-1.0], [], [1.0, 2.0], [[1.0]], ["1.0"], [True], 1.0],
)
def test_salida_invalida_500(cliente_api, registro_p1, servido_api, monkeypatch, salida):
    monkeypatch.setattr(servido_api, "predecir", lambda _: salida)
    respuesta = cliente_api.post("/predict", json={"registros": [registro_p1]})
    assert respuesta.status_code == 500
    assert respuesta.json() == {"detail": api.ERROR_PREDICCION}


def test_salida_numpy_serializable(cliente_api, registro_p1, servido_api, monkeypatch):
    monkeypatch.setattr(servido_api, "predecir", lambda _: np.array([2.5], dtype=np.float32))
    respuesta = cliente_api.post("/predict", json={"registros": [registro_p1]})
    assert respuesta.status_code == 200 and respuesta.json()["predicciones"] == [2.5]


@pytest.mark.parametrize("salida", [[True, 2.5], [2, True]])
def test_salida_mixta_no_oculta_booleanos(
    cliente_api, registro_p1, servido_api, monkeypatch, salida
):
    monkeypatch.setattr(servido_api, "predecir", lambda _: salida)
    respuesta = cliente_api.post("/predict", json={"registros": [registro_p1, registro_p1]})
    assert respuesta.status_code == 500
    assert respuesta.json() == {"detail": api.ERROR_PREDICCION}


def test_peticion_en_curso_conserva_metadatos_del_modelo(
    cliente_api, registro_p1, servido_api, monkeypatch
):
    def predecir(_):
        # Simula otra petición que declara perdido el modelo mientras esta infiere.
        cliente_api.app.state.modelo = None
        cliente_api.app.state.modelo_publico = None
        return [2.5]

    monkeypatch.setattr(servido_api, "predecir", predecir)
    respuesta = cliente_api.post("/predict", json={"registros": [registro_p1]})
    assert respuesta.status_code == 200
    assert respuesta.json()["predicciones"] == [2.5]
    assert respuesta.json()["modelo"]["nombre"] == "modelo-sintetico-p1"
    assert cliente_api.get("/health").status_code == 503


def test_no_impone_los_maximos_observados_en_train(cliente_api, registro_p1):
    registro_p1.update(capacidad=50, demanda_lag_1=1000.0, demanda_media_7=900.0)
    respuesta = cliente_api.post("/predict", json={"registros": [registro_p1]})
    assert respuesta.status_code == 200 and respuesta.json()["predicciones"] == [1100.0]


def test_no_muta_entrada(cliente_api, registro_p1):
    antes = pd.DataFrame([registro_p1])
    assert cliente_api.post("/predict", json={"registros": [registro_p1]}).status_code == 200
    pd.testing.assert_frame_equal(pd.DataFrame([registro_p1]), antes)
