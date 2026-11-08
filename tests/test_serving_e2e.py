"""Contrato HTTP de P1 contra el contenedor real indicado por MIBICI_API_URL.

Sin esa variable se omite el módulo; no se inicia una API ni se sustituye el modelo.
"""

import json
import math
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

URL_BASE = os.environ.get("MIBICI_API_URL", "").rstrip("/")
if not URL_BASE:
    pytest.skip(
        "Requiere el contenedor de la API en ejecución y MIBICI_API_URL definida",
        allow_module_level=True,
    )

pytestmark = pytest.mark.contenedor

EJEMPLO = Path(__file__).resolve().parents[1] / "docs/serving/ejemplo_entrada_p1.json"
REZAGOS = ["demanda_lag_1", "demanda_lag_7", "demanda_media_7"]
PREDICCIONES_ESPERADAS = [40.697957479000046, 276.77983750628846]


def solicitar(ruta, entrada=None):
    """Devuelve código, cabeceras y cuerpo, incluso cuando HTTP responde con un error."""
    datos = None if entrada is None else json.dumps(entrada, allow_nan=False).encode("utf-8")
    cabeceras = {} if datos is None else {"Content-Type": "application/json"}
    peticion = Request(f"{URL_BASE}{ruta}", data=datos, headers=cabeceras)
    try:
        respuesta = urlopen(peticion, timeout=10)
    except HTTPError as error:
        respuesta = error
    with respuesta:
        return respuesta.code, respuesta.headers, respuesta.read().decode("utf-8")


@pytest.fixture
def ejemplo():
    """Cada prueba recibe una copia independiente del ejemplo real del contrato."""
    return json.loads(EJEMPLO.read_text(encoding="utf-8"))


@pytest.fixture
def modelo_publico():
    codigo, cabeceras, cuerpo = solicitar("/health")
    assert codigo == 200, cuerpo
    assert cabeceras.get_content_type() == "application/json"
    salud = json.loads(cuerpo)
    assert salud["estado"] == "listo"
    modelo = salud["modelo"]
    assert set(modelo) == {"nombre", "version", "alias", "run_id"}
    assert modelo["nombre"] == "mibici-equipo16-p1-demanda"
    assert modelo["alias"] == "champion"
    assert isinstance(modelo["version"], str) and modelo["version"]
    assert isinstance(modelo["run_id"], str) and modelo["run_id"]
    return modelo


def test_salud_confirma_el_champion_listo(modelo_publico):
    assert modelo_publico["alias"] == "champion"


def test_predice_el_ejemplo_real_con_el_mismo_modelo_que_salud(ejemplo, modelo_publico):
    codigo, cabeceras, cuerpo = solicitar("/predict", ejemplo)
    assert codigo == 200, cuerpo
    assert cabeceras.get_content_type() == "application/json"
    salida = json.loads(cuerpo)
    predicciones = salida["predicciones"]
    assert len(predicciones) == len(ejemplo["registros"])
    assert all(type(p) is float and math.isfinite(p) and p > 0 for p in predicciones)
    assert salida["modelo"] == modelo_publico
    assert predicciones == pytest.approx(PREDICCIONES_ESPERADAS, rel=1e-6)


@pytest.mark.parametrize("nulos", [True, False], ids=["nulos", "omitidos"])
def test_acepta_un_registro_con_todos_los_rezagos_nulos_o_ausentes(ejemplo, nulos):
    registro = ejemplo["registros"][0]
    for variable in REZAGOS:
        if nulos:
            registro[variable] = None
        else:
            registro.pop(variable)
    codigo, cabeceras, cuerpo = solicitar("/predict", {"registros": [registro]})
    assert codigo == 200, cuerpo
    assert cabeceras.get_content_type() == "application/json"
    predicciones = json.loads(cuerpo)["predicciones"]
    assert len(predicciones) == 1
    assert all(type(p) is float and math.isfinite(p) and p > 0 for p in predicciones)


@pytest.mark.parametrize(
    "caso",
    ["falta_obligatoria", "tipo_invalido", "fuera_de_dominio", "extra", "vacia", "incoherente"],
)
def test_rechaza_entradas_invalidas_sin_filtrar_rutas(ejemplo, caso):
    registro = ejemplo["registros"][0]
    if caso == "falta_obligatoria":
        registro.pop("capacidad")
    elif caso == "tipo_invalido":
        registro["mes"] = "septiembre"
    elif caso == "fuera_de_dominio":
        registro["mes"] = 13
    elif caso == "extra":
        registro["estacion_id"] = 55
    elif caso == "vacia":
        ejemplo["registros"] = []
    elif caso == "incoherente":
        registro["es_fin_semana"] = 1 - int(registro["dia_semana"] >= 5)
    codigo, cabeceras, cuerpo = solicitar("/predict", ejemplo)
    assert codigo == 422, cuerpo
    assert cabeceras.get_content_type() == "application/json"
    detalles = json.loads(cuerpo)["detail"]
    assert isinstance(detalles, list) and detalles
    assert "/app" not in cuerpo
    assert "Traceback" not in cuerpo


def test_documentacion_y_esquema_publican_los_endpoints():
    codigo, cabeceras, cuerpo = solicitar("/docs")
    assert codigo == 200, cuerpo
    assert cabeceras.get_content_type() == "text/html"
    codigo, cabeceras, cuerpo = solicitar("/openapi.json")
    assert codigo == 200, cuerpo
    assert cabeceras.get_content_type() == "application/json"
    rutas = json.loads(cuerpo)["paths"]
    assert "post" in rutas["/predict"]
    assert "get" in rutas["/health"]
