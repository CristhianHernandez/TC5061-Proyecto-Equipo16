"""Pruebas de la prueba de carga (loadtests/) sin levantar Locust ni el stack."""

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

from mibici.serving.schemas import PeticionPrediccion

LOADTESTS = Path(__file__).resolve().parents[1] / "loadtests"
sys.path.insert(0, str(LOADTESTS))

from manifiesto import leer_recursos, resumen_recursos
from validacion import validar_respuesta

MODELO = {"nombre": "mibici-equipo16-p1-demanda", "version": "1"}


def test_respuesta_valida_es_exitosa():
    assert validar_respuesta({"predicciones": [40.7], "modelo": MODELO}, 1) is None


@pytest.mark.parametrize(
    "datos",
    [
        [],  # no es objeto
        {"modelo": MODELO},  # sin predicciones
        {"predicciones": [], "modelo": MODELO},  # lista vacía
        {"predicciones": [1.0, 2.0], "modelo": MODELO},  # cantidad distinta
        {"predicciones": [0.0], "modelo": MODELO},  # no positiva
        {"predicciones": [-3.0], "modelo": MODELO},
        {"predicciones": [float("nan")], "modelo": MODELO},
        {"predicciones": [float("inf")], "modelo": MODELO},
        {"predicciones": ["40.7"], "modelo": MODELO},  # texto
        {"predicciones": [True], "modelo": MODELO},  # booleano
        {"predicciones": [40.7]},  # sin modelo
        {"predicciones": [40.7], "modelo": {"nombre": "otro"}},
    ],
)
def test_respuesta_200_invalida_cuenta_como_fallida(datos):
    assert validar_respuesta(datos, 1) is not None


def test_pool_de_payloads_cumple_el_contrato_http():
    """Cada registro del pool, enviado solo, debe ser una petición válida para la API."""
    registros = json.loads((LOADTESTS / "payloads_p1.json").read_text())["registros"]
    assert len(registros) >= 10
    for registro in registros:
        PeticionPrediccion.model_validate({"registros": [registro]})


def test_recursos_se_convierten_y_resumen_por_ventana(tmp_path):
    ruta = tmp_path / "recursos.csv"
    ruta.write_text(
        "timestamp_utc,contenedor,cpu_pct,memoria\n"
        "2026-11-18T10:00:00Z,mibici-api-1,50.00%,200MiB / 2GiB\n"
        "2026-11-18T10:00:00Z,mibici-mlflow-1,0.20%,360MiB / 1GiB\n"
        "2026-11-18T10:00:03Z,mibici-api-1,150.00%,1GiB / 2GiB\n"
        "2026-11-18T10:00:09Z,mibici-api-1,999.00%,300MiB / 2GiB\n"  # fuera de la ventana
    )
    recursos = leer_recursos(ruta)
    assert recursos["mem_mib"].iloc[2] == pytest.approx(1024)
    assert recursos["limite_mib"].iloc[0] == pytest.approx(2048)
    assert set(recursos["servicio"]) == {"api", "mlflow"}

    resumen = resumen_recursos(recursos, "2026-11-18T10:00:00Z", "2026-11-18T10:00:05Z")
    assert resumen["api"]["muestras"] == 2
    assert resumen["api"]["cpu_pct_prom"] == pytest.approx(100.0)
    assert resumen["api"]["cpu_pct_max"] == pytest.approx(150.0)
    assert resumen["mlflow"]["mem_mib_max"] == pytest.approx(360.0)
    assert isinstance(recursos["timestamp"].iloc[0], pd.Timestamp)
