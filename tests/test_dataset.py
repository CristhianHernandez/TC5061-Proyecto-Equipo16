import hashlib
import json

import pandas as pd
import pytest

from mibici import dataset


def test_leer_csv_utf8(tmp_path):
    ruta = tmp_path / "viajes.csv"
    ruta.write_text("nombre\nMéxico\n", encoding="utf-8")

    df = dataset.leer_csv(ruta)

    assert df.loc[0, "nombre"] == "México"


def test_leer_csv_latin1(tmp_path):
    ruta = tmp_path / "viajes.csv"
    ruta.write_text("nombre\nMéxico\n", encoding="latin-1")

    df = dataset.leer_csv(ruta)

    assert df.loc[0, "nombre"] == "México"


def test_convertir_fecha():
    fechas = pd.Series([
        "01/10/2024 0:00",
        "2025-03-01 00:00:34",
    ])

    resultado = dataset.convertir_fecha(fechas)

    assert resultado.iloc[0] == pd.Timestamp("2024-10-01 00:00:00")
    assert resultado.iloc[1] == pd.Timestamp("2025-03-01 00:00:34")


def test_procesar_viajes(tmp_path):
    ruta_csv = tmp_path / "viajes.csv"
    ruta_parquet = tmp_path / "viajes.parquet"

    datos = pd.DataFrame({
        "Viaje_Id": [1, 2],
        "Usuario_Id": [100, 200],
        "Genero": ["M", "F"],
        "Año_de_nacimiento": [1995, 1998],
        "Inicio_del_viaje": ["01/10/2024 0:00", "01/10/2024 0:15"],
        "Fin_del_viaje": ["01/10/2024 0:10", "01/10/2024 0:25"],
        "Origen_Id": [40, 25],
        "Destino_Id": [213, 50],
    })

    datos.to_csv(ruta_csv, index=False)

    filas = dataset.procesar_viajes(ruta_csv, ruta_parquet)
    resultado = pd.read_parquet(ruta_parquet)

    assert filas == 2
    assert list(resultado.columns) == [
        "viaje_id", "inicio", "fin", "origen_id", "destino_id"
    ]
    assert resultado["viaje_id"].dtype == "int64"
    assert resultado["origen_id"].dtype == "int32"
    assert resultado["destino_id"].dtype == "int32"
    assert pd.api.types.is_datetime64_any_dtype(resultado["inicio"])
    assert pd.api.types.is_datetime64_any_dtype(resultado["fin"])

    # Los datos personales no deben aparecer en el Parquet.
    assert "Usuario_Id" not in resultado.columns
    assert "Genero" not in resultado.columns
    assert "Año_de_nacimiento" not in resultado.columns


def test_procesar_estaciones(tmp_path, monkeypatch):
    ruta_csv = tmp_path / "estaciones.csv"
    ruta_parquet = tmp_path / "estaciones.parquet"

    datos = pd.DataFrame({
        "id": [1, 2],
        "name": ["GDL Centro", "ex-ZPN Norte"],
        "latitude": [20.67, 20.70],
        "longitude": [-103.35, -103.40],
        "dpcapacity": [20, 15],
    })

    datos.to_csv(ruta_csv, index=False)
    monkeypatch.setattr(dataset, "ESTACIONES_PARQUET", ruta_parquet)

    filas = dataset.procesar_estaciones(ruta_csv)
    resultado = pd.read_parquet(ruta_parquet)

    assert filas == 2
    assert list(resultado.columns) == [
        "estacion_id", "nombre", "municipio",
        "latitud", "longitud", "capacidad", "activa"
    ]
    assert resultado["municipio"].tolist() == ["GDL", "ZPN"]
    assert resultado["activa"].tolist() == [True, False]
    assert resultado["nombre"].tolist() == ["GDL Centro", "ZPN Norte"]


def test_calcular_sha256(tmp_path):
    ruta = tmp_path / "archivo.csv"
    ruta.write_bytes(b"datos de prueba")

    esperado = hashlib.sha256(b"datos de prueba").hexdigest()

    assert dataset.calcular_sha256(ruta) == esperado


def test_generar_manifiesto(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    viajes = tmp_path / "viajes"
    manifiesto = tmp_path / "manifiesto.json"

    raw.mkdir()
    viajes.mkdir()

    mes = "2024_10"
    nombre = f"datos_abiertos_{mes}.csv"

    (raw / nombre).write_text("datos de prueba", encoding="utf-8")
    pd.DataFrame({"viaje_id": [1, 2]}).to_parquet(
        viajes / f"viajes_{mes}.parquet",
        index=False,
    )

    monkeypatch.setattr(dataset, "RAW_DATA_DIR", raw)
    monkeypatch.setattr(dataset, "VIAJES_DIR", viajes)
    monkeypatch.setattr(dataset, "MANIFIESTO_DATOS", manifiesto)
    monkeypatch.setattr(dataset, "URLS_VIAJES", {mes: "https://ejemplo.com"})

    dataset.generar_manifiesto()
    primera_version = manifiesto.read_bytes()

    dataset.generar_manifiesto()
    segunda_version = manifiesto.read_bytes()

    assert primera_version == segunda_version

    contenido = json.loads(primera_version)
    assert len(contenido["archivos"]) == 1
    assert contenido["archivos"][0]["filas"] == 2
    assert contenido["archivos"][0]["archivo"] == nombre
    assert contenido["archivos"][0]["sha256"] == hashlib.sha256(
        b"datos de prueba"
    ).hexdigest()