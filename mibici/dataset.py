import hashlib
import json
import os
from pathlib import Path

from loguru import logger
import pandas as pd
import requests
import typer

from mibici.config import (
    CABECERAS_DESCARGA,
    ESTACIONES_PARQUET,
    FECHA_FIN,
    FECHA_INICIO,
    FUENTE_DATOS,
    MANIFIESTO_DATOS,
    RAW_DATA_DIR,
    URL_NOMENCLATURA,
    URLS_VIAJES,
    VIAJES_DIR,
)

app = typer.Typer()


def descargar_archivo(url: str, ruta: Path):
    """Descarga el archivo únicamente cuando no existe."""

    if ruta.exists():
        logger.info(f"Ya existe: {ruta.name}")
        return

    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + ".part")

    try:
        with requests.get(
            url,
            headers=CABECERAS_DESCARGA,
            stream=True,
            timeout=(15, 120),
        ) as respuesta:
            respuesta.raise_for_status()

            with temporal.open("wb") as archivo:
                for bloque in respuesta.iter_content(chunk_size=1024 * 1024):
                    if bloque:
                        archivo.write(bloque)

        temporal.replace(ruta)
        logger.info(f"Descargado: {ruta.name}")

    except Exception:
        temporal.unlink(missing_ok=True)
        raise


def leer_csv(ruta: Path):
    """Lee un CSV de MiBici probando UTF-8 y, si falla, latin-1 (varía entre meses)."""
    try:
        df = pd.read_csv(ruta, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(ruta, encoding="latin-1")

    df.columns = df.columns.str.strip().str.replace("\ufeff", "")
    return df


def convertir_fecha(columna):
    """Considera los formatos de fecha encontrados en los viajes."""

    columna = columna.astype("string").str.strip()
    formato_iso = columna.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)

    fechas = pd.Series(pd.NaT, index=columna.index, dtype="datetime64[ns]")

    fechas.loc[formato_iso] = pd.to_datetime(
        columna.loc[formato_iso],
        format="mixed",
        errors="raise",
    )

    # Los registros más antiguos utilizan día/mes/año.
    fechas.loc[~formato_iso] = pd.to_datetime(
        columna.loc[~formato_iso],
        format="mixed",
        dayfirst=True,
        errors="raise",
    )

    return fechas


def procesar_viajes(ruta_csv: Path, ruta_parquet: Path):
    """Normaliza los viajes y los guarda en Parquet."""

    df = leer_csv(ruta_csv)

    df = df.rename(
        columns={
            "Viaje_Id": "viaje_id",
            "Inicio_del_viaje": "inicio",
            "Inicio_de_viaje": "inicio",
            "Fin_del_viaje": "fin",
            "Fin_de_viaje": "fin",
            "Origen_Id": "origen_id",
            "Destino_Id": "destino_id",
        }
    )

    # Solo se conservan las columnas necesarias para el proyecto.
    columnas = ["viaje_id", "inicio", "fin", "origen_id", "destino_id"]
    df = df[columnas].copy()

    df["viaje_id"] = df["viaje_id"].astype("int64")
    df["origen_id"] = df["origen_id"].astype("int32")
    df["destino_id"] = df["destino_id"].astype("int32")

    df["inicio"] = convertir_fecha(df["inicio"])
    df["fin"] = convertir_fecha(df["fin"])

    ruta_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ruta_parquet, index=False)

    logger.info(f"{ruta_parquet.name}: {len(df)} viajes")
    return len(df)


def procesar_estaciones(ruta_csv: Path):
    """Prepara el catálogo de estaciones de MiBici."""

    df = leer_csv(ruta_csv)

    df = df.rename(
        columns={
            "id": "estacion_id",
            "name": "nombre",
            "latitude": "latitud",
            "longitude": "longitud",
            "dpcapacity": "capacidad",
        }
    )

    columnas = [
        "estacion_id",
        "nombre",
        "latitud",
        "longitud",
        "capacidad",
    ]

    df = df[columnas].copy()
    df["nombre"] = df["nombre"].astype("string").str.strip()

    # El prefijo ex- indica que la estación dejó de operar.
    df["activa"] = ~df["nombre"].str.lower().str.startswith("ex-", na=False)

    df["nombre"] = df["nombre"].str.replace(r"^ex-\s*", "", regex=True, case=False)

    # Municipio identificado en el prefijo del nombre; en la nomenclatura real viene entre
    # paréntesis, por ejemplo "(GDL-001) C. Epigmenio Glez./ Av. 16 de Sept.".
    df["municipio"] = df["nombre"].str.extract(r"^\(?(GDL|ZPN|TLQ)\b", expand=False)

    df["estacion_id"] = df["estacion_id"].astype("int32")
    df["capacidad"] = df["capacidad"].astype("int32")
    df["latitud"] = df["latitud"].astype("float64")
    df["longitud"] = df["longitud"].astype("float64")

    df["municipio"] = df["municipio"].astype("string")
    df["activa"] = df["activa"].astype(bool)

    df = df[
        [
            "estacion_id",
            "nombre",
            "municipio",
            "latitud",
            "longitud",
            "capacidad",
            "activa",
        ]
    ]

    ESTACIONES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ESTACIONES_PARQUET, index=False)

    logger.info(f"Estaciones procesadas: {len(df)}")
    return len(df)


def cargar_viajes(meses: list[str] | None = None):
    """Carga todos los meses disponibles o únicamente los solicitados."""

    if meses is None:
        archivos = sorted(VIAJES_DIR.glob("viajes_*.parquet"))
    else:
        archivos = [VIAJES_DIR / f"viajes_{mes}.parquet" for mes in meses]

    if not archivos:
        raise FileNotFoundError("No se encontraron viajes procesados.")

    datos = [pd.read_parquet(archivo) for archivo in archivos]
    return pd.concat(datos, ignore_index=True)


def cargar_estaciones():
    """Lee el catálogo de estaciones procesado."""

    return pd.read_parquet(ESTACIONES_PARQUET)


def calcular_sha256(ruta: Path):
    """Obtiene la huella SHA-256 de un archivo."""

    sha256 = hashlib.sha256()

    # Lectura por bloques para evitar cargar el CSV completo en memoria.
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            sha256.update(bloque)

    return sha256.hexdigest()


def generar_manifiesto():
    """Registra los archivos utilizados y sus hashes."""

    archivos = []

    for mes in sorted(URLS_VIAJES):
        nombre_csv = f"datos_abiertos_{mes}.csv"
        ruta_csv = RAW_DATA_DIR / nombre_csv
        ruta_parquet = VIAJES_DIR / f"viajes_{mes}.parquet"

        filas = len(pd.read_parquet(ruta_parquet, columns=["viaje_id"]))

        archivos.append(
            {
                "archivo": nombre_csv,
                "sha256": calcular_sha256(ruta_csv),
                "filas": filas,
                "parquet": f"viajes/viajes_{mes}.parquet",
            }
        )

    manifiesto = {
        "fuente": FUENTE_DATOS,
        "fecha_inicio": FECHA_INICIO,
        "fecha_fin": FECHA_FIN,
        "archivos": archivos,
    }

    MANIFIESTO_DATOS.parent.mkdir(parents=True, exist_ok=True)

    # Se mantiene un orden fijo y no se agrega fecha de generación.
    temporal = MANIFIESTO_DATOS.with_suffix(".json.part")

    with temporal.open("w", encoding="utf-8") as archivo:
        json.dump(
            manifiesto,
            archivo,
            indent=2,
            ensure_ascii=False,
        )
        archivo.write("\n")

    temporal.replace(MANIFIESTO_DATOS)

    logger.info(f"Manifiesto generado: {len(archivos)} archivos")
    return manifiesto


def ejecutar_ingesta():
    """Ejecuta la descarga, transformación y registro de los datos."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    VIAJES_DIR.mkdir(parents=True, exist_ok=True)

    for mes, url in URLS_VIAJES.items():
        ruta_csv = RAW_DATA_DIR / f"datos_abiertos_{mes}.csv"
        ruta_parquet = VIAJES_DIR / f"viajes_{mes}.parquet"

        descargar_archivo(url, ruta_csv)

        if ruta_parquet.exists():
            logger.info(f"Ya procesado: {mes}")
            continue

        procesar_viajes(ruta_csv, ruta_parquet)

    # El catálogo de estaciones se procesa por separado.
    nombre = URL_NOMENCLATURA.split("/")[-1]
    ruta_estaciones = RAW_DATA_DIR / nombre

    descargar_archivo(URL_NOMENCLATURA, ruta_estaciones)

    if not ESTACIONES_PARQUET.exists():
        procesar_estaciones(ruta_estaciones)

    generar_manifiesto()

    logger.info("Ingesta terminada.")


def registrar_ingesta():
    """Bajo `mlflow run` documenta la ingesta: manifiesto en data/ y versión de los datos.

    `mlflow run` crea una corrida para la entrada `ingest`; se reanuda para que el experimento
    conserve qué versión de los datos se generó, sin registrar los datos mismos.
    """
    run_id = os.environ.pop("MLFLOW_RUN_ID", None)
    if run_id is None:
        return
    import mlflow

    from mibici.tracking import version_datos

    with mlflow.start_run(run_id=run_id):
        mlflow.set_tags({"mlflow.runName": "ingesta", "uso": "ingesta"})
        mlflow.log_params(
            {
                "fuente": FUENTE_DATOS,
                "fecha_inicio": FECHA_INICIO,
                "fecha_fin": FECHA_FIN,
                "meses": len(URLS_VIAJES),
                "version_datos": version_datos(MANIFIESTO_DATOS),
            }
        )
        mlflow.log_artifact(str(MANIFIESTO_DATOS), "data")


@app.command()
def main():
    """Entrada `ingest` del MLflow Project: descarga, normaliza y versiona los datos."""
    ejecutar_ingesta()
    registrar_ingesta()


if __name__ == "__main__":
    app()
