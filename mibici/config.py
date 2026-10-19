"""Configuración central del proyecto: única fuente de rutas, fechas, semillas y parámetros.

Ningún otro módulo debe definir rutas, fechas de corte ni semillas; todos los importan de aquí.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REFERENCES_DIR = PROJ_ROOT / "references"

# Datos procesados (esquema acordado en references/plan-de-implementacion.md)
VIAJES_DIR = PROCESSED_DATA_DIR / "viajes"  # un archivo viajes_AAAA_MM.parquet por mes
ESTACIONES_PARQUET = PROCESSED_DATA_DIR / "estaciones.parquet"

# Manifiesto de versión de los datos: vive fuera de data/ porque sí se versiona en Git
MANIFIESTO_DATOS = REFERENCES_DIR / "manifiesto_datos.json"

# Rango de fechas y partición temporal (inicio inclusivo; cada corte inicia el siguiente conjunto)
FECHA_INICIO = "2024-10-01"
INICIO_VALIDACION = "2026-04-01"
INICIO_PRUEBA = "2026-07-01"
FECHA_FIN = "2026-09-30"

# Reproducibilidad
SEMILLA = 16

# Fuente: https://mibici.net/es/datos-abiertos/ (consultado el 2026-10-14)
URL_BASE_MIBICI = "https://mibici.net/site/assets/files"
URL_NOMENCLATURA = f"{URL_BASE_MIBICI}/1118/nomenclatura_2026_09.csv"

# El identificador de carpeta del portal cambia por mes, por eso se listan uno a uno
_CARPETAS_VIAJES = {
    "2024_10": 91758,
    "2024_11": 93516,
    "2024_12": 94962,
    "2025_01": 97653,
    "2025_02": 99879,
    "2025_03": 101986,
    "2025_04": 103813,
    "2025_05": 105513,
    "2025_06": 107088,
    "2025_07": 108898,
    "2025_08": 111021,
    "2025_09": 113216,
    "2025_10": 115104,
    "2025_11": 116773,
    "2025_12": 118153,
    "2026_01": 120163,
    "2026_02": 121761,
    "2026_03": 123236,
    "2026_04": 124927,
    "2026_05": 125943,
    "2026_06": 127734,
    "2026_07": 129006,
    "2026_08": 130362,
    "2026_09": 131825,
}
URLS_VIAJES = {
    mes: f"{URL_BASE_MIBICI}/{carpeta}/datos_abiertos_{mes}.csv"
    for mes, carpeta in _CARPETAS_VIAJES.items()
}

# El servidor de MiBici responde 406 a los User-Agent `python-requests/*` y `Mozilla/5.0`;
# acepta uno descriptivo que identifica al proyecto (verificado el 2026-10-18)
CABECERAS_DESCARGA = {
    "User-Agent": "mibici-equipo16/0.1 (proyecto academico TC5061)",
    "Accept": "text/csv,text/plain,*/*",
}

# P2: riesgo aproximado con flujo neto y capacidad (ver references/propuesta-parte-a.md)
ALFA_RIESGO = 0.5
HORAS_FRANJA = 3

# MLflow: los runs se envían al servidor que levanta `make mlflow` (ver README).
# En macOS el puerto 5000 lo ocupa AirPlay Receiver; en ese caso se cambia el URI en .env
EQUIPO = "16"
EXPERIMENTO_MLFLOW = "mibici-equipo16"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
