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

# DH4: rejillas pequeñas y protocolo reproducible de entrenamiento.
REJILLA_P1 = {
    "learning_rate": [0.05, 0.1],
    "max_leaf_nodes": [31, 63],
    "max_iter": [100, 200],
}
REJILLA_LOGISTICA = {"C": [0.1, 1.0]}
REJILLA_P2 = {"class_weight": [None, "balanced"], "learning_rate": [0.1]}
ITERACIONES_HGB = 300
RANGO_K_P3 = tuple(range(2, 9))
N_SEMILLAS_P3 = 3
CLASES_RIESGO = ("normal", "sin_anclajes", "sin_bicicletas")
MINIMO_POISSON = 1e-6
UMBRAL_EMPATE_P1 = 0.02  # diferencia relativa de MAE de validación para comparar estabilidad y costo

# Entrega 2: MLflow Project, experimentación y Model Registry.
# Los parámetros vacíos del MLproject se resuelven con estos valores, así el MLproject no repite
# fechas, rutas, semillas ni hiperparámetros.
FUENTE_DATOS = "https://mibici.net/es/datos-abiertos/"
PREGUNTAS_PROYECTO = ("p1", "p2", "p3")

# Modelo de referencia (baseline) de cada pregunta: es el que corre `train` sin `model` y contra el
# que cada candidata calcula su mejora.
BASELINES = {"p1": "naive", "p2": "persistencia", "p3": "municipio"}

# Métrica principal y dirección: se usa para elegir combinaciones en las búsquedas, para la mejora
# contra el baseline y para proponer el modelo a registrar. Se elige siempre con validación.
METRICA_PRINCIPAL = {"p1": "mae", "p2": "f1_macro", "p3": "silueta"}
MINIMIZAR = {"p1": True, "p2": False, "p3": False}

# Hiperparámetros por defecto de cada modelo (los usa `train` cuando no se pasan con -P)
HIPERPARAMETROS = {
    "p1": {
        "hgb": {
            "features": "v1",
            "learning_rate": 0.1,
            "max_leaf_nodes": 31,
            "max_iter": ITERACIONES_HGB,
        },
        "rf": {"n_estimators": 300, "max_depth": None},
        "poisson": {"alpha": 1.0, "max_iter": 1000},
    },
    "p2": {
        "logistica": {"C": 1.0},
        "hgb": {"learning_rate": 0.1, "class_weight": None, "max_iter": ITERACIONES_HGB},
    },
    "p3": {
        "kmeans": {"k": 4},
        "ward": {"k": 4},
    },
}

# Rejillas de las búsquedas organizadas como nested runs (`-P search=true`)
REJILLAS = {
    "p1": {"hgb": REJILLA_P1},
    "p2": {"hgb": REJILLA_P2, "logistica": REJILLA_LOGISTICA},
    "p3": {"kmeans": {"k": list(RANGO_K_P3)}},
}

# Model Registry: un modelo por pregunta, con aliases (las etapas están obsoletas desde MLflow 2.9)
MODELOS_REGISTRADOS = {
    "p1": "mibici-equipo16-p1-demanda",
    "p2": "mibici-equipo16-p2-riesgo",
    "p3": "mibici-equipo16-p3-grupos",
}
ALIAS_PRODUCCION = "champion"
ALIAS_RETADOR = "challenger"
DESCRIPCIONES_MODELOS = {
    "p1": (
        "P1 · Regresión de la demanda diaria: número de viajes que inician en cada estación al día "
        "siguiente (D+1). Usuario: operador de MiBici, para dimensionar flota y personal. "
        "Métrica de decisión: MAE en viajes por estación y día."
    ),
    "p2": (
        "P2 · Clasificación del riesgo operativo por estación y franja de 3 horas: sin_bicicletas, "
        "sin_anclajes o normal. La etiqueta es un proxy (flujo neto contra alfa por capacidad), "
        "no inventario observado. Usuario: equipo de rebalanceo. Métrica de decisión: F1 macro, "
        "con atención al recall de las clases de riesgo."
    ),
    "p3": (
        "P3 · Agrupación de estaciones por patrón horario de uso (perfiles de salidas y llegadas, "
        "volumen, balance y duración). Usuario: planeación urbana. Métrica de decisión: silueta, "
        "con estabilidad medida por ARI."
    ),
}
