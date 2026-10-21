"""Datos sintéticos y MLflow temporal compartidos por las pruebas (sin datos reales ni red)."""

import numpy as np
import pandas as pd
import pytest

from mibici import features, tracking
from mibici.config import FECHA_INICIO, HORAS_FRANJA, INICIO_PRUEBA, INICIO_VALIDACION, SEMILLA
from mibici.modeling import comun, predict, registro
from mibici.modeling import p1_demanda as p1
from mibici.modeling import p2_riesgo as p2
from mibici.modeling import p3_clusters as p3


@pytest.fixture(scope="session")
def sinteticos():
    """Construye datos pequeños con todas las clases, periodos y perfiles de P1/P2/P3."""
    azar = np.random.default_rng(SEMILLA)
    estaciones = pd.DataFrame(
        {
            "estacion_id": range(12),
            "capacidad": [20] * 12,
            "latitud": np.linspace(20, 21, 12),
            "longitud": np.linspace(-104, -103, 12),
            "municipio": np.repeat(["GDL", "ZPN", "TLQ"], 4),
        }
    )
    fechas = [
        pd.date_range(corte, periods=12)
        for corte in (FECHA_INICIO, INICIO_VALIDACION, INICIO_PRUEBA)
    ]
    demanda = pd.DataFrame(
        [
            {
                "estacion_id": estacion,
                "fecha": fecha,
                "demanda": 1 + estacion * 2 + i % 3,
                "demanda_lag_1": 1 + estacion * 2,
                "demanda_lag_7": np.nan if i == 0 else 2 + estacion * 2,
                "demanda_media_7": 2 + estacion * 2,
            }
            for periodo in fechas
            for i, fecha in enumerate(periodo)
            for estacion in range(12)
        ]
    )
    demanda = features.agregar_features_calendario(demanda)
    riesgo = demanda[["estacion_id", "fecha", "dia_semana"]].rename(
        columns={"fecha": "fecha_franja"}
    )
    valores = np.tile([-12.0, 0.0, 12.0], len(riesgo) // 3)
    riesgo = riesgo.assign(
        capacidad=20,
        hora=np.arange(len(riesgo)) % 8 * HORAS_FRANJA,
        flujo_lag_1=valores,
        flujo_lag_56=valores,
    )
    riesgo["riesgo"] = p2.predecir_persistencia(riesgo)
    perfiles = estaciones[["estacion_id"]].copy()
    for tipo in ("salidas", "llegadas"):
        for dia in ("laboral", "fin_semana"):
            matriz = azar.uniform(0, 0.1, (12, 24))
            for i in range(12):
                matriz[i, (i // 4) * 8] += 2
            matriz /= matriz.sum(axis=1, keepdims=True)
            for h in range(24):
                perfiles[f"{tipo}_{dia}_{h:02d}"] = matriz[:, h]
    perfiles = perfiles.assign(
        total_salidas=np.arange(12) * 10 + 10,
        total_llegadas=np.arange(12) * 12 + 10,
        duracion_media_min=[np.nan] + [15.0] * 11,
    )
    perfiles = perfiles.copy()
    perfiles["volumen_total"] = perfiles["total_salidas"] + perfiles["total_llegadas"]
    perfiles["balance_neto"] = perfiles["total_salidas"] - perfiles["total_llegadas"]
    return estaciones, demanda, riesgo, perfiles


@pytest.fixture(scope="session")
def conjuntos(sinteticos):
    """Conjuntos train/val/test de las tres preguntas, construidos como en producción."""
    estaciones, demanda, riesgo, perfiles = sinteticos
    return {
        "p1": p1.construir_conjuntos(demanda, estaciones),
        "p2": p2.construir_conjuntos(riesgo, estaciones),
        "p3": p3.construir_conjuntos(perfiles, perfiles.copy(), perfiles.copy(), estaciones),
    }


@pytest.fixture(scope="session")
def mlflow_sesion(tmp_path_factory):
    """MLflow con backend SQLite y manifiesto en un directorio temporal de la sesión."""
    ruta = tmp_path_factory.mktemp("mlflow")
    manifiesto = ruta / "manifiesto_datos.json"
    manifiesto.write_text('{"archivos": []}')
    uri = f"sqlite:///{ruta / 'mlflow.db'}"
    with pytest.MonkeyPatch.context() as parche:
        parche.chdir(ruta)
        parche.delenv("MLFLOW_RUN_ID", raising=False)
        for modulo in (tracking, registro, predict):
            parche.setattr(modulo, "MLFLOW_TRACKING_URI", uri)
        for modulo in (tracking, comun):
            parche.setattr(modulo, "MANIFIESTO_DATOS", manifiesto)
        parche.setattr(comun, "FIGURES_DIR", ruta / "figuras")
        parche.setattr(predict, "PROCESSED_DATA_DIR", ruta / "procesados")
        yield {"ruta": ruta, "uri": uri, "manifiesto": manifiesto}
