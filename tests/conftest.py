"""Datos sintéticos y MLflow temporal compartidos por las pruebas (sin datos reales ni red).

Fixtures disponibles para todo el equipo:

- `semilla`: semilla del proyecto (config.SEMILLA = 16).
- `estaciones_sinteticas` y `viajes_sinteticos`: catálogo y viajes con el esquema exacto que produce
  la ingesta (`mibici/dataset.py`), con casos conocidos para validación de datos.
- `sinteticos` y `conjuntos`: tablas ya transformadas de P1, P2 y P3 y sus particiones
  train/val/test, construidas con las mismas funciones que producción.
- `mlflow_sesion`: un MLflow con SQLite temporal compartido por la sesión (rápido).
- `mlflow_aislado`: un MLflow con SQLite temporal nuevo para cada prueba (sin estado compartido).

Ninguna prueba descarga datos, usa data/ ni se conecta al servidor de MLflow del equipo.
"""

import mlflow
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
def semilla():
    """Semilla del proyecto, tomada de config.py."""
    return SEMILLA


@pytest.fixture
def estaciones_sinteticas():
    """Catálogo de 6 estaciones con el esquema de `data/processed/estaciones.parquet`.

    La estación 6 está dada de baja (`activa=False`) y la 5 tiene capacidad 0, como ocurre en la
    nomenclatura real.
    """
    return pd.DataFrame(
        {
            "estacion_id": pd.array(range(1, 7), dtype="int32"),
            "nombre": pd.array([f"(GDL-00{i}) Estación {i}" for i in range(1, 7)], dtype="string"),
            "municipio": pd.array(["GDL", "GDL", "ZPN", "ZPN", "TLQ", "TLQ"], dtype="string"),
            "latitud": np.linspace(20.64, 20.74, 6),
            "longitud": np.linspace(-103.42, -103.30, 6),
            "capacidad": pd.array([15, 20, 12, 18, 0, 15], dtype="int32"),
            "activa": [True, True, True, True, True, False],
        }
    )


@pytest.fixture
def viajes_sinteticos(semilla):
    """200 viajes con el esquema de `data/processed/viajes/*.parquet`, repartidos en train/val/test.

    Son viajes válidos: sin duplicados, con origen y destino del catálogo y duración entre 2 y 60
    minutos. Cada prueba puede alterar su copia (duplicados, nulos, tipos) sin afectar a las demás.
    """
    azar = np.random.default_rng(semilla)
    n = 200
    cortes = pd.to_datetime([FECHA_INICIO, INICIO_VALIDACION, INICIO_PRUEBA])
    inicio = cortes[np.arange(n) % 3] + pd.to_timedelta(azar.integers(0, 60 * 24 * 20, n), "min")
    return pd.DataFrame(
        {
            "viaje_id": np.arange(1, n + 1, dtype="int64"),
            "inicio": inicio,
            "fin": inicio + pd.to_timedelta(azar.integers(2, 61, n), "min"),
            "origen_id": azar.integers(1, 6, n).astype("int32"),
            "destino_id": azar.integers(1, 6, n).astype("int32"),
        }
    )


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
                "demanda_media_28": 2 + estacion * 2,
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


def _preparar_mlflow(parche, ruta):
    """Apunta tracking, registro, artefactos, manifiesto y salidas del proyecto a `ruta`."""
    manifiesto = ruta / "manifiesto_datos.json"
    manifiesto.write_text('{"archivos": []}')
    uri = f"sqlite:///{ruta / 'mlflow.db'}"
    parche.chdir(ruta)  # los artefactos locales de MLflow quedan en ruta/mlruns
    parche.delenv("MLFLOW_RUN_ID", raising=False)
    for modulo in (tracking, registro, predict):
        parche.setattr(modulo, "MLFLOW_TRACKING_URI", uri)
    for modulo in (tracking, comun):
        parche.setattr(modulo, "MANIFIESTO_DATOS", manifiesto)
    parche.setattr(comun, "FIGURES_DIR", ruta / "figuras")
    parche.setattr(predict, "PROCESSED_DATA_DIR", ruta / "procesados")
    return {"ruta": ruta, "uri": uri, "manifiesto": manifiesto}


@pytest.fixture(scope="session")
def mlflow_sesion(tmp_path_factory):
    """MLflow con backend SQLite y manifiesto en un directorio temporal de la sesión."""
    with pytest.MonkeyPatch.context() as parche:
        yield _preparar_mlflow(parche, tmp_path_factory.mktemp("mlflow"))


@pytest.fixture
def mlflow_aislado(tmp_path, monkeypatch):
    """MLflow con SQLite propio de cada prueba: ninguna corrida ni registro se comparte."""
    anterior = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield _preparar_mlflow(monkeypatch, tmp_path)
    mlflow.set_tracking_uri(anterior)
