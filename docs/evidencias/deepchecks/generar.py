"""Reproduce métricas agregadas usando exactamente los fixtures sintéticos de CH1.

Desde la raíz: python docs/evidencias/deepchecks/generar.py
No carga data/, no descarga datos ni entrena modelos. No guarda muestras ni displays.
"""

import importlib.metadata
import json
from pathlib import Path
import platform
import runpy

import pandas as pd

from mibici import config
from mibici.validation.deepchecks import (
    evaluar_drift,
    resumir_resultados,
    validar_estaciones,
    validar_particiones,
    validar_viajes,
)


def generar():
    # __wrapped__ ejecuta el proveedor de datos del fixture, sin iniciar sesiones MLflow.
    fixtures = runpy.run_path(str(config.PROJ_ROOT / "tests" / "conftest.py"))
    estaciones = fixtures["estaciones_sinteticas"].__wrapped__()
    viajes = fixtures["viajes_sinteticos"].__wrapped__(config.SEMILLA)
    sinteticos = fixtures["sinteticos"].__wrapped__()
    conjuntos = fixtures["conjuntos"].__wrapped__(sinteticos)
    casos = {
        "viajes_validos": validar_viajes(viajes, estaciones),
        "catalogo_valido": validar_estaciones(estaciones),
        "catalogo_p2_filtrado": validar_estaciones(
            estaciones.loc[estaciones.capacidad.gt(0)], para_p2=True
        ),
        "catalogo_p2_sin_filtrar": validar_estaciones(estaciones, para_p2=True),
    }
    anomalos = viajes.convert_dtypes()
    anomalos.loc[0, "destino_id"] = pd.NA
    anomalos.loc[1, "origen_id"] = estaciones.estacion_id.max() + 1
    anomalos.loc[2, "fin"] = anomalos.loc[2, "inicio"] - pd.Timedelta(minutes=1)
    anomalos = pd.concat([anomalos, anomalos.iloc[3:4]], ignore_index=True)
    casos["viajes_con_anomalias"] = validar_viajes(anomalos, estaciones)
    tipos_invalidos = viajes.astype({"destino_id": "object"})
    tipos_invalidos.loc[0, "destino_id"] = "tipo_invalido_sintetico"
    casos["tipos_invalidos"] = validar_viajes(tipos_invalidos, estaciones)
    temporal = {
        "p1": validar_particiones(conjuntos["p1"]),
        "p2": validar_particiones(conjuntos["p2"], "fecha_franja"),
    }
    fuga = {k: df.copy() for k, df in conjuntos["p1"].items()}
    fuga["train"] = pd.concat([fuga["train"], fuga["test"].iloc[:1]])
    temporal["fuga_intencional"] = validar_particiones(fuga)
    casos["temporal"] = temporal
    columnas = ["demanda_lag_1", "demanda_media_7"]
    train = conjuntos["p1"]["train"]
    test = conjuntos["p1"]["test"]
    desplazado = test.copy()
    desplazado[columnas] += 10000
    casos["drift"] = {
        "control_identico": evaluar_drift(train, train.copy(), columnas),
        "train_vs_test": evaluar_drift(train, test, columnas),
        "desplazamiento_intencional": evaluar_drift(train, desplazado, columnas),
    }
    return {
        "base_main": "3f1f54d069dd7d2f992987cf59c63d9f35eb92c9",
        "origen": "fixtures sintéticos compartidos de tests/conftest.py; sin PII",
        "python": platform.python_version(),
        "versiones": {
            p: importlib.metadata.version(p)
            for p in ("deepchecks", "numpy", "pandas", "scikit-learn", "pytest", "ruff")
        },
        "cortes_config": {
            k: getattr(config, k)
            for k in ("FECHA_INICIO", "INICIO_VALIDACION", "INICIO_PRUEBA", "FECHA_FIN", "SEMILLA")
        },
        "filas": {"viajes": len(viajes), "estaciones": len(estaciones)},
        "casos": {nombre: resumir_resultados(checks) for nombre, checks in casos.items()},
    }


if __name__ == "__main__":
    destino = Path(__file__).with_name("resultados.json")
    destino.write_text(
        json.dumps(generar(), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("Evidencia agregada escrita en docs/evidencias/deepchecks/resultados.json")
