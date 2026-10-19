"""Selección y registro compartidos de los experimentos P1, P2 y P3."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from loguru import logger
import matplotlib
import mlflow
import pandas as pd
from sklearn.model_selection import ParameterGrid

from mibici import tracking
from mibici.config import FIGURES_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def seleccionar_modelo(
    entrenamiento, validacion, variables, objetivo, construir, rejilla, evaluar, maximizar=False
):
    """Elige hiperparámetros de P1/P2 con validación, sin recibir prueba ni reajustar con ella."""
    mejor = None
    filas = []
    for parametros in ParameterGrid(rejilla):
        modelo = construir(parametros)
        modelo.fit(entrenamiento[variables], entrenamiento[objetivo])
        valor = evaluar(validacion[objetivo], modelo.predict(validacion[variables]))
        filas.append({**parametros, "val_objetivo": valor})
        if mejor is None or (valor > mejor[0] if maximizar else valor < mejor[0]):
            mejor = (valor, modelo, parametros)
    return mejor[1], mejor[2], pd.DataFrame(filas)


def registrar_resultado(pregunta, nombre, parametros, metricas, figuras, tablas=None):
    """Registra métricas y artefactos agregados de P1/P2/P3, y devuelve el identificador del run."""
    with tracking.iniciar_run(pregunta, nombre, parametros) as run:
        tracking.registrar_metricas(metricas)
        with TemporaryDirectory() as temporal:
            ruta = Path(temporal)
            (ruta / "metricas.json").write_text(json.dumps(metricas, indent=2), encoding="utf-8")
            for archivo, tabla in (tablas or {}).items():
                tabla.to_csv(ruta / f"{archivo}.csv", index=False)
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            for archivo, figura in figuras.items():
                figura.savefig(ruta / f"{archivo}.png", dpi=120, bbox_inches="tight")
                figura.savefig(
                    FIGURES_DIR / f"{nombre}-{archivo}.png", dpi=120, bbox_inches="tight"
                )
                plt.close(figura)
            for archivo in ruta.iterdir():
                mlflow.log_artifact(str(archivo))
        principal = {"P1": "mae", "P2": "f1_macro", "P3": "silueta"}[pregunta]
        # Leer MLflow confirma que el resumen corresponde a las métricas persistidas.
        registradas = mlflow.get_run(run.info.run_id).data.metrics
        logger.success(
            "{} | val_{}={:.6f} | test_{}={:.6f}",
            nombre,
            principal,
            registradas.get(f"val_{principal}", float("nan")),
            principal,
            registradas.get(f"test_{principal}", float("nan")),
        )
        return run.info.run_id


def parametros_conjuntos(modelo, variables, conjuntos):
    """Describe variables y tamaños de partición de P1/P2/P3 para auditoría sin datos personales."""
    return {
        "modelo": modelo,
        "variables": json.dumps(list(variables)),
        "ajuste": "solo train; sin reajuste train+val",
        **{f"n_{nombre}": len(datos) for nombre, datos in conjuntos.items()},
    }
