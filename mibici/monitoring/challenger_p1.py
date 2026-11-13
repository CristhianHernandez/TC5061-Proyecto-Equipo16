"""Challenger de P1 para el A/B del Entregable 8: el champion reentrenado con datos recientes.

El monitoreo y el A/B se hacen sobre P1 (`mibici-equipo16-p1-demanda`), así que el challenger debe
ser **otra versión del mismo modelo registrado**. `train_challenger.py` (PR de HH2) aplica el patrón
a P2; este módulo lo aplica a P1 con la misma regla: registrar una versión nueva, asignarle el alias
`challenger` y comprobar que el alias `champion` no se mueve.

Qué cambia frente al champion (versión 2) y qué no:

- **Igual:** algoritmo (HistGradientBoostingRegressor con pérdida Poisson), hiperparámetros
  (`learning_rate` 0.05, `max_iter` 100, `max_leaf_nodes` 63, elegidos en la validación de abr–jun
  2026), las 11 variables y su firma, semilla y pipeline de variables.
- **Distinto:** la ventana de entrenamiento. El champion vio oct-2024 a mar-2026 (`referencia`); el
  challenger ve referencia + `reciente` (abr–sep 2026), es decir, todo lo publicado antes de la
  evaluación común (octubre de 2026), incluido el historial de las 99 estaciones nuevas.

Así la comparación aísla una sola pregunta, la de Continuous Training: ¿reentrenar con datos
recientes mejora al modelo vigente? No se eligen hiperparámetros con octubre ni con ningún dato
posterior al 2026-09-30.

Antes del ajuste final se hace una **verificación previa sin octubre**: el mismo modelo entrenado
con referencia + abr–jun se evalúa en jul–sep y se compara contra el champion en esas mismas filas.

Desde la raíz, con el servidor de MLflow en marcha y los conjuntos de `make monitoring-data`:

    python -m mibici.monitoring.challenger_p1        # o: make monitoring-challenger
"""

import json

from loguru import logger
import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import typer

from mibici import tracking
from mibici.config import (
    ALIAS_PRODUCCION,
    ALIAS_RETADOR,
    FUENTE_DATOS,
    INICIO_PRUEBA,
    MANIFIESTO_MONITOREO,
    MLFLOW_TRACKING_URI,
    MODELOS_REGISTRADOS,
    SEMILLA,
)
from mibici.modeling.comun import registrar_modelo
from mibici.modeling.p1_demanda import metricas_demanda
from mibici.modeling.registro import version_de_corrida
from mibici.monitoring.datasets import (
    OBJETIVO,
    VARIABLES,
    cargar_conjuntos,
    datos_entrenamiento_challenger,
    huella,
    validar_sin_fuga,
)

app = typer.Typer()

NOMBRE = MODELOS_REGISTRADOS["p1"]
NOMBRE_CORRIDA = "p1-hgb-challenger-reentrenado-abr-sep-2026"
# Hiperparámetros del champion versión 2 (corrida 47933c5b..., docs/serving/contrato_p1.md)
HIPERPARAMETROS = {"learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 63}


def nuevo_modelo() -> HistGradientBoostingRegressor:
    """Mismo estimador y configuración que el champion; determinista por la semilla."""
    return HistGradientBoostingRegressor(
        loss="poisson", random_state=SEMILLA, early_stopping=False, **HIPERPARAMETROS
    )


def ajustar(datos: pd.DataFrame) -> HistGradientBoostingRegressor:
    """Ajusta el estimador con las 11 variables de la firma del champion."""
    return nuevo_modelo().fit(datos[VARIABLES], datos[OBJETIVO])


def partir_reciente(reciente: pd.DataFrame, corte: str = INICIO_PRUEBA):
    """Divide la ventana reciente en abr–jun (antes del corte) y jul–sep (desde el corte)."""
    fechas = pd.to_datetime(reciente["fecha"])
    return reciente.loc[fechas < corte], reciente.loc[fechas >= corte]


def verificacion_previa(conjuntos: dict[str, pd.DataFrame]) -> dict[str, float]:
    """¿Reentrenar con datos recientes ayuda fuera de muestra? Se responde sin tocar octubre.

    Compara, en jul–sep 2026, el champion (entrenado solo con la referencia) contra el mismo
    modelo entrenado con referencia + abr–jun. Ninguno de los dos ve jul–sep ni octubre.
    """
    abr_jun, jul_sep = partir_reciente(conjuntos["reciente"])
    con_recientes = pd.concat([conjuntos["referencia"], abr_jun], ignore_index=True)
    validar_sin_fuga(con_recientes, jul_sep)
    real = jul_sep[OBJETIVO]
    champion = metricas_demanda(real, ajustar(conjuntos["referencia"]).predict(jul_sep[VARIABLES]))
    reentrenado = metricas_demanda(real, ajustar(con_recientes).predict(jul_sep[VARIABLES]))
    return {
        **{f"val_jul_sep_champion_{k}": v for k, v in champion.items()},
        **{f"val_jul_sep_reentrenado_{k}": v for k, v in reentrenado.items()},
        "val_jul_sep_mejora_mae_pct": 100
        * (champion["mae"] - reentrenado["mae"])
        / champion["mae"],
        "val_jul_sep_filas": float(len(jul_sep)),
    }


def entrenar(conjuntos: dict[str, pd.DataFrame]) -> dict:
    """Verificación previa y ajuste final con referencia + reciente (todo antes de octubre)."""
    entrenamiento = datos_entrenamiento_challenger(conjuntos)  # valida que no haya fuga
    previa = verificacion_previa(conjuntos)
    modelo = ajustar(entrenamiento)
    ajuste = metricas_demanda(entrenamiento[OBJETIVO], modelo.predict(entrenamiento[VARIABLES]))
    fechas = pd.to_datetime(entrenamiento["fecha"])
    return {
        "modelo": modelo,
        "entrenamiento": entrenamiento,
        "metricas": {**previa, **{f"train_{k}": v for k, v in ajuste.items()}},
        "params": {
            "modelo": "hgb",
            "features": "v1",
            "variables": json.dumps(VARIABLES),
            **{f"hp_{k}": v for k, v in HIPERPARAMETROS.items()},
            "loss": "poisson",
            "early_stopping": False,
            "ventana_train_inicio": str(fechas.min().date()),
            "ventana_train_fin": str(fechas.max().date()),
            "n_train": len(entrenamiento),
            "estaciones_train": int(entrenamiento["estacion_id"].nunique()),
            "huella_train": huella(entrenamiento),
            "cambio_vs_champion": "misma receta; se agrega la ventana reciente (abr-sep 2026)",
        },
    }


def registrar(resultado: dict, cliente: MlflowClient | None = None) -> dict:
    """Registra la corrida y una versión nueva de P1 con alias challenger; el champion no se mueve."""
    cliente = cliente or MlflowClient()
    champion_antes = cliente.get_model_version_by_alias(NOMBRE, ALIAS_PRODUCCION)
    etiquetas = {
        "stage_of_work": "candidate",
        "model_family": "boosting",
        "rol": "challenger",
        "entregable": "8-drift-ab",
        "champion_version_base": str(champion_antes.version),
    }
    with tracking.iniciar_run("p1", NOMBRE_CORRIDA, resultado["params"], etiquetas) as run:
        tracking.registrar_metricas(resultado["metricas"])
        if MANIFIESTO_MONITOREO.exists():
            mlflow.log_artifact(str(MANIFIESTO_MONITOREO), "data")
        entrenamiento = resultado["entrenamiento"][[*VARIABLES, OBJETIVO]]
        mlflow.log_input(
            mlflow.data.from_pandas(
                entrenamiento,
                source=FUENTE_DATOS,
                name="p1-challenger-entrenamiento",
                targets=OBJETIVO,
            ),
            context="training",
        )
        uri = registrar_modelo(resultado["modelo"], entrenamiento[VARIABLES].head(5))
        run_id = run.info.run_id

    version = version_de_corrida(NOMBRE, run_id) or mlflow.register_model(uri, NOMBRE)
    if str(version.version) == str(champion_antes.version):
        raise RuntimeError("La versión challenger coincide con el champion")
    for clave, valor in {
        "rol": "challenger",
        "run_id": run_id,
        "ventana_train": f"{resultado['params']['ventana_train_inicio']}/"
        f"{resultado['params']['ventana_train_fin']}",
        "champion_version_base": str(champion_antes.version),
    }.items():
        cliente.set_model_version_tag(NOMBRE, version.version, clave, valor)
    cliente.set_registered_model_alias(NOMBRE, ALIAS_RETADOR, version.version)

    champion_despues = cliente.get_model_version_by_alias(NOMBRE, ALIAS_PRODUCCION)
    if str(champion_despues.version) != str(champion_antes.version):
        raise RuntimeError("El alias champion cambió durante el registro del challenger")
    return {
        "nombre": NOMBRE,
        "run_id": run_id,
        "champion": str(champion_antes.version),
        "challenger": str(version.version),
    }


@app.command()
def main():
    """Entrena y registra el challenger de P1 (referencia + reciente) sin mover el champion."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    resultado = entrenar(cargar_conjuntos())
    m = resultado["metricas"]
    logger.info(
        "Verificación previa jul–sep 2026: MAE champion {:.4f} → reentrenado {:.4f} ({:+.2f} %)",
        m["val_jul_sep_champion_mae"],
        m["val_jul_sep_reentrenado_mae"],
        m["val_jul_sep_mejora_mae_pct"],
    )
    if not np.isfinite(m["train_mae"]):
        raise RuntimeError("El ajuste final produjo métricas no finitas")
    registro = registrar(resultado)
    logger.success(
        "{}: champion v{} conservado; challenger v{} (corrida {})",
        registro["nombre"],
        registro["champion"],
        registro["challenger"],
        registro["run_id"],
    )


if __name__ == "__main__":
    app()
