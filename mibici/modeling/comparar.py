"""Entrada `compare` del MLflow Project: tablas comparativas por pregunta contra el baseline.

Uso:

    mlflow run . -e compare --env-manager uv --experiment-name mibici-equipo16

Lee el experimento del equipo y escribe `reports/tabla_comparativa_p1.csv`, `p2` y `p3` (fuera de
Git, para el reporte). Cada fila es una corrida comparable de la pregunta: el baseline, las
candidatas y la corrida padre de cada búsqueda (las hijas solo tienen validación y quedan en
`metrics/busqueda.csv` de su padre). Se excluyen los análisis aparte (`stage_of_work=analysis`),
porque cambian las etiquetas y no son comparables, y las corridas de otra versión de los datos.
Si una corrida se repitió con el mismo nombre se conserva la más reciente.
"""

import os

from loguru import logger
import mlflow
from mlflow import MlflowClient
import pandas as pd
import typer

from mibici import tracking
from mibici.config import (
    ALIAS_PRODUCCION,
    ALIAS_RETADOR,
    BASELINES,
    EXPERIMENTO_MLFLOW,
    MANIFIESTO_DATOS,
    METRICA_PRINCIPAL,
    MINIMIZAR,
    MLFLOW_TRACKING_URI,
    MODELOS_REGISTRADOS,
    PREGUNTAS_PROYECTO,
    TABLAS_COMPARATIVAS,
)
from mibici.modeling.comun import PREFIJO_HIPERPARAMETRO

app = typer.Typer()

ETAPAS_COMPARABLES = ("baseline", "search", "candidate")
# Métricas de desempeño y estabilidad que muestra la tabla de cada pregunta (además del costo)
METRICAS_TABLA = {
    "p1": [
        "val_mae",
        "test_mae",
        "val_rmse",
        "test_rmse",
        "val_desviacion_poisson",
        "test_desviacion_poisson",
        "test_mae_std_mensual",
    ],
    "p2": [
        "val_f1_macro",
        "test_f1_macro",
        "val_recall_sin_bicicletas",
        "val_recall_sin_anclajes",
        "test_recall_sin_bicicletas",
        "test_recall_sin_anclajes",
        "val_ap_macro_riesgo",
        "test_ap_macro_riesgo",
        "val_balanced_accuracy",
        "test_balanced_accuracy",
        "test_f1_macro_std_mensual",
    ],
    "p3": [
        "train_silueta",
        "val_silueta",
        "test_silueta",
        "train_davies_bouldin",
        "train_calinski_harabasz",
        "train_ari_semillas",
        "val_ari_periodo",
        "test_ari_periodo",
    ],
}
COSTO = ["train_duracion_s", "test_latencia_ms_por_1000"]


def hiperparametros_clave(params: dict) -> str:
    """Hiperparámetros efectivos de la corrida (los registrados con el prefijo hp_)."""
    efectivos = {
        k.removeprefix(PREFIJO_HIPERPARAMETRO): v
        for k, v in sorted(params.items())
        if k.startswith(PREFIJO_HIPERPARAMETRO)
    }
    return ", ".join(f"{k}={v}" for k, v in efectivos.items())


def aliases(question: str) -> dict[str, str]:
    """Corrida de origen de cada alias del modelo registrado de la pregunta (si existe)."""
    cliente = MlflowClient()
    resultado = {}
    for alias in (ALIAS_PRODUCCION, ALIAS_RETADOR):
        try:
            version = cliente.get_model_version_by_alias(MODELOS_REGISTRADOS[question], alias)
        except mlflow.exceptions.MlflowException:
            continue
        resultado[version.run_id] = f"{alias} (v{version.version})"
    return resultado


def corridas_comparables(question: str, version: str | None = None) -> list:
    """Corridas terminadas de la pregunta que entran a la tabla (ver docstring del módulo)."""
    cliente = MlflowClient()
    experimento = cliente.get_experiment_by_name(EXPERIMENTO_MLFLOW)
    if experimento is None:
        raise ValueError(f"No existe el experimento {EXPERIMENTO_MLFLOW}")
    filtro = f"tags.question = '{question}' and attributes.status = 'FINISHED'"
    if version:
        filtro += f" and tags.version_datos = '{version}'"
    runs = cliente.search_runs(
        [experimento.experiment_id],
        filter_string=filtro,
        order_by=["attributes.start_time DESC"],
        max_results=5000,
    )
    vistos, elegidas = set(), []
    for run in runs:
        etiquetas = run.data.tags
        if etiquetas.get("stage_of_work") not in ETAPAS_COMPARABLES:
            continue
        if etiquetas.get("search_role") == "child" or run.info.run_name in vistos:
            continue
        vistos.add(run.info.run_name)
        elegidas.append(run)
    return elegidas


def tabla_comparativa(question: str, version: str | None = None) -> pd.DataFrame:
    """Construye la tabla de una pregunta: modelo, hiperparámetros, métricas, mejora y costo."""
    principal = METRICA_PRINCIPAL[question]
    runs = corridas_comparables(question, version)
    produccion = aliases(question)
    filas = []
    for run in runs:
        metricas = run.data.metrics
        filas.append(
            {
                "corrida": run.info.run_name,
                "modelo": run.data.params.get("modelo", ""),
                "model_family": run.data.tags.get("model_family", ""),
                "stage_of_work": run.data.tags.get("stage_of_work", ""),
                "hiperparametros": hiperparametros_clave(run.data.params),
                **{m: metricas.get(m) for m in METRICAS_TABLA[question]},
                f"test_mejora_{principal}_pct": metricas.get(f"test_mejora_{principal}_pct"),
                **{m: metricas.get(m) for m in COSTO},
                "alias": produccion.get(run.info.run_id, ""),
                "run_id": run.info.run_id,
                "git_commit": run.data.tags.get("git_commit", ""),
                "version_datos": run.data.tags.get("version_datos", ""),
            }
        )
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla
    # Mejora absoluta en la métrica principal de prueba contra el baseline de referencia
    # (positiva = mejor), útil cuando el baseline está cerca de cero (silueta de P3).
    base = tabla.loc[tabla["modelo"] == BASELINES[question], f"test_{principal}"]
    if not base.empty and pd.notna(base.iloc[0]):
        signo = -1 if MINIMIZAR[question] else 1
        tabla[f"test_mejora_{principal}_abs"] = signo * (tabla[f"test_{principal}"] - base.iloc[0])
    orden = {"baseline": 0, "search": 1, "candidate": 2}
    clave_val = f"val_{principal}" if f"val_{principal}" in tabla else f"train_{principal}"
    return (
        tabla.assign(_etapa=tabla["stage_of_work"].map(orden))
        .sort_values(["_etapa", clave_val], ascending=[True, MINIMIZAR[question]])
        .drop(columns="_etapa")
        .reset_index(drop=True)
    )


def comparar(version: str | None = None) -> dict:
    """Escribe las tres tablas comparativas y devuelve sus rutas por pregunta."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    version = version or tracking.version_datos(MANIFIESTO_DATOS)
    rutas = {}
    for question in PREGUNTAS_PROYECTO:
        tabla = tabla_comparativa(question, version)
        ruta = TABLAS_COMPARATIVAS[question]
        ruta.parent.mkdir(parents=True, exist_ok=True)
        tabla.to_csv(ruta, index=False)
        logger.info(f"{question}: {len(tabla)} corridas -> {ruta}")
        rutas[question] = ruta
    return rutas


@app.command()
def main():
    """Genera las tablas comparativas de P1, P2 y P3 a partir del experimento del equipo."""
    # `mlflow run` crea una corrida para este comando: se reanuda para guardar las tablas.
    propio = os.environ.pop("MLFLOW_RUN_ID", None)
    version = tracking.version_datos(MANIFIESTO_DATOS)
    rutas = comparar(version)
    if propio is not None:
        with mlflow.start_run(run_id=propio):
            mlflow.set_tags({"mlflow.runName": "comparacion-p1-p2-p3", "uso": "comparacion"})
            mlflow.log_param("version_datos", version)
            for ruta in rutas.values():
                mlflow.log_artifact(str(ruta), "metrics")
    typer.echo("\n".join(str(r) for r in rutas.values()))


if __name__ == "__main__":
    app()
