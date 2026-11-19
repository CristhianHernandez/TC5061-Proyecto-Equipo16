"""Inicializa el Model Registry del stack de Docker Compose con el champion de P1 (Entregable 7).

Es el servicio `model-init` de docker-compose.yml: corre una vez, después de que MLflow está sano y
antes de que arranque la API, que carga `models:/mibici-equipo16-p1-demanda@champion` de ese
registry. Decide así:

1. **Copia del champion en la imagen** (`make export-model` antes del build; queda en
   /app/model_bundle): la registra tal cual (mismos archivos, firma y ejemplo de entrada) con el
   linaje de su registry de origen. Si ya está registrada (misma huella), no hace nada, así que
   recrear los contenedores no crea versiones nuevas: el volumen de MLflow conserva el estado.
2. **Sin copia** (clon limpio, runner de CI): si el registry está vacío, registra un **modelo de
   prueba** determinista con la misma firma de 11 variables, etiquetado `origen=modelo-de-prueba`,
   para que la API arranque y prediga. Nunca reemplaza a un champion real.

En ambos casos verifica que la versión registrada carga y predice antes de asignarle `champion`; la
versión anterior conserva el alias `challenger` para un rollback.

    python -m mibici.serving.inicializar      # MLFLOW_TRACKING_URI apunta al servidor del stack
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import Model, infer_signature
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import typer

from mibici.config import (
    ALIAS_PRODUCCION,
    ALIAS_RETADOR,
    EQUIPO,
    EXPERIMENTO_MLFLOW,
    METADATOS_MODELO,
    MLFLOW_TRACKING_URI,
    MODEL_BUNDLE_INICIAL,
    MODELOS_REGISTRADOS,
    ORIGEN_MODELO_PRUEBA,
    SEMILLA,
)
from mibici.serving.model_loader import TIPOS_FIRMA
from mibici.serving.schemas import FIRMA_P1

app = typer.Typer()

NOMBRE = MODELOS_REGISTRADOS["p1"]
COLUMNAS = [nombre for nombre, _, _ in FIRMA_P1]


def firma_de(ruta: Path) -> list[tuple[str, str, bool]]:
    """Firma de entrada de una carpeta MLflow como (nombre, tipo, obligatoria)."""
    esquema = Model.load(str(ruta)).get_input_schema()
    if esquema is None or not esquema.has_input_names():
        return []  # sin firma o sin columnas con nombre: no puede cumplir el contrato tabular
    return [(c.name, getattr(c.type, "name", str(c.type)), c.required) for c in esquema.inputs]


def validar_firma(ruta: Path) -> None:
    """La API solo sirve modelos con la firma de P1; cualquier otro contrato se rechaza aquí."""
    firma = firma_de(ruta)
    if firma != list(FIRMA_P1):
        raise ValueError(f"La firma de {ruta} no coincide con el contrato de P1: {firma}")


def datos_de_prueba(n: int = 400, semilla: int = SEMILLA) -> tuple[pd.DataFrame, np.ndarray]:
    """Variables de P1 sintéticas y coherentes (calendario real, rezagos y estaciones plausibles).

    Los tipos salen de la firma del champion (int32, int64 o float64) y los primeros días no tienen
    rezagos, como al inicio del historial de una estación, para que la firma los marque opcionales.
    """
    azar = np.random.default_rng(semilla)
    fechas = pd.date_range("2025-01-01", periods=n)
    media_7 = azar.uniform(5, 60, n)
    valores = {
        "dia_semana": fechas.dayofweek,
        "mes": fechas.month,
        "es_fin_semana": fechas.dayofweek >= 5,
        "dia_anio": fechas.dayofyear,
        "es_festivo": np.zeros(n),
        "demanda_lag_1": azar.poisson(media_7),
        "demanda_lag_7": azar.poisson(media_7),
        "demanda_media_7": media_7,
        "capacidad": azar.choice([15, 19, 23, 39], n),
        "latitud": azar.uniform(20.64, 20.74, n),
        "longitud": azar.uniform(-103.42, -103.30, n),
    }
    datos = pd.DataFrame(
        {
            nombre: pd.Series(valores[nombre]).astype(TIPOS_FIRMA[tipo])
            for nombre, tipo, _ in FIRMA_P1
        }
    )
    datos.loc[:6, [nombre for nombre, _, requerida in FIRMA_P1 if not requerida]] = np.nan
    demanda = azar.poisson(0.8 * media_7 + 0.1 * datos["capacidad"].to_numpy())
    return datos, demanda


def guardar_modelo_de_prueba(destino: Path) -> Path:
    """Entrena y guarda un HGB Poisson pequeño con la firma exacta de P1 (determinista)."""
    from skops.io import dumps, get_untrusted_types

    datos, demanda = datos_de_prueba()
    modelo = HistGradientBoostingRegressor(
        loss="poisson", max_iter=50, random_state=SEMILLA, early_stopping=False
    ).fit(datos, demanda)
    ejemplo = datos.head(5)
    mlflow.sklearn.save_model(
        modelo,
        str(destino),
        signature=infer_signature(ejemplo, modelo.predict(ejemplo)),
        input_example=ejemplo,
        serialization_format="skops",
        skops_trusted_types=get_untrusted_types(data=dumps(modelo)),
    )
    validar_firma(destino)
    return destino


def huella(ruta: Path) -> str:
    """Identificador estable de una copia exportada: el model_uuid de su MLmodel."""
    return str(Model.load(str(ruta)).model_uuid)


def linaje_de(ruta: Path) -> dict[str, str]:
    """Linaje que `make export-model` guardó junto a la copia (versión y corrida de origen)."""
    archivo = ruta / METADATOS_MODELO
    if not archivo.is_file():
        return {}
    metadatos = json.loads(archivo.read_text(encoding="utf-8"))
    linaje = metadatos.get("linaje", {})
    return {
        "version_origen": str(metadatos.get("version", "")),
        "run_id_origen": str(metadatos.get("run_id", "")),
        "git_commit": str(linaje.get("git_commit", "")),
        "version_datos": str(linaje.get("version_datos", "")),
        "run_name_origen": str(linaje.get("run_name", "")),
    }


def champion_actual(cliente: MlflowClient, nombre: str):
    """Versión con alias champion, o None si el modelo o el alias aún no existen."""
    try:
        return cliente.get_model_version_by_alias(nombre, ALIAS_PRODUCCION)
    except MlflowException:
        return None


def registrar(ruta: Path, nombre: str, etiquetas: dict[str, str]) -> dict:
    """Registra la carpeta como versión nueva, verifica que predice y le asigna champion."""
    cliente = MlflowClient()
    anterior = champion_actual(cliente, nombre)
    mlflow.set_experiment(EXPERIMENTO_MLFLOW)
    with mlflow.start_run(run_name=f"p1-champion-compose-{etiquetas['origen']}") as run:
        mlflow.set_tags({"equipo": EQUIPO, "question": "p1", "uso": "inicializacion-compose"})
        mlflow.set_tags(etiquetas)
        mlflow.log_artifacts(str(ruta), "model")
    version = mlflow.register_model(f"runs:/{run.info.run_id}/model", nombre)
    for clave, valor in etiquetas.items():
        cliente.set_model_version_tag(nombre, version.version, clave, valor)
    cliente.update_model_version(
        nombre,
        version.version,
        description=(
            f"Champion de P1 servido por el stack de Docker Compose (origen: {etiquetas['origen']}"
            + (
                f", versión {etiquetas['version_origen']} del registry del equipo"
                if etiquetas.get("version_origen")
                else ""
            )
            + ")."
        ),
    )
    modelo = mlflow.pyfunc.load_model(f"models:/{nombre}/{version.version}")
    modelo.predict(modelo.input_example)  # falla aquí, no en la API, si la copia está dañada
    if anterior is not None:
        cliente.set_registered_model_alias(nombre, ALIAS_RETADOR, anterior.version)
    cliente.set_registered_model_alias(nombre, ALIAS_PRODUCCION, version.version)
    return {
        "accion": "registrado",
        "version": str(version.version),
        "challenger": str(anterior.version) if anterior else None,
        "run_id": run.info.run_id,
        **etiquetas,
    }


def inicializar(bundle: Path | None = None, nombre: str = NOMBRE) -> dict:
    """Deja un champion de P1 en el registry del stack; ver el docstring del módulo."""
    bundle = MODEL_BUNDLE_INICIAL if bundle is None else Path(bundle)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    cliente = MlflowClient()
    actual = champion_actual(cliente, nombre)
    if (bundle / "MLmodel").is_file():
        validar_firma(bundle)
        nueva = huella(bundle)
        if actual is not None and actual.tags.get("huella_modelo") == nueva:
            return {"accion": "sin_cambios", "version": str(actual.version), "origen": "bundle"}
        return registrar(
            bundle, nombre, {"origen": "bundle", "huella_modelo": nueva, **linaje_de(bundle)}
        )
    if actual is not None:
        # Sin copia en la imagen se conserva lo registrado: nunca se degrada a un modelo de prueba
        return {
            "accion": "sin_cambios",
            "version": str(actual.version),
            "origen": actual.tags.get("origen", "desconocido"),
        }
    logger.warning(f"No hay copia del champion en {bundle}; se registra un modelo de prueba")
    with TemporaryDirectory() as temporal:
        ruta = guardar_modelo_de_prueba(Path(temporal) / "modelo")
        return registrar(
            ruta, nombre, {"origen": "modelo-de-prueba", "huella_modelo": ORIGEN_MODELO_PRUEBA}
        )


@app.command()
def main(
    bundle: str = typer.Option(str(MODEL_BUNDLE_INICIAL), help="Copia exportada del champion"),
):
    """Registra el champion de P1 en el MLflow del stack (idempotente)."""
    resultado = inicializar(Path(bundle))
    logger.success(f"{NOMBRE}@{ALIAS_PRODUCCION}: {resultado}")
    typer.echo(json.dumps(resultado, ensure_ascii=False))


if __name__ == "__main__":
    app()
