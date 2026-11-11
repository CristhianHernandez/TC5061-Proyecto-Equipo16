"""Conjuntos de referencia, reciente y actual para el drift y el A/B testing de P1 (Entregable 8).

Contrato común para Evidently (LH3), el challenger (HH2) y la comparación A/B (DH4); el detalle está
en docs/monitoring/contrato_datasets.md. Todos los conjuntos tienen la misma granularidad que el
champion de P1 (una fila por estación y día) y las mismas columnas: `estacion_id`, `fecha`, las 11
variables del modelo y la demanda observada.

- referencia: exactamente el train con el que se entrenó el champion (oct-2024 a mar-2026);
- reciente: val + test (abr a sep-2026), ya etiquetado; el challenger puede entrenar con él;
- actual: octubre de 2026, publicado después del entrenamiento; drift contra la referencia y
  evaluación común del A/B. No se usa para entrenar ni para elegir hiperparámetros.

Desde la raíz, con los datos de entrenamiento ya procesados (`make data`) y el servidor de MLflow
en marcha para anotar la versión del champion:

    python -m mibici.monitoring.datasets           # o: make monitoring-data
    python -m mibici.monitoring.datasets --sin-registry

Descarga el mes actual y su catálogo de estaciones, guarda los tres conjuntos en
data/processed/monitoreo/ (fuera de Git) y escribe references/manifiesto_monitoreo.json.
"""

import hashlib
from itertools import pairwise
import json
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from mibici.config import (
    ALIAS_PRODUCCION,
    ESTACIONES_MONITOREO_PARQUET,
    FECHA_INICIO,
    FUENTE_DATOS,
    INICIO_VALIDACION,
    MANIFIESTO_DATOS,
    MANIFIESTO_MONITOREO,
    MLFLOW_TRACKING_URI,
    MODELOS_REGISTRADOS,
    MONITOREO_DIR,
    PERIODOS_MONITOREO,
    RAW_DATA_DIR,
    SEMILLA,
    URL_NOMENCLATURA,
    URL_NOMENCLATURA_MONITOREO,
    URLS_VIAJES_MONITOREO,
    VIAJES_DIR,
    VIAJES_MONITOREO_DIR,
)
from mibici.dataset import (
    calcular_sha256,
    cargar_estaciones,
    descargar_archivo,
    procesar_estaciones,
    procesar_viajes,
)
from mibici.features import limpiar_viajes, preparar_datos_p1
from mibici.modeling import p1_demanda as p1
from mibici.tracking import version_datos

app = typer.Typer()

VENTANAS = ("referencia", "reciente", "actual")
COLUMNAS_ID = ["estacion_id", "fecha"]
VARIABLES = list(p1.VARIABLES)  # firma del champion features-v1, en su orden
OBJETIVO = p1.OBJETIVO
# Para Evidently: indicadores y calendario discretos como categóricas; el resto, numéricas
CATEGORICAS = ["dia_semana", "mes", "es_fin_semana", "es_festivo"]
NUMERICAS = [v for v in VARIABLES if v not in CATEGORICAS]
COLUMNAS = [*COLUMNAS_ID, *VARIABLES, OBJETIVO]

REGLAS_SIN_FUGA = [
    "Las tres ventanas son consecutivas y no se traslapan: referencia < reciente < actual.",
    "La referencia es exactamente el train del champion (mismos cortes y mismo catálogo).",
    (
        "El challenger solo puede entrenar o elegir hiperparámetros con fechas anteriores al "
        "inicio de la ventana actual (referencia + reciente)."
    ),
    "La ventana actual es la evaluación común del A/B: ambos modelos predicen las mismas filas.",
    (
        "Los rezagos de la ventana actual usan solo días anteriores a la fecha de cada fila; el 1 "
        "de octubre toma la demanda del 30 de septiembre, que ya era conocida."
    ),
    "El A/B fija versiones numéricas del registry; los aliases no se mueven antes de decidir.",
]


def validar_periodos(periodos: dict | None = None) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Comprueba que las ventanas existan, estén en orden y no se traslapen; devuelve fechas.

    La referencia debe coincidir con el train del champion (FECHA_INICIO al día anterior a
    INICIO_VALIDACION); si no, el drift no compararía contra lo que el modelo realmente vio.
    """
    periodos = PERIODOS_MONITOREO if periodos is None else periodos
    if tuple(periodos) != VENTANAS:
        raise ValueError(f"Se esperaban las ventanas {VENTANAS} en ese orden")
    fechas = {}
    for ventana, (inicio, fin) in periodos.items():
        inicio, fin = pd.Timestamp(inicio), pd.Timestamp(fin)
        if inicio > fin:
            raise ValueError(f"{ventana}: el inicio {inicio.date()} es posterior al fin")
        fechas[ventana] = (inicio, fin)
    for anterior, siguiente in pairwise(VENTANAS):
        if fechas[anterior][1] >= fechas[siguiente][0]:
            raise ValueError(
                f"Fuga temporal: {anterior} termina el {fechas[anterior][1].date()} y {siguiente} "
                f"empieza el {fechas[siguiente][0].date()}"
            )
    entrenamiento = (
        pd.Timestamp(FECHA_INICIO),
        pd.Timestamp(INICIO_VALIDACION) - pd.Timedelta(days=1),
    )
    if fechas["referencia"] != entrenamiento:
        raise ValueError("La referencia debe ser el periodo de entrenamiento del champion")
    return fechas


def recortar(datos: pd.DataFrame, inicio, fin) -> pd.DataFrame:
    """Filas con `fecha` entre inicio y fin (inclusive), con las columnas del contrato y ordenadas."""
    fechas = pd.to_datetime(datos["fecha"])
    filas = datos.loc[fechas.between(pd.Timestamp(inicio), pd.Timestamp(fin)), COLUMNAS]
    return filas.sort_values(COLUMNAS_ID).reset_index(drop=True)


def validar_sin_fuga(entrenamiento: pd.DataFrame, evaluacion: pd.DataFrame) -> None:
    """Falla si el entrenamiento tiene fechas iguales o posteriores al inicio de la evaluación."""
    if entrenamiento.empty or evaluacion.empty:
        raise ValueError("Los conjuntos de entrenamiento y evaluación no pueden estar vacíos")
    ultimo = pd.to_datetime(entrenamiento["fecha"]).max()
    primero = pd.to_datetime(evaluacion["fecha"]).min()
    if ultimo >= primero:
        raise ValueError(
            f"Fuga temporal: el entrenamiento llega al {ultimo.date()} y la evaluación empieza el "
            f"{primero.date()}"
        )


def construir_referencia_y_reciente(conjuntos: dict | None = None, periodos: dict | None = None):
    """Referencia (train del champion) y reciente (val + test) desde la partición original de P1."""
    fechas = validar_periodos(periodos)
    conjuntos = p1.preparar_conjuntos() if conjuntos is None else conjuntos
    historia = pd.concat([conjuntos["train"], conjuntos["val"], conjuntos["test"]])
    referencia = recortar(historia, *fechas["referencia"])
    if len(referencia) != len(conjuntos["train"]):
        raise ValueError("La referencia no reproduce el train del champion")
    return referencia, recortar(historia, *fechas["reciente"])


def construir_actual(
    archivos_historial: list[Path],
    archivos_actual: list[Path],
    estaciones: pd.DataFrame,
    periodos: dict | None = None,
) -> pd.DataFrame:
    """Ventana actual con el mismo pipeline de P1 y el catálogo vigente en ese periodo.

    Las variables se calculan sobre todo el historial más el mes nuevo, igual que en producción,
    para que los rezagos del primer día del mes usen los días previos; después se recorta.
    """
    inicio, fin = validar_periodos(periodos)["actual"]
    datos = preparar_datos_p1(
        [*archivos_historial, *archivos_actual], estaciones, FECHA_INICIO, str(fin.date())
    )
    return recortar(p1.unir_estaciones(datos, estaciones), inicio, fin)


def huella(datos: pd.DataFrame) -> str:
    """SHA-256 (12 caracteres) del contenido, para comprobar que un conjunto se reconstruye igual."""
    valores = pd.util.hash_pandas_object(datos[COLUMNAS], index=False).to_numpy()
    return hashlib.sha256(valores.tobytes()).hexdigest()[:12]


def resumen(datos: pd.DataFrame, inicio, fin, catalogo: str) -> dict:
    """Tamaño, cobertura y nulos de un conjunto, para el manifiesto y la metodología del reporte."""
    fechas = pd.to_datetime(datos["fecha"])
    return {
        "inicio": str(pd.Timestamp(inicio).date()),
        "fin": str(pd.Timestamp(fin).date()),
        "dias": int(fechas.dt.normalize().nunique()),
        "filas": len(datos),
        "estaciones": int(datos["estacion_id"].nunique()),
        "demanda_media": round(float(datos[OBJETIVO].mean()), 4),
        "nulos": {c: int(n) for c, n in datos[VARIABLES].isna().sum().items() if n},
        "catalogo": catalogo,
        "huella": huella(datos),
    }


def cambios_catalogo(anterior: pd.DataFrame, vigente: pd.DataFrame) -> dict:
    """Estaciones dadas de alta o de baja y cambios de capacidad entre dos catálogos."""
    unidos = anterior.merge(vigente, on="estacion_id", how="outer", suffixes=("_ant", "_vig"))
    ambos = unidos.dropna(subset=["capacidad_ant", "capacidad_vig"])
    cambio = ambos.loc[ambos["capacidad_ant"] != ambos["capacidad_vig"]]
    return {
        "estaciones_anterior": len(anterior),
        "estaciones_vigente": len(vigente),
        "altas": sorted(int(i) for i in unidos.loc[unidos["capacidad_ant"].isna(), "estacion_id"]),
        "bajas": sorted(int(i) for i in unidos.loc[unidos["capacidad_vig"].isna(), "estacion_id"]),
        "cambios_capacidad": [
            {
                "estacion_id": int(f.estacion_id),
                "antes": int(f.capacidad_ant),
                "ahora": int(f.capacidad_vig),
            }
            for f in cambio.sort_values("estacion_id").itertuples()
        ],
    }


def calidad_viajes(archivos: list[Path], estaciones: pd.DataFrame) -> dict:
    """Viajes crudos del periodo actual y cuántos descarta la limpieza de P1 (sin datos personales)."""
    viajes = pd.concat([pd.read_parquet(a) for a in archivos], ignore_index=True)
    sin_duplicados = viajes.drop_duplicates()
    ids = set(estaciones["estacion_id"])
    en_catalogo = sin_duplicados["origen_id"].isin(ids) & sin_duplicados["destino_id"].isin(ids)
    limpios = limpiar_viajes(viajes, estaciones)
    return {
        "viajes": len(viajes),
        "filas_duplicadas": len(viajes) - len(sin_duplicados),
        "fuera_de_catalogo": int((~en_catalogo).sum()),
        "duracion_invalida": int(en_catalogo.sum()) - len(limpios),
        "viajes_limpios": len(limpios),
    }


def ingerir_actual() -> dict:
    """Descarga y normaliza el mes actual y su catálogo, con el mismo código de la ingesta."""
    archivos = {}
    for mes, url in URLS_VIAJES_MONITOREO.items():
        crudo = RAW_DATA_DIR / f"datos_abiertos_{mes}.csv"
        procesado = VIAJES_MONITOREO_DIR / f"viajes_{mes}.parquet"
        descargar_archivo(url, crudo)
        if not procesado.exists():
            procesar_viajes(crudo, procesado)
        archivos[crudo.name] = {"url": url, "sha256": calcular_sha256(crudo), "parquet": procesado}
    catalogo = RAW_DATA_DIR / URL_NOMENCLATURA_MONITOREO.split("/")[-1]
    descargar_archivo(URL_NOMENCLATURA_MONITOREO, catalogo)
    if not ESTACIONES_MONITOREO_PARQUET.exists():
        procesar_estaciones(catalogo, ESTACIONES_MONITOREO_PARQUET)
    archivos[catalogo.name] = {
        "url": URL_NOMENCLATURA_MONITOREO,
        "sha256": calcular_sha256(catalogo),
    }
    return archivos


def champion_registrado() -> dict:
    """Nombre, versión numérica y corrida del champion de P1 en el registry (servidor de MLflow)."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    nombre = MODELOS_REGISTRADOS["p1"]
    version = MlflowClient().get_model_version_by_alias(nombre, ALIAS_PRODUCCION)
    return {
        "nombre": nombre,
        "alias": ALIAS_PRODUCCION,
        "version": str(version.version),
        "uri_version": f"models:/{nombre}/{version.version}",
        "run_id": version.run_id,
        "version_datos": version.tags.get("version_datos"),
    }


def generar(registry: bool = True) -> dict:
    """Construye y guarda los tres conjuntos y devuelve el manifiesto (determinista, sin fechas)."""
    fechas = validar_periodos()
    archivos = ingerir_actual()
    referencia, reciente = construir_referencia_y_reciente()
    vigente = pd.read_parquet(ESTACIONES_MONITOREO_PARQUET)
    archivos_actual = [a["parquet"] for a in archivos.values() if "parquet" in a]
    actual = construir_actual(
        sorted(VIAJES_DIR.glob("viajes_*.parquet")), archivos_actual, vigente
    )
    validar_sin_fuga(pd.concat([referencia, reciente]), actual)

    MONITOREO_DIR.mkdir(parents=True, exist_ok=True)
    conjuntos = {"referencia": referencia, "reciente": reciente, "actual": actual}
    for ventana, datos in conjuntos.items():
        datos.to_parquet(MONITOREO_DIR / f"{ventana}.parquet", index=False)

    catalogo_entrenamiento = URL_NOMENCLATURA.split("/")[-1]
    catalogo_actual = URL_NOMENCLATURA_MONITOREO.split("/")[-1]
    manifiesto = {
        "fuente": FUENTE_DATOS,
        "pregunta": "p1",
        "granularidad": "estación × día",
        "modelo": champion_registrado() if registry else None,
        "version_datos_entrenamiento": version_datos(MANIFIESTO_DATOS),
        "semilla": SEMILLA,
        "periodos": {v: [str(i.date()), str(f.date())] for v, (i, f) in fechas.items()},
        "reglas_sin_fuga": REGLAS_SIN_FUGA,
        "columnas": {
            "id": COLUMNAS_ID,
            "variables": VARIABLES,
            "categoricas": CATEGORICAS,
            "numericas": NUMERICAS,
            "objetivo": OBJETIVO,
        },
        "conjuntos": {
            v: resumen(
                datos,
                *fechas[v],
                catalogo_actual if v == "actual" else catalogo_entrenamiento,
            )
            for v, datos in conjuntos.items()
        },
        "archivos_actual": {
            nombre: {k: v for k, v in datos.items() if k != "parquet"}
            for nombre, datos in archivos.items()
        },
        "calidad_actual": calidad_viajes(archivos_actual, vigente),
        "cambios_catalogo": cambios_catalogo(cargar_estaciones(), vigente),
    }
    MANIFIESTO_MONITOREO.write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifiesto


def cargar_conjuntos() -> dict[str, pd.DataFrame]:
    """Los tres conjuntos guardados por `generar`; es la entrada de Evidently, challenger y A/B."""
    rutas = {v: MONITOREO_DIR / f"{v}.parquet" for v in VENTANAS}
    faltantes = [str(r) for r in rutas.values() if not r.exists()]
    if faltantes:
        raise FileNotFoundError(f"Faltan {faltantes}; ejecutar make monitoring-data")
    return {v: pd.read_parquet(r) for v, r in rutas.items()}


def datos_entrenamiento_challenger(conjuntos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Referencia + reciente: todo lo que el challenger puede usar, verificado contra la evaluación."""
    entrenamiento = pd.concat([conjuntos["referencia"], conjuntos["reciente"]], ignore_index=True)
    validar_sin_fuga(entrenamiento, conjuntos["actual"])
    return entrenamiento


@app.command()
def main(
    registry: bool = typer.Option(
        True, "--registry/--sin-registry", help="Anotar la versión del champion desde MLflow"
    ),
):
    """Prepara los conjuntos de referencia, reciente y actual del monitoreo de P1."""
    manifiesto = generar(registry)
    for ventana, datos in manifiesto["conjuntos"].items():
        logger.info(
            f"{ventana}: {datos['inicio']} a {datos['fin']}, {datos['filas']:,} filas, "
            f"{datos['estaciones']} estaciones, huella {datos['huella']}"
        )
    logger.success(f"Manifiesto escrito en {MANIFIESTO_MONITOREO}")


if __name__ == "__main__":
    app()
