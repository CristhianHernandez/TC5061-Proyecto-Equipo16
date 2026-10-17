from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd

from mibici.config import PROCESSED_DATA_DIR

app = typer.Typer()

def demanda_diaria(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
    fecha_inicio: str,
    fecha_fin: str,
) -> pd.DataFrame:
    """
    Calcula el número diario de viajes iniciados por estación.

    Genera una fila por estación y día, incluyendo demanda cero.
    No determina si la estación estaba operativa.
    """

    inicio = pd.Timestamp(fecha_inicio).normalize()
    fin = pd.Timestamp(fecha_fin).normalize()

    if inicio > fin:
        raise ValueError(
            "fecha_inicio no puede ser posterior a fecha_fin"
        )

    # Contar salidas por estación y fecha
    viajes_periodo = viajes.loc[
        (viajes["inicio"] >= inicio)
        & (viajes["inicio"] < fin + pd.Timedelta(days=1)),
        ["origen_id", "inicio"],
    ].copy()

    viajes_periodo["fecha"] = (
        viajes_periodo["inicio"].dt.normalize()
    )

    conteos = (
        viajes_periodo
        .groupby(["origen_id", "fecha"])
        .size()
        .rename("demanda")
    )

    # Construir calendario completo
    ids_estaciones = (
        estaciones["estacion_id"]
        .drop_duplicates()
        .sort_values()
    )

    fechas = pd.date_range(inicio, fin, freq="D")

    calendario = pd.MultiIndex.from_product(
        [ids_estaciones, fechas],
        names=["estacion_id", "fecha"],
    )

    demanda = (
        conteos
        .rename_axis(index=["estacion_id", "fecha"])
        .reindex(calendario, fill_value=0)
        .reset_index()
    )

    return demanda

def filtrar_por_primera_actividad(
    demanda: pd.DataFrame,
    actividad_estaciones: pd.DataFrame,
) -> pd.DataFrame:
    """
    Excluye fechas anteriores a la primera actividad observada
    de cada estación.

    Es un filtro exploratorio, no una validación oficial
    de las fechas de operación.
    """

    calendario = demanda.merge(
        actividad_estaciones[
            ["estacion_id", "primera_actividad"]
        ],
        on="estacion_id",
        how="left",
        validate="many_to_one",
    )

    calendario = calendario.loc[
        calendario["primera_actividad"].notna()
        & (
            calendario["fecha"]
            >= calendario["primera_actividad"]
        )
    ]

    return (
        calendario
        .drop(columns="primera_actividad")
        .reset_index(drop=True)
    )

def agregar_features_calendario(
    demanda: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrega variables temporales para la predicción
    de demanda diaria por estación.
    """

    df = demanda.copy()

    df["fecha"] = pd.to_datetime(df["fecha"])

    df["dia_semana"] = df["fecha"].dt.dayofweek

    df["mes"] = df["fecha"].dt.month

    df["es_fin_semana"] = (
        df["dia_semana"] >= 5
    ).astype(int)

    df["dia_anio"] = df["fecha"].dt.dayofyear

    return df

def agregar_lag_demanda(
    demanda: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula la demanda del día calendario anterior por estación.
    Si falta el día anterior, devuelve NaN.
    """

    df = demanda.copy()

    df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()

    df = df.sort_values(
        ["estacion_id", "fecha"]
    ).reset_index(drop=True)

    # Demanda de la fila anterior
    df["demanda_lag_1"] = (
        df.groupby("estacion_id")["demanda"]
        .shift(1)
    )

    # Fecha de la fila anterior
    fecha_anterior = (
        df.groupby("estacion_id")["fecha"]
        .shift(1)
    )

    # Comprobar que las fechas sean consecutivas
    diferencia = df["fecha"] - fecha_anterior

    df.loc[
        diferencia != pd.Timedelta(days=1),
        "demanda_lag_1"
    ] = float("nan")

    return df

def agregar_lag_7(
    demanda: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrega la demanda de siete días calendario antes
    para cada estación.
    """

    df = demanda.copy()
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()

    if df.duplicated(["estacion_id", "fecha"]).any():
        raise ValueError(
            "Existen combinaciones estación-fecha duplicadas"
        )

    historial = df[
        ["estacion_id", "fecha", "demanda"]
    ].copy()

    # Desplazar la fecha del historial siete días hacia adelante
    historial["fecha"] += pd.Timedelta(days=7)

    historial = historial.rename(
        columns={"demanda": "demanda_lag_7"}
    )

    return df.merge(
        historial,
        on=["estacion_id", "fecha"],
        how="left",
        validate="one_to_one",
    )

def agregar_media_movil_7(
    demanda: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula la demanda promedio de los 7 días calendario
    anteriores, sin incluir el día actual.

    Requiere los 7 días completos; de lo contrario,
    devuelve NaN.
    """

    df = demanda.copy()
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()

    if df.duplicated(["estacion_id", "fecha"]).any():
        raise ValueError(
            "Existen combinaciones estación-fecha duplicadas"
        )

    df = df.sort_values(
        ["estacion_id", "fecha"]
    ).reset_index(drop=True)

    historial = df.set_index("fecha")

    df["demanda_media_7"] = (
        historial
        .groupby("estacion_id")["demanda"]
        .transform(
            lambda serie: serie
            .shift(1, freq="D")
            .rolling("7D", min_periods=7)
            .mean()
            .reindex(serie.index)
        )
        .to_numpy()
    )

    return df

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
