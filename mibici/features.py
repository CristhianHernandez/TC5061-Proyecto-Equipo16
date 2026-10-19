from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from mibici.config import PROCESSED_DATA_DIR

app = typer.Typer()


def limpiar_viajes(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
) -> pd.DataFrame:
    """
    Limpia los viajes para construir las variables de P1, P2 y P3.

    Elimina duplicados exactos, estaciones fuera del catálogo
    y viajes con duración <= 0 o superior a 24 horas.

    No modifica los DataFrames originales.
    """
    columnas_viajes = {"viaje_id", "inicio", "fin", "origen_id", "destino_id"}

    if not columnas_viajes.issubset(viajes.columns):
        raise ValueError("Faltan columnas requeridas en viajes")

    if "estacion_id" not in estaciones.columns:
        raise ValueError("Falta estacion_id en estaciones")

    df = viajes.drop_duplicates().copy()

    ids_validos = set(estaciones["estacion_id"].dropna())

    df = df.loc[df["origen_id"].isin(ids_validos) & df["destino_id"].isin(ids_validos)].copy()

    duracion_min = (df["fin"] - df["inicio"]).dt.total_seconds() / 60

    df = df.loc[duracion_min.gt(0) & duracion_min.le(1440)].copy()

    return df.reset_index(drop=True)


def particion_temporal(
    datos: pd.DataFrame,
    columna_fecha: str = "fecha",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide los datos cronológicamente en entrenamiento,
    validación y prueba para P1 y P2.

    Utiliza los cortes temporales definidos en config.py.
    No modifica el DataFrame original.
    """
    from mibici.config import (
        FECHA_FIN,
        FECHA_INICIO,
        INICIO_PRUEBA,
        INICIO_VALIDACION,
    )

    if columna_fecha not in datos.columns:
        raise ValueError(f"Falta la columna {columna_fecha}")

    fechas = pd.to_datetime(datos[columna_fecha])

    inicio = pd.Timestamp(FECHA_INICIO)
    validacion = pd.Timestamp(INICIO_VALIDACION)
    prueba = pd.Timestamp(INICIO_PRUEBA)
    fin_exclusivo = pd.Timestamp(FECHA_FIN) + pd.Timedelta(days=1)

    entrenamiento = datos.loc[(fechas >= inicio) & (fechas < validacion)].copy()

    df_validacion = datos.loc[(fechas >= validacion) & (fechas < prueba)].copy()

    df_prueba = datos.loc[(fechas >= prueba) & (fechas < fin_exclusivo)].copy()

    return entrenamiento, df_validacion, df_prueba


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
        raise ValueError("fecha_inicio no puede ser posterior a fecha_fin")

    # Contar salidas por estación y fecha
    viajes_periodo = viajes.loc[
        (viajes["inicio"] >= inicio) & (viajes["inicio"] < fin + pd.Timedelta(days=1)),
        ["origen_id", "inicio"],
    ].copy()

    viajes_periodo["fecha"] = viajes_periodo["inicio"].dt.normalize()

    conteos = viajes_periodo.groupby(["origen_id", "fecha"]).size().rename("demanda")

    # Construir calendario completo
    ids_estaciones = estaciones["estacion_id"].drop_duplicates().sort_values()

    fechas = pd.date_range(inicio, fin, freq="D")

    calendario = pd.MultiIndex.from_product(
        [ids_estaciones, fechas],
        names=["estacion_id", "fecha"],
    )

    demanda = (
        conteos.rename_axis(index=["estacion_id", "fecha"])
        .reindex(calendario, fill_value=0)
        .reset_index()
    )

    return demanda


def consolidar_demanda_diaria(
    directorio_viajes: Path,
    estaciones: pd.DataFrame,
    fecha_inicio: str,
    fecha_fin: str,
) -> pd.DataFrame:
    """
    Consolida la demanda diaria de archivos Parquet mensuales.

    Limpia los viajes por mes y construye un calendario
    continuo por estación, incluyendo días con demanda cero.
    """

    resumenes = []

    for archivo in sorted(directorio_viajes.glob("viajes_*.parquet")):
        viajes = pd.read_parquet(archivo)
        limpios = limpiar_viajes(viajes, estaciones)

        conteos = (
            limpios.assign(fecha=limpios["inicio"].dt.normalize())
            .groupby(["origen_id", "fecha"])
            .size()
            .rename("demanda")
            .reset_index()
            .rename(columns={"origen_id": "estacion_id"})
        )

        resumenes.append(conteos)

    if not resumenes:
        raise ValueError("No se encontraron archivos de viajes")

    conteos_totales = pd.concat(resumenes, ignore_index=True)

    inicio = pd.Timestamp(fecha_inicio).normalize()
    fin = pd.Timestamp(fecha_fin).normalize()

    if inicio > fin:
        raise ValueError("fecha_inicio no puede ser posterior a fecha_fin")

    conteos_totales = conteos_totales.loc[conteos_totales["fecha"].between(inicio, fin)]

    conteos_totales = conteos_totales.groupby(["estacion_id", "fecha"])["demanda"].sum()

    calendario = pd.MultiIndex.from_product(
        [
            estaciones["estacion_id"].drop_duplicates().sort_values(),
            pd.date_range(inicio, fin, freq="D"),
        ],
        names=["estacion_id", "fecha"],
    )

    return conteos_totales.reindex(calendario, fill_value=0).reset_index()


def calcular_primera_actividad(
    demanda: pd.DataFrame,
) -> pd.DataFrame:
    """
    Obtiene la primera fecha con demanda positiva por estación.

    Representa la primera salida observada, no necesariamente
    la fecha oficial de entrada en operación.
    """
    actividad = (
        demanda.loc[demanda["demanda"] > 0]
        .groupby("estacion_id", as_index=False)["fecha"]
        .min()
        .rename(columns={"fecha": "primera_actividad"})
    )

    return actividad


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
        actividad_estaciones[["estacion_id", "primera_actividad"]],
        on="estacion_id",
        how="left",
        validate="many_to_one",
    )

    calendario = calendario.loc[
        calendario["primera_actividad"].notna()
        & (calendario["fecha"] >= calendario["primera_actividad"])
    ]

    return calendario.drop(columns="primera_actividad").reset_index(drop=True)


def agregar_features_calendario(
    demanda: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrega variables temporales para la predicción
    de demanda diaria por estación.
    """
    import holidays

    df = demanda.copy()

    df["fecha"] = pd.to_datetime(df["fecha"])
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["mes"] = df["fecha"].dt.month
    df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["dia_anio"] = df["fecha"].dt.dayofyear

    festivos_mexico = holidays.country_holidays(
        "MX",
        years=df["fecha"].dt.year.unique().tolist(),
    )

    df["es_festivo"] = (df["fecha"].dt.date.isin(festivos_mexico)).astype(int)

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

    df = df.sort_values(["estacion_id", "fecha"]).reset_index(drop=True)

    # Demanda de la fila anterior
    df["demanda_lag_1"] = df.groupby("estacion_id")["demanda"].shift(1)

    # Fecha de la fila anterior
    fecha_anterior = df.groupby("estacion_id")["fecha"].shift(1)

    # Comprobar que las fechas sean consecutivas
    diferencia = df["fecha"] - fecha_anterior

    df.loc[diferencia != pd.Timedelta(days=1), "demanda_lag_1"] = float("nan")

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
        raise ValueError("Existen combinaciones estación-fecha duplicadas")

    historial = df[["estacion_id", "fecha", "demanda"]].copy()

    # Desplazar la fecha del historial siete días hacia adelante
    historial["fecha"] += pd.Timedelta(days=7)

    historial = historial.rename(columns={"demanda": "demanda_lag_7"})

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
        raise ValueError("Existen combinaciones estación-fecha duplicadas")

    df = df.sort_values(["estacion_id", "fecha"]).reset_index(drop=True)

    historial = df.set_index("fecha")

    df["demanda_media_7"] = (
        historial.groupby("estacion_id")["demanda"]
        .transform(
            lambda serie: (
                serie.shift(1, freq="D").rolling("7D", min_periods=7).mean().reindex(serie.index)
            )
        )
        .to_numpy()
    )

    return df


def preparar_datos_p1(
    directorio_viajes: Path,
    estaciones: pd.DataFrame,
    fecha_inicio: str,
    fecha_fin: str,
) -> pd.DataFrame:
    """
    Prepara la demanda diaria y las variables predictoras de P1.

    Conserva el calendario completo para calcular rezagos
    y después excluye fechas anteriores a la primera
    salida observada de cada estación.
    """

    demanda = consolidar_demanda_diaria(
        directorio_viajes,
        estaciones,
        fecha_inicio,
        fecha_fin,
    )

    actividad = calcular_primera_actividad(demanda)

    datos = agregar_features_calendario(demanda)
    datos = agregar_lag_demanda(datos)
    datos = agregar_lag_7(datos)
    datos = agregar_media_movil_7(datos)

    datos = filtrar_por_primera_actividad(
        datos,
        actividad,
    )

    return datos


def flujo_por_franja(viajes: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula salidas, llegadas y flujo neto por estación
    y franja fija de 3 horas.

    Recibe viajes previamente limpiados.
    """
    columnas = {"inicio", "fin", "origen_id", "destino_id"}

    if not columnas.issubset(viajes.columns):
        raise ValueError("Faltan columnas requeridas en viajes")

    salidas = (
        viajes.assign(fecha_franja=viajes["inicio"].dt.floor("3h"))
        .groupby(["origen_id", "fecha_franja"])
        .size()
        .rename("salidas")
    )

    llegadas = (
        viajes.assign(fecha_franja=viajes["fin"].dt.floor("3h"))
        .groupby(["destino_id", "fecha_franja"])
        .size()
        .rename("llegadas")
    )

    salidas.index = salidas.index.set_names(["estacion_id", "fecha_franja"])

    llegadas.index = llegadas.index.set_names(["estacion_id", "fecha_franja"])

    resultado = pd.concat([salidas, llegadas], axis=1).fillna(0).astype(int).reset_index()

    resultado["flujo_neto"] = resultado["salidas"] - resultado["llegadas"]

    # Completar franjas de 3 horas sin movimientos
    if resultado.empty:
        return resultado

    franjas = pd.date_range(
        start=resultado["fecha_franja"].min(),
        end=resultado["fecha_franja"].max(),
        freq="3h",
    )

    estaciones = resultado["estacion_id"].unique()

    indice_completo = pd.MultiIndex.from_product(
        [estaciones, franjas],
        names=["estacion_id", "fecha_franja"],
    )

    resultado = (
        resultado.set_index(["estacion_id", "fecha_franja"])
        .reindex(indice_completo, fill_value=0)
        .reset_index()
    )

    resultado["flujo_neto"] = resultado["salidas"] - resultado["llegadas"]

    return resultado.sort_values(["estacion_id", "fecha_franja"]).reset_index(drop=True)


def agregar_rezagos_flujo(
    flujos: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrega rezagos del flujo neto por estación:
    - 1 franja anterior (3 horas).
    - 56 franjas anteriores (7 días).

    No utiliza flujos futuros.
    """
    columnas = {"estacion_id", "fecha_franja", "flujo_neto"}

    if not columnas.issubset(flujos.columns):
        raise ValueError("Faltan columnas requeridas en flujos")

    df = flujos.copy()
    df["fecha_franja"] = pd.to_datetime(df["fecha_franja"])

    df = df.sort_values(["estacion_id", "fecha_franja"]).reset_index(drop=True)

    if df.duplicated(["estacion_id", "fecha_franja"]).any():
        raise ValueError("Existen franjas duplicadas por estación")

    historial = df.set_index(["estacion_id", "fecha_franja"])["flujo_neto"]

    for nombre, horas in [
        ("flujo_lag_1", 3),
        ("flujo_lag_56", 168),
    ]:
        fechas_previas = pd.MultiIndex.from_arrays(
            [
                df["estacion_id"],
                df["fecha_franja"] - pd.Timedelta(hours=horas),
            ],
            names=["estacion_id", "fecha_franja"],
        )

        df[nombre] = historial.reindex(fechas_previas).to_numpy()

    return df


def etiquetar_riesgo(
    flujos: pd.DataFrame,
    estaciones: pd.DataFrame,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """
    Clasifica el riesgo de cada estación y franja a partir
    del flujo neto observado y su capacidad.

    No genera variables predictoras ni modifica los datos originales.
    """
    columnas_flujos = {"estacion_id", "flujo_neto"}
    columnas_estaciones = {"estacion_id", "capacidad"}

    if not columnas_flujos.issubset(flujos.columns):
        raise ValueError("Faltan columnas requeridas en flujos")

    if not columnas_estaciones.issubset(estaciones.columns):
        raise ValueError("Faltan columnas requeridas en estaciones")

    if not 0 < alpha <= 1:
        raise ValueError("alpha debe estar entre 0 y 1")

    capacidades = estaciones[["estacion_id", "capacidad"]].drop_duplicates(subset=["estacion_id"])

    df = flujos.merge(
        capacidades,
        on="estacion_id",
        how="left",
        validate="many_to_one",
    )

    if df["capacidad"].isna().any() or df["capacidad"].le(0).any():
        raise ValueError("Existen estaciones sin capacidad válida")

    umbral = alpha * df["capacidad"]

    import numpy as np

    df["riesgo"] = np.select(
        [
            df["flujo_neto"] >= umbral,
            df["flujo_neto"] <= -umbral,
        ],
        [
            "sin_bicicletas",
            "sin_anclajes",
        ],
        default="normal",
    )

    return df


def preparar_datos_p2(
    flujos: pd.DataFrame,
    estaciones: pd.DataFrame,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """
    Prepara la tabla de P2 para predecir el riesgo
    durante una franja de 3 horas.

    Los predictores de flujo provienen exclusivamente
    de franjas anteriores.
    """
    df = etiquetar_riesgo(
        flujos,
        estaciones,
        alpha=alpha,
    )

    df = agregar_rezagos_flujo(df)

    df["hora"] = df["fecha_franja"].dt.hour
    df["dia_semana"] = df["fecha_franja"].dt.dayofweek

    columnas = [
        "estacion_id",
        "fecha_franja",
        "hora",
        "dia_semana",
        "capacidad",
        "flujo_lag_1",
        "flujo_lag_56",
        "riesgo",
    ]

    return df[columnas].copy()


def perfiles_estacion(
    viajes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye perfiles horarios normalizados por estación
    utilizando exclusivamente viajes del entrenamiento.

    Devuelve una fila por estación y 96 variables horarias.
    """
    from mibici.config import FECHA_INICIO, INICIO_VALIDACION

    columnas = {"inicio", "fin", "origen_id", "destino_id"}

    if not columnas.issubset(viajes.columns):
        raise ValueError("Faltan columnas requeridas en viajes")

    inicio = pd.Timestamp(FECHA_INICIO)
    corte = pd.Timestamp(INICIO_VALIDACION)

    # Filtrar por inicio del viaje para evitar fuga temporal
    df = viajes.loc[
        (viajes["inicio"] >= inicio) & (viajes["inicio"] < corte) & (viajes["fin"] < corte)
    ].copy()

    perfiles = []

    for tipo, columna_fecha, columna_estacion in [
        ("salidas", "inicio", "origen_id"),
        ("llegadas", "fin", "destino_id"),
    ]:
        movimientos = df[[columna_fecha, columna_estacion]].copy()

        movimientos.columns = ["fecha", "estacion_id"]

        movimientos["hora"] = movimientos["fecha"].dt.hour

        movimientos["tipo_dia"] = (movimientos["fecha"].dt.dayofweek >= 5).map(
            {False: "laboral", True: "fin_semana"}
        )

        conteos = (
            movimientos.groupby(["estacion_id", "tipo_dia", "hora"])
            .size()
            .unstack(["tipo_dia", "hora"], fill_value=0)
        )

        # Mantener siempre las 48 combinaciones horarias
        columnas_horarias = pd.MultiIndex.from_product(
            [["laboral", "fin_semana"], range(24)],
            names=["tipo_dia", "hora"],
        )

        conteos = conteos.reindex(
            columns=columnas_horarias,
            fill_value=0,
        )

        # Normalizar por estación y tipo de día
        totales = conteos.T.groupby(level=0).transform("sum").T

        normalizado = conteos.div(totales.where(totales.ne(0), 1))

        normalizado.columns = [f"{tipo}_{dia}_{hora:02d}" for dia, hora in normalizado.columns]

        perfiles.append(normalizado)

        resultado = pd.concat(perfiles, axis=1).fillna(0).copy()

    # Volumen de movimientos por estación
    total_salidas = df.groupby("origen_id").size()
    total_llegadas = df.groupby("destino_id").size()

    resultado["total_salidas"] = total_salidas.reindex(resultado.index, fill_value=0)

    resultado["total_llegadas"] = total_llegadas.reindex(resultado.index, fill_value=0)

    resultado["volumen_total"] = resultado["total_salidas"] + resultado["total_llegadas"]

    resultado["balance_neto"] = resultado["total_salidas"] - resultado["total_llegadas"]

    # Duración media de viajes iniciados en cada estación
    duracion_min = (df["fin"] - df["inicio"]).dt.total_seconds() / 60

    duracion_por_estacion = (
        df.assign(duracion_min=duracion_min).groupby("origen_id")["duracion_min"].mean()
    )

    resultado["duracion_media_min"] = duracion_por_estacion.reindex(resultado.index)

    return resultado.reset_index()


@app.command()
def main(
    fecha_inicio: str = "2024-10-01",
    fecha_fin: str = "2026-09-30",
):
    """Genera las características de demanda diaria para P1."""
    from mibici.config import ESTACIONES_PARQUET, VIAJES_DIR

    logger.info("Iniciando generación de características P1...")

    estaciones = pd.read_parquet(ESTACIONES_PARQUET)

    datos = preparar_datos_p1(
        VIAJES_DIR,
        estaciones,
        fecha_inicio,
        fecha_fin,
    )

    salida = PROCESSED_DATA_DIR / "features_p1.parquet"
    salida.parent.mkdir(parents=True, exist_ok=True)

    datos.to_parquet(salida, index=False)

    logger.success(f"Características P1 generadas: {len(datos):,} registros")
    logger.info(f"Archivo guardado en: {salida}")


if __name__ == "__main__":
    app()
