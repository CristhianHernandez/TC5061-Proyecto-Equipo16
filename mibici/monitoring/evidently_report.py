"""Reporte de drift y calidad de datos con Evidently para el champion de P1 (Entregable 8).

Consume los conjuntos de `mibici.monitoring.datasets` (PR de CH1) y compara la referencia (train
del champion) contra la ventana actual (octubre de 2026). Para interpretar el resultado agrega dos
comparaciones de contexto: referencia contra la ventana reciente (¿el cambio ya existía antes de
octubre?) y referencia contra octubre solo en las estaciones que el champion vio (¿cuánto del
drift se debe a las estaciones nuevas?).

Monitorea solo las 11 variables de entrada del modelo; los identificadores (`estacion_id`, `fecha`)
no entran y la demanda observada se reporta aparte como drift del objetivo, sin contar en la
proporción de variables con drift.

Desde la raíz, con los conjuntos ya generados (`make monitoring-data`):

    python -m mibici.monitoring.evidently_report       # o: make monitoring-drift
    python -m mibici.monitoring.evidently_report --sin-html

Escribe en docs/evidencias/drift/: los reportes HTML de Evidently, `drift_por_variable.csv`,
`resumen_drift.json` y tres gráficas PNG. Las tablas no llevan fechas de ejecución, así que una
nueva corrida con los mismos datos produce los mismos archivos.
"""

import json
from pathlib import Path
from typing import Annotated

from loguru import logger
import pandas as pd
import typer

from mibici.config import (
    DRIFT_DIR,
    METODO_DRIFT_CATEGORICO,
    METODO_DRIFT_NUMERICO,
    PERIODOS_MONITOREO,
    PROPORCION_DRIFT_DATASET,
    UMBRAL_DRIFT,
)
from mibici.monitoring.datasets import (
    CATEGORICAS,
    COLUMNAS,
    COLUMNAS_ID,
    NUMERICAS,
    OBJETIVO,
    VARIABLES,
    cargar_conjuntos,
)

app = typer.Typer()

# Grupo de cada columna, para leer el drift por su causa probable y no variable por variable
GRUPOS = {
    "dia_semana": "calendario",
    "mes": "calendario",
    "es_fin_semana": "calendario",
    "dia_anio": "calendario",
    "es_festivo": "calendario",
    "demanda_lag_1": "historial de demanda",
    "demanda_lag_7": "historial de demanda",
    "demanda_media_7": "historial de demanda",
    "capacidad": "estación",
    "latitud": "estación",
    "longitud": "estación",
    OBJETIVO: "objetivo",
}

# nombre -> (ventana de referencia, ventana comparada, ¿HTML?, ¿solo estaciones de la referencia?)
COMPARACIONES = {
    "referencia_vs_actual": ("referencia", "actual", True, False),
    "referencia_vs_reciente": ("referencia", "reciente", True, False),
    "referencia_vs_actual_estaciones_conocidas": ("referencia", "actual", False, True),
}
COMPARACION_PRINCIPAL = "referencia_vs_actual"

COLUMNAS_TABLA = [
    "comparacion",
    "columna",
    "grupo",
    "tipo",
    "metodo",
    "umbral",
    "puntaje",
    "drift",
    "media_referencia",
    "media_actual",
    "cambio_media_pct",
    "nulos_referencia_pct",
    "nulos_actual_pct",
    "fuera_de_referencia_pct",
]


# --- Evidently ---------------------------------------------------------------------------------


def definicion_datos():
    """Tipos del contrato de CH1: indicadores y calendario discretos como categóricas."""
    from evidently import DataDefinition

    return DataDefinition(
        numerical_columns=[*NUMERICAS, OBJETIVO], categorical_columns=list(CATEGORICAS)
    )


def construir_reporte():
    """Drift de las variables de entrada, drift del objetivo y resumen de calidad de datos."""
    from evidently import Report
    from evidently.metrics import ValueDrift
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    return Report(
        [
            DataDriftPreset(
                columns=VARIABLES,
                num_method=METODO_DRIFT_NUMERICO,
                cat_method=METODO_DRIFT_CATEGORICO,
                threshold=UMBRAL_DRIFT,
                drift_share=PROPORCION_DRIFT_DATASET,
            ),
            ValueDrift(column=OBJETIVO, method=METODO_DRIFT_NUMERICO, threshold=UMBRAL_DRIFT),
            DataSummaryPreset(columns=[*VARIABLES, OBJETIVO]),
        ],
        include_tests=False,
    )


def ejecutar_evidently(referencia: pd.DataFrame, actual: pd.DataFrame):
    """Corre el reporte con la referencia como base; devuelve el snapshot de Evidently."""
    from evidently import Dataset

    columnas = [*VARIABLES, OBJETIVO]
    validar_esquema(referencia)
    validar_esquema(actual)
    definicion = definicion_datos()
    base = Dataset.from_pandas(referencia[columnas], data_definition=definicion)
    nuevo = Dataset.from_pandas(actual[columnas], data_definition=definicion)
    return construir_reporte().run(nuevo, base)


def validar_esquema(datos: pd.DataFrame) -> None:
    """Falla si faltan columnas del contrato o si una variable no es numérica."""
    faltantes = [c for c in [*VARIABLES, OBJETIVO] if c not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas del contrato de monitoreo: {faltantes}")
    no_numericas = [
        c for c in [*VARIABLES, OBJETIVO] if not pd.api.types.is_numeric_dtype(datos[c])
    ]
    if no_numericas:
        raise TypeError(f"Columnas con tipo no numérico: {no_numericas}")
    if datos.empty:
        raise ValueError("El conjunto a monitorear está vacío")


def puntajes_drift(snapshot) -> dict[str, dict]:
    """Método, umbral y puntaje de cada `ValueDrift` del snapshot, por columna."""
    resultado = {}
    for metrica in snapshot.dict()["metrics"]:
        config = metrica["config"]
        if config["type"].endswith(":ValueDrift"):
            resultado[config["column"]] = {
                "metodo": config["method"],
                "umbral": float(config["threshold"]),
                "puntaje": float(metrica["value"]),
            }
    return resultado


def columnas_con_drift_evidently(snapshot) -> dict:
    """Conteo y proporción de variables con drift calculados por el propio Evidently."""
    for metrica in snapshot.dict()["metrics"]:
        if metrica["config"]["type"].endswith(":DriftedColumnsCount"):
            return {k: float(v) for k, v in metrica["value"].items()}
    raise ValueError("El snapshot no tiene DriftedColumnsCount")


# --- tabla por variable y calidad --------------------------------------------------------------


def fuera_de_referencia(referencia: pd.Series, actual: pd.Series, categorica: bool) -> float:
    """Proporción de valores actuales que la referencia nunca tuvo (categoría o rango)."""
    actual = actual.dropna()
    if actual.empty:
        return 0.0
    if categorica:
        fuera = ~actual.isin(set(referencia.dropna().unique()))
    else:
        fuera = (actual < referencia.min()) | (actual > referencia.max())
    return float(fuera.mean())


def tabla_drift(
    snapshot, referencia: pd.DataFrame, actual: pd.DataFrame, comparacion: str
) -> pd.DataFrame:
    """Una fila por columna monitoreada: puntaje de Evidently, decisión y magnitud del cambio."""
    puntajes = puntajes_drift(snapshot)
    filas = []
    for columna in [*VARIABLES, OBJETIVO]:
        datos = puntajes[columna]
        categorica = columna in CATEGORICAS
        media_ref = float(referencia[columna].mean())
        media_act = float(actual[columna].mean())
        filas.append(
            {
                "comparacion": comparacion,
                "columna": columna,
                "grupo": GRUPOS[columna],
                "tipo": "categorica" if categorica else "numerica",
                **datos,
                "drift": datos["puntaje"] >= datos["umbral"],
                "media_referencia": media_ref,
                "media_actual": media_act,
                "cambio_media_pct": 100 * (media_act - media_ref) / media_ref
                if media_ref
                else None,
                "nulos_referencia_pct": 100 * float(referencia[columna].isna().mean()),
                "nulos_actual_pct": 100 * float(actual[columna].isna().mean()),
                "fuera_de_referencia_pct": 100
                * fuera_de_referencia(referencia[columna], actual[columna], categorica),
            }
        )
    tabla = pd.DataFrame(filas, columns=COLUMNAS_TABLA)
    return tabla.round(4)


def resumen_comparacion(tabla: pd.DataFrame, snapshot, n_ref: int, n_act: int) -> dict:
    """Indicadores globales de una comparación; verifica que coincidan con los de Evidently."""
    entradas = tabla[tabla["columna"] != OBJETIVO]
    con_drift = entradas[entradas["drift"]].sort_values("puntaje", ascending=False)
    evidently = columnas_con_drift_evidently(snapshot)
    if int(evidently["count"]) != len(con_drift):
        raise RuntimeError(
            f"La tabla marca {len(con_drift)} variables con drift y Evidently {evidently['count']}"
        )
    proporcion = len(con_drift) / len(entradas)
    objetivo = tabla[tabla["columna"] == OBJETIVO].iloc[0]
    return {
        "filas_referencia": n_ref,
        "filas_actual": n_act,
        "variables_monitoreadas": len(entradas),
        "variables_con_drift": len(con_drift),
        "proporcion_con_drift": round(proporcion, 4),
        "drift_del_conjunto": proporcion >= PROPORCION_DRIFT_DATASET,
        "variables_con_drift_por_puntaje": con_drift["columna"].tolist(),
        "drift_por_grupo": {
            grupo: sorted(datos.loc[datos["drift"], "columna"].tolist())
            for grupo, datos in entradas.groupby("grupo")
        },
        "objetivo": {
            "columna": OBJETIVO,
            "puntaje": float(objetivo["puntaje"]),
            "drift": bool(objetivo["drift"]),
            "media_referencia": float(objetivo["media_referencia"]),
            "media_actual": float(objetivo["media_actual"]),
        },
    }


def calidad(datos: pd.DataFrame, referencia: pd.DataFrame | None = None) -> dict:
    """Calidad de un conjunto: esquema, duplicados, nulos, constantes y valores imposibles."""
    columnas = set(datos.columns)
    variables = [c for c in VARIABLES if c in columnas]
    resultado = {
        "filas": len(datos),
        "estaciones": int(datos["estacion_id"].nunique()) if "estacion_id" in columnas else None,
        "columnas_faltantes": [c for c in COLUMNAS if c not in columnas],
        "columnas_extra": sorted(columnas - set(COLUMNAS)),
        "filas_duplicadas": int(datos.duplicated().sum()),
        "claves_duplicadas": int(datos.duplicated(subset=COLUMNAS_ID).sum())
        if set(COLUMNAS_ID) <= columnas
        else None,
        "nulos": {c: int(n) for c, n in datos[variables].isna().sum().items() if n},
        "variables_constantes": [c for c in variables if datos[c].nunique(dropna=True) <= 1],
        "demanda_negativa": int((datos[OBJETIVO] < 0).sum()) if OBJETIVO in columnas else None,
    }
    if referencia is not None:
        resultado["tipos_distintos_a_referencia"] = {
            c: [str(referencia[c].dtype), str(datos[c].dtype)]
            for c in variables
            if c in referencia.columns and referencia[c].dtype != datos[c].dtype
        }
    return resultado


def estaciones_nuevas(referencia: pd.DataFrame, actual: pd.DataFrame) -> dict:
    """Estaciones del periodo comparado que no aparecen en la referencia y su peso en las filas."""
    conocidas = actual["estacion_id"].isin(set(referencia["estacion_id"]))
    return {
        "estaciones_referencia": int(referencia["estacion_id"].nunique()),
        "estaciones_conocidas": int(actual.loc[conocidas, "estacion_id"].nunique()),
        "estaciones_nuevas": int(actual.loc[~conocidas, "estacion_id"].nunique()),
        "filas_estaciones_nuevas_pct": round(100 * float((~conocidas).mean()), 2),
        "demanda_media_conocidas": round(float(actual.loc[conocidas, OBJETIVO].mean()), 4),
        "demanda_media_nuevas": round(float(actual.loc[~conocidas, OBJETIVO].mean()), 4)
        if (~conocidas).any()
        else None,
    }


def demanda_mismo_mes(conjuntos: dict[str, pd.DataFrame], ventana: str = "actual") -> dict:
    """Demanda media de las estaciones de la referencia en los mismos meses de años previos.

    Separa estacionalidad de un cambio real: si la ventana actual se parece a los mismos meses
    de años anteriores, el drift de la demanda es estacional; si no, el nivel cambió.
    """
    referencia = conjuntos["referencia"]
    conocidas = set(referencia["estacion_id"])
    historia = pd.concat(conjuntos.values(), ignore_index=True)
    historia = historia[historia["estacion_id"].isin(conocidas)]
    fechas = pd.to_datetime(historia["fecha"])
    meses = set(pd.to_datetime(conjuntos[ventana]["fecha"]).dt.month)
    inicio_ventana = pd.to_datetime(conjuntos[ventana]["fecha"]).min()
    filtro = fechas.dt.month.isin(meses)
    por_anio = historia[filtro].groupby(fechas[filtro].dt.year)[OBJETIVO].mean()
    anio_actual = int(inicio_ventana.year)
    previos = por_anio.drop(anio_actual, errors="ignore")
    resultado = {
        "meses": sorted(int(m) for m in meses),
        "demanda_media_por_anio": {int(a): round(float(v), 4) for a, v in por_anio.items()},
    }
    if not previos.empty and anio_actual in por_anio:
        base = float(previos.mean())
        resultado["cambio_vs_anios_previos_pct"] = round(
            100 * (float(por_anio[anio_actual]) - base) / base, 2
        )
    return resultado


# --- gráficas ----------------------------------------------------------------------------------

# Paleta categórica validada (azul, naranja, aqua) y tinta neutra para umbral y ejes
AZUL, NARANJA, AQUA, GRIS, TINTA = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e"


def _estilo(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRIS)
    ax.tick_params(colors=TINTA, labelsize=9)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)


def graficar_puntajes(tabla: pd.DataFrame, ruta: Path) -> None:
    """Puntaje de drift por variable en las dos comparaciones completas, con el umbral."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datos = tabla[tabla["comparacion"].isin(["referencia_vs_reciente", COMPARACION_PRINCIPAL])]
    datos = datos.pivot(index="columna", columns="comparacion", values="puntaje")
    datos = datos.sort_values(COMPARACION_PRINCIPAL)
    posiciones = range(len(datos))
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    alto = 0.38
    series = [
        ("referencia_vs_reciente", AQUA, "Referencia vs reciente (abr–sep 2026)", alto / 2),
        (COMPARACION_PRINCIPAL, AZUL, "Referencia vs actual (oct 2026)", -alto / 2),
    ]
    for comparacion, color, etiqueta, desplazamiento in series:
        ax.barh(
            [p + desplazamiento for p in posiciones],
            datos[comparacion],
            height=alto - 0.04,
            color=color,
            label=etiqueta,
        )
    ax.axvline(UMBRAL_DRIFT, color=TINTA, linestyle="--", linewidth=1.2)
    ax.text(UMBRAL_DRIFT, len(datos) - 0.4, f" umbral {UMBRAL_DRIFT}", color=TINTA, fontsize=8.5)
    ax.set_yticks(list(posiciones), [f"{c}  ({GRUPOS[c]})" for c in datos.index])
    ax.set_xlabel("Distancia (Wasserstein normalizada / Jensen-Shannon)", color=TINTA)
    ax.set_title("Drift por variable respecto a la referencia (train del champion)", fontsize=11)
    _estilo(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def variables_a_graficar(tabla: pd.DataFrame, por_grupo: int = 2) -> list[str]:
    """Variables numéricas de entrada con drift en la comparación principal, las de mayor puntaje
    de cada grupo. El calendario se omite porque su cambio es esperado con una ventana de un mes.
    """
    candidatas = tabla[
        (tabla["comparacion"] == COMPARACION_PRINCIPAL)
        & tabla["drift"]
        & (tabla["tipo"] == "numerica")
        & ~tabla["grupo"].isin(["calendario", "objetivo"])
    ].sort_values("puntaje", ascending=False)
    return candidatas.groupby("grupo", sort=False).head(por_grupo)["columna"].tolist()


def graficar_distribuciones(
    referencia: pd.DataFrame, actual: pd.DataFrame, columnas: list[str], ruta: Path
) -> None:
    """Densidad de la referencia contra la ventana actual para las variables indicadas."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    columnas_grafica = min(len(columnas), 2)
    filas_grafica = -(-len(columnas) // columnas_grafica)
    fig, ejes = plt.subplots(
        filas_grafica, columnas_grafica, figsize=(10, 3.2 * filas_grafica), dpi=150, squeeze=False
    )
    for ax in ejes.flat[len(columnas) :]:
        ax.set_visible(False)
    for ax, columna in zip(ejes.flat, columnas):
        ref, act = referencia[columna].dropna(), actual[columna].dropna()
        # Sin el 0.5 % de cada cola para que los valores extremos no aplasten la forma
        limites = np.quantile(pd.concat([ref, act]), [0.005, 0.995])
        bordes = np.linspace(*limites, 41) if limites[1] > limites[0] else 10
        for valores, color, etiqueta in [
            (ref, AZUL, "Referencia (oct 2024–mar 2026)"),
            (act, NARANJA, "Actual (oct 2026)"),
        ]:
            ax.hist(
                valores[valores.between(*limites)],
                bins=bordes,
                density=True,
                histtype="step",
                linewidth=2,
                color=color,
                label=etiqueta,
            )
        ax.set_title(f"{columna} ({GRUPOS[columna]})", fontsize=10)
        ax.set_yticks([])
        _estilo(ax)
        ax.grid(False)
    ejes.flat[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Variables de entrada con más drift: referencia vs octubre 2026", fontsize=11)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def graficar_demanda_diaria(conjuntos: dict[str, pd.DataFrame], ruta: Path) -> None:
    """Demanda media diaria (media móvil de 7 días) de estaciones conocidas y nuevas."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    historia = pd.concat(conjuntos.values(), ignore_index=True)
    historia["fecha"] = pd.to_datetime(historia["fecha"])
    conocidas = set(conjuntos["referencia"]["estacion_id"])
    historia["grupo"] = (
        historia["estacion_id"]
        .isin(conocidas)
        .map({True: "Estaciones de la referencia", False: "Estaciones nuevas (desde may 2026)"})
    )
    diaria = historia.groupby(["fecha", "grupo"])[OBJETIVO].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    for (ventana, (inicio, fin)), tono in zip(
        PERIODOS_MONITOREO.items(), ["#f3f2ef", "#e8f5ef", "#fdeee6"]
    ):
        ax.axvspan(pd.Timestamp(inicio), pd.Timestamp(fin), color=tono, linewidth=0)
        ax.text(
            pd.Timestamp(inicio),
            1,
            f" {ventana}",
            transform=ax.get_xaxis_transform(),
            va="top",
            fontsize=8.5,
            color=TINTA,
        )
    for grupo, color in zip(diaria.columns, [AZUL, NARANJA]):
        serie = diaria[grupo].dropna().rolling(7, min_periods=1).mean()
        ax.plot(serie.index, serie, color=color, linewidth=2, label=grupo)
    ax.set_ylabel("Viajes por estación y día", color=TINTA)
    ax.set_title(
        "Demanda media diaria por grupo de estaciones (media móvil de 7 días)", fontsize=11
    )
    _estilo(ax)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


# --- orquestación ------------------------------------------------------------------------------


def preparar_comparacion(conjuntos: dict[str, pd.DataFrame], nombre: str):
    """Referencia y conjunto comparado de una comparación de COMPARACIONES."""
    base, nuevo, _, solo_conocidas = COMPARACIONES[nombre]
    referencia, actual = conjuntos[base], conjuntos[nuevo]
    if solo_conocidas:
        actual = actual[actual["estacion_id"].isin(set(referencia["estacion_id"]))]
    return referencia, actual.reset_index(drop=True)


def generar(conjuntos: dict[str, pd.DataFrame], salida: Path = DRIFT_DIR, html: bool = True):
    """Corre las comparaciones y escribe HTML, tabla por variable, resumen JSON y gráficas."""
    salida.mkdir(parents=True, exist_ok=True)
    tablas, resumenes = [], {}
    for nombre, (_, _, con_html, _) in COMPARACIONES.items():
        referencia, actual = preparar_comparacion(conjuntos, nombre)
        logger.info(
            f"{nombre}: {len(referencia):,} filas de referencia y {len(actual):,} actuales"
        )
        snapshot = ejecutar_evidently(referencia, actual)
        if html and con_html:
            snapshot.save_html(str(salida / f"reporte_{nombre}.html"))
        tabla = tabla_drift(snapshot, referencia, actual, nombre)
        tablas.append(tabla)
        resumenes[nombre] = resumen_comparacion(tabla, snapshot, len(referencia), len(actual))

    tabla = pd.concat(tablas, ignore_index=True)
    tabla.to_csv(salida / "drift_por_variable.csv", index=False)

    referencia = conjuntos["referencia"]
    resumen = {
        "configuracion": {
            "metodo_numerico": METODO_DRIFT_NUMERICO,
            "metodo_categorico": METODO_DRIFT_CATEGORICO,
            "umbral": UMBRAL_DRIFT,
            "proporcion_drift_conjunto": PROPORCION_DRIFT_DATASET,
            "variables": VARIABLES,
            "categoricas": CATEGORICAS,
            "numericas": NUMERICAS,
            "objetivo": OBJETIVO,
            "excluidas": COLUMNAS_ID,
        },
        "periodos": {v: list(f) for v, f in PERIODOS_MONITOREO.items()},
        "comparaciones": resumenes,
        "calidad": {
            v: calidad(d, None if v == "referencia" else referencia) for v, d in conjuntos.items()
        },
        "estaciones_nuevas": {
            v: estaciones_nuevas(referencia, conjuntos[v]) for v in ("reciente", "actual")
        },
        "estacionalidad_demanda": demanda_mismo_mes(conjuntos),
    }
    (salida / "resumen_drift.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    top = variables_a_graficar(tabla)
    graficar_puntajes(tabla, salida / "grafica_drift_por_variable.png")
    if top:
        graficar_distribuciones(
            referencia, conjuntos["actual"], top, salida / "grafica_distribuciones.png"
        )
    graficar_demanda_diaria(conjuntos, salida / "grafica_demanda_diaria.png")
    return resumen


@app.command()
def main(
    salida: Annotated[Path, typer.Option(help="Carpeta de salida de la evidencia")] = DRIFT_DIR,
    html: Annotated[
        bool, typer.Option("--html/--sin-html", help="Guardar los reportes HTML")
    ] = True,
):
    """Genera el reporte de drift de Evidently del champion de P1."""
    resumen = generar(cargar_conjuntos(), salida, html)
    for nombre, datos in resumen["comparaciones"].items():
        logger.info(
            f"{nombre}: {datos['variables_con_drift']} de {datos['variables_monitoreadas']} "
            f"variables con drift ({datos['proporcion_con_drift']:.0%}); "
            f"drift del conjunto: {datos['drift_del_conjunto']}; "
            f"objetivo: {datos['objetivo']['puntaje']:.3f}; "
            f"orden: {', '.join(datos['variables_con_drift_por_puntaje'])}"
        )
    logger.success(f"Evidencia escrita en {salida}")


if __name__ == "__main__":
    app()
