"""A/B offline de P1: champion contra challenger en la evaluación común de octubre de 2026.

Entregable 8, PR de DH4. Compara dos versiones **fijas** del mismo modelo registrado
(`mibici-equipo16-p1-demanda`) sobre exactamente las mismas filas de la ventana `actual` del
contrato de CH1 (docs/monitoring/contrato_datasets.md) y recomienda qué versión debe quedar
vigente y si conviene un esquema de Continuous Training.

Es una **comparación offline controlada**, no un experimento online: no hay tráfico aleatorizado
entre variantes; ambos modelos predicen el mismo conjunto histórico que ninguno usó para entrenar
ni para elegir hiperparámetros. Por eso la comparación es pareada: cada fila (estación × día) se
evalúa con A y con B, y la incertidumbre se estima con un bootstrap **por bloques de días**, porque
las filas del mismo día están correlacionadas (clima, eventos, calendario) y tratarlas como
independientes subestimaría el intervalo.

Regla de decisión (definida en `decidir`, antes de leer los resultados de octubre):

1. **Promover challenger** si reduce el MAE, el intervalo al 95 % de ΔMAE (B − A) queda por
   debajo de 0 y ninguna guarda falla: RMSE de B no peor que el de A y ningún segmento
   (estaciones conocidas/nuevas, laborable/fin de semana) empeora más de `TOLERANCIA_SEGMENTO_PCT`.
2. **Conservar champion** si B empeora el MAE con intervalo por encima de 0 o falla una guarda.
3. **Recoger más datos** si el intervalo incluye el 0: la diferencia no se distingue del ruido.

Las versiones se pasan por número o por alias; un alias se resuelve una sola vez a su número
y ese número es el que queda en los resultados. El alias `champion` no se modifica aquí.

Desde la raíz, con el servidor de MLflow en marcha y los conjuntos de `make monitoring-data`:

    python -m mibici.monitoring.ab_compare                      # champion vs challenger
    python -m mibici.monitoring.ab_compare --version-a 2 --version-b 3
"""

from dataclasses import dataclass
import json
from pathlib import Path

from loguru import logger
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import typer

from mibici.config import (
    ALIAS_PRODUCCION,
    ALIAS_RETADOR,
    INICIO_PRUEBA,
    MANIFIESTO_MONITOREO,
    MINIMO_POISSON,
    MLFLOW_TRACKING_URI,
    MODELOS_REGISTRADOS,
    PROJ_ROOT,
    SEMILLA,
)
from mibici.monitoring.datasets import OBJETIVO, VARIABLES, cargar_conjuntos, huella

matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = typer.Typer()

NOMBRE = MODELOS_REGISTRADOS["p1"]
SALIDA_DIR = PROJ_ROOT / "docs" / "reportes" / "actividad10"
N_BOOTSTRAP = 2000
NIVEL_CONFIANZA = 0.95
TOLERANCIA_SEGMENTO_PCT = 2.0  # un segmento no puede empeorar más que esto para promover

COLOR_A = "#2a6fb0"
COLOR_B = "#d9822b"


@dataclass(frozen=True)
class Version:
    """Una versión fija del registry: número, corrida y de qué referencia se resolvió."""

    etiqueta: str  # "A" o "B"
    numero: str
    run_id: str | None = None
    referencia: str | None = None  # alias o número con que se pidió
    ventana_train: str | None = None

    @property
    def uri(self) -> str:
        return f"models:/{NOMBRE}/{self.numero}"


# --- métricas ---------------------------------------------------------------------------------


def metricas(real, prediccion) -> dict[str, float]:
    """MAE (principal), RMSE, desviación Poisson, sesgo y MAE relativo a la demanda media."""
    real = np.asarray(real, dtype=float)
    prediccion = np.asarray(prediccion, dtype=float)
    error = prediccion - real
    recortada = np.clip(prediccion, MINIMO_POISSON, None)
    # Desviación Poisson unitaria; y·log(y/ŷ) vale 0 cuando y = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        termino = np.where(real > 0, real * np.log(real / recortada), 0.0)
    media = float(real.mean())
    mae = float(np.abs(error).mean())
    return {
        "mae": mae,
        "rmse": float(np.sqrt((error**2).mean())),
        "desviacion_poisson": float(np.mean(2 * (termino - (real - recortada)))),
        "sesgo": float(error.mean()),
        "mae_relativo_pct": 100 * mae / media if media else float("nan"),
        "media_real": media,
        "media_prediccion": float(prediccion.mean()),
        "n": len(real),
    }


def diferencia(valor_a: float, valor_b: float) -> dict[str, float]:
    """Diferencia B − A absoluta y relativa a A (negativa = B tiene menos error)."""
    absoluta = valor_b - valor_a
    return {"absoluta": absoluta, "relativa_pct": 100 * absoluta / valor_a if valor_a else np.nan}


# --- segmentos y comparación pareada ----------------------------------------------------------


def segmentos(evaluacion: pd.DataFrame, referencia: pd.DataFrame) -> dict[str, np.ndarray]:
    """Máscaras de los segmentos de interés para el drift observado por LH3.

    - estaciones conocidas/nuevas: si la estación aparece en la referencia (lo que vio el champion);
    - laborable/fin de semana;
    - demanda alta/baja: estaciones conocidas con media de referencia ≥ / < la mediana.
    """
    conocidas = set(referencia["estacion_id"].unique())
    es_conocida = evaluacion["estacion_id"].isin(conocidas).to_numpy()
    fin = evaluacion["es_fin_semana"].astype(int).to_numpy() == 1
    medias = referencia.groupby("estacion_id")[OBJETIVO].mean()
    altas = set(medias.index[medias >= medias.median()])
    es_alta = evaluacion["estacion_id"].isin(altas).to_numpy()
    return {
        "todas": np.ones(len(evaluacion), dtype=bool),
        "estaciones_conocidas": es_conocida,
        "estaciones_nuevas": ~es_conocida,
        "laborable": ~fin,
        "fin_de_semana": fin,
        "conocidas_demanda_alta": es_conocida & es_alta,
        "conocidas_demanda_baja": es_conocida & ~es_alta,
    }


def tabla_segmentos(real, pred_a, pred_b, mascaras: dict[str, np.ndarray]) -> pd.DataFrame:
    """MAE y RMSE de A y B por segmento, con la diferencia relativa de MAE."""
    filas = []
    for nombre, mascara in mascaras.items():
        if not mascara.any():
            continue
        a = metricas(real[mascara], pred_a[mascara])
        b = metricas(real[mascara], pred_b[mascara])
        filas.append(
            {
                "segmento": nombre,
                "n": a["n"],
                "demanda_media": a["media_real"],
                "mae_a": a["mae"],
                "mae_b": b["mae"],
                "delta_mae": b["mae"] - a["mae"],
                "delta_mae_pct": diferencia(a["mae"], b["mae"])["relativa_pct"],
                "rmse_a": a["rmse"],
                "rmse_b": b["rmse"],
                "sesgo_a": a["sesgo"],
                "sesgo_b": b["sesgo"],
            }
        )
    return pd.DataFrame(filas)


def tabla_diaria(evaluacion: pd.DataFrame, pred_a, pred_b) -> pd.DataFrame:
    """Por día: demanda media, MAE de A y B y su diferencia (unidad del análisis pareado)."""
    real = evaluacion[OBJETIVO].to_numpy(dtype=float)
    datos = pd.DataFrame(
        {
            "fecha": pd.to_datetime(evaluacion["fecha"]).dt.normalize(),
            "real": real,
            "error_a": np.abs(pred_a - real),
            "error_b": np.abs(pred_b - real),
        }
    )
    diaria = datos.groupby("fecha").agg(
        n=("real", "size"),
        demanda_media=("real", "mean"),
        mae_a=("error_a", "mean"),
        mae_b=("error_b", "mean"),
    )
    diaria["delta_mae"] = diaria["mae_b"] - diaria["mae_a"]
    return diaria.reset_index()


def bootstrap_por_dias(
    fechas,
    error_a,
    error_b,
    n: int = N_BOOTSTRAP,
    nivel: float = NIVEL_CONFIANZA,
    semilla: int = SEMILLA,
) -> dict[str, float]:
    """Intervalo de ΔMAE = MAE(B) − MAE(A) remuestreando días completos con reemplazo.

    Cada réplica toma tantos días como tiene la evaluación y suma los errores de todas sus filas,
    así el MAE de la réplica pondera por filas igual que el MAE global.
    """
    fechas = pd.to_datetime(pd.Series(fechas)).dt.normalize().to_numpy()
    dias, posicion = np.unique(fechas, return_inverse=True)
    suma_a = np.bincount(posicion, weights=np.asarray(error_a, dtype=float))
    suma_b = np.bincount(posicion, weights=np.asarray(error_b, dtype=float))
    filas = np.bincount(posicion).astype(float)
    azar = np.random.default_rng(semilla)
    muestra = azar.integers(0, len(dias), size=(n, len(dias)))
    total = filas[muestra].sum(axis=1)
    deltas = (suma_b[muestra].sum(axis=1) - suma_a[muestra].sum(axis=1)) / total
    mae_a = suma_a[muestra].sum(axis=1) / total
    alfa = (1 - nivel) / 2
    return {
        "delta_mae": float((suma_b.sum() - suma_a.sum()) / filas.sum()),
        "ic_inferior": float(np.quantile(deltas, alfa)),
        "ic_superior": float(np.quantile(deltas, 1 - alfa)),
        "ic_inferior_pct": float(np.quantile(100 * deltas / mae_a, alfa)),
        "ic_superior_pct": float(np.quantile(100 * deltas / mae_a, 1 - alfa)),
        "prob_b_mejor": float((deltas < 0).mean()),
        "replicas": int(n),
        "dias": len(dias),
        "nivel": nivel,
        "_deltas": deltas,  # para la gráfica; no se serializa
    }


def pruebas_pareadas(diaria: pd.DataFrame, por_estacion: pd.DataFrame) -> dict[str, float]:
    """Proporción de días y estaciones donde B gana y prueba de Wilcoxon sobre el MAE diario."""
    resultado = {
        "dias_b_mejor": int((diaria["delta_mae"] < 0).sum()),
        "dias": len(diaria),
        "estaciones_b_mejor": int((por_estacion["delta_mae"] < 0).sum()),
        "estaciones": len(por_estacion),
    }
    diferencias = diaria["delta_mae"].to_numpy()
    if len(diferencias) >= 6 and np.any(diferencias != 0):
        prueba = wilcoxon(diferencias)
        resultado["wilcoxon_estadistico"] = float(prueba.statistic)
        resultado["wilcoxon_p"] = float(prueba.pvalue)
    return resultado


def tabla_estaciones(evaluacion: pd.DataFrame, pred_a, pred_b) -> pd.DataFrame:
    """MAE de A y B por estación en la ventana de evaluación."""
    real = evaluacion[OBJETIVO].to_numpy(dtype=float)
    datos = pd.DataFrame(
        {
            "estacion_id": evaluacion["estacion_id"].to_numpy(),
            "error_a": np.abs(pred_a - real),
            "error_b": np.abs(pred_b - real),
        }
    )
    tabla = datos.groupby("estacion_id").agg(mae_a=("error_a", "mean"), mae_b=("error_b", "mean"))
    tabla["delta_mae"] = tabla["mae_b"] - tabla["mae_a"]
    return tabla.reset_index()


# --- decisión ---------------------------------------------------------------------------------


def decidir(
    global_a: dict,
    global_b: dict,
    intervalo: dict,
    por_segmento: pd.DataFrame,
    tolerancia_segmento_pct: float = TOLERANCIA_SEGMENTO_PCT,
) -> dict:
    """Aplica la regla de promoción del docstring del módulo y devuelve decisión y motivos."""
    guardas = []
    if global_b["rmse"] > global_a["rmse"]:
        guardas.append(
            f"RMSE de B ({global_b['rmse']:.3f}) peor que el de A ({global_a['rmse']:.3f})"
        )
    for fila in por_segmento.itertuples():
        if fila.segmento != "todas" and fila.delta_mae_pct > tolerancia_segmento_pct:
            guardas.append(
                f"segmento {fila.segmento}: MAE de B {fila.delta_mae_pct:+.2f} % frente a A "
                f"(tolerancia {tolerancia_segmento_pct:.1f} %)"
            )
    mejora = global_b["mae"] < global_a["mae"]
    excluye_cero_abajo = intervalo["ic_superior"] < 0
    excluye_cero_arriba = intervalo["ic_inferior"] > 0
    ic = f"IC {intervalo['nivel']:.0%} de ΔMAE [{intervalo['ic_inferior']:+.3f}, {intervalo['ic_superior']:+.3f}]"
    if mejora and excluye_cero_abajo and not guardas:
        decision = "promover_challenger"
        motivos = [f"B reduce el MAE y el {ic} queda por debajo de 0", "Ninguna guarda falla"]
    elif excluye_cero_arriba or guardas:
        decision = "conservar_champion"
        motivos = ([f"B empeora el MAE ({ic})"] if excluye_cero_arriba else []) + guardas
    else:
        decision = "recoger_mas_datos"
        motivos = [f"El {ic} incluye el 0: la diferencia no se distingue del ruido entre días"]
    return {"decision": decision, "motivos": motivos, "guardas_fallidas": guardas}


# --- Continuous Training ----------------------------------------------------------------------

# Umbrales iniciales de la política de reentrenamiento. Son hipótesis por calibrar con más meses
# de historia, no valores universales (docs/reportes/actividad10/README.md).
UMBRAL_DRIFT = 0.1  # mismo umbral por columna que el reporte de Evidently (LH3)
UMBRAL_ESTACIONES_NUEVAS_PCT = 10.0  # % de filas de estaciones que el modelo no vio
# Aumento del MAE relativo (MAE / demanda media) del champion frente a su periodo base. Se usa el
# relativo porque el MAE absoluto crece con el nivel de demanda aunque el modelo no empeore.
UMBRAL_DEGRADACION_MAE_PCT = 15.0


def politica_continuous_training(
    drift: dict | None, actual: dict, base: dict, decision_ab: str
) -> dict:
    """Evalúa los disparadores de reentrenamiento y separa *reentrenar* de *promover*.

    `drift` es `docs/evidencias/drift/resumen_drift.json` (PR de LH3). Primero se descarta un
    problema de datos (no se entrena con datos corruptos); luego se pide evidencia de cambio en la
    demanda **y** de impacto (degradación o estaciones no vistas). La promoción no la decide la
    alerta: la decide el A/B (`decision_ab`).

    `actual` y `base` son las `metricas` del champion en la ventana actual y en su periodo base;
    la degradación se mide con el MAE relativo a la demanda media de cada periodo.
    """
    disparadores = []

    def agregar(nombre, valor, umbral, activo, fuente):
        disparadores.append(
            {
                "disparador": nombre,
                "valor": valor,
                "umbral": umbral,
                "activo": bool(activo),
                "fuente": fuente,
            }
        )

    calidad_ok = True
    if drift:
        calidad = drift.get("calidad", {}).get("actual", {})
        problemas = (
            len(calidad.get("columnas_faltantes", []))
            + calidad.get("filas_duplicadas", 0)
            + calidad.get("claves_duplicadas", 0)
            + calidad.get("demanda_negativa", 0)
        )
        calidad_ok = problemas == 0
        agregar(
            "problemas de calidad en la ventana actual",
            problemas,
            0,
            not calidad_ok,
            "Evidently / resumen_drift.json",
        )
        comparacion = drift["comparaciones"]["referencia_vs_actual"]
        objetivo = comparacion["objetivo"]["puntaje"]
        agregar(
            "drift del objetivo (demanda)",
            objetivo,
            UMBRAL_DRIFT,
            objetivo >= UMBRAL_DRIFT,
            "Evidently / Wasserstein",
        )
        historial = len(comparacion.get("drift_por_grupo", {}).get("historial de demanda", []))
        agregar(
            "variables de historial de demanda con drift",
            historial,
            1,
            historial >= 1,
            "Evidently / Wasserstein",
        )
        nuevas = drift.get("estaciones_nuevas", {}).get("actual", {})
        pct = nuevas.get("filas_estaciones_nuevas_pct", 0.0)
        agregar(
            "% de filas de estaciones no vistas",
            pct,
            UMBRAL_ESTACIONES_NUEVAS_PCT,
            pct >= UMBRAL_ESTACIONES_NUEVAS_PCT,
            "catálogo vs referencia",
        )
    relativo_base = base["mae_relativo_pct"]
    degradacion = (
        100 * (actual["mae_relativo_pct"] - relativo_base) / relativo_base
        if relativo_base
        else 0.0
    )
    agregar(
        "aumento del MAE relativo del champion vs su periodo base (%)",
        degradacion,
        UMBRAL_DEGRADACION_MAE_PCT,
        degradacion >= UMBRAL_DEGRADACION_MAE_PCT,
        "A/B: octubre vs jul–sep 2026",
    )

    activos = {d["disparador"] for d in disparadores if d["activo"]}
    cambio = bool(
        activos & {"drift del objetivo (demanda)", "variables de historial de demanda con drift"}
    )
    impacto = bool(
        activos
        & {
            "% de filas de estaciones no vistas",
            "aumento del MAE relativo del champion vs su periodo base (%)",
        }
    )
    if not calidad_ok:
        accion = "corregir_pipeline_de_datos"
    elif cambio and impacto:
        accion = "reentrenar_candidato"
    elif cambio:
        accion = "investigar_drift"
    else:
        accion = "seguir_monitoreando"
    return {
        "accion": accion,
        "promocion": decision_ab,
        "champion_actual": {k: actual[k] for k in ("mae", "mae_relativo_pct", "media_real")},
        "champion_base": {k: base[k] for k in ("mae", "mae_relativo_pct", "media_real")},
        "aumento_mae_absoluto_pct": 100 * (actual["mae"] - base["mae"]) / base["mae"],
        "disparadores": disparadores,
    }


def cargar_resumen_drift(
    ruta: Path = PROJ_ROOT / "docs" / "evidencias" / "drift" / "resumen_drift.json",
):
    """Resumen del reporte de Evidently de LH3, si existe."""
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


# --- registry ---------------------------------------------------------------------------------


def resolver_version(cliente, referencia: str, etiqueta: str) -> Version:
    """Convierte un alias o un número en una versión fija del registry de P1."""
    if str(referencia).isdigit():
        version = cliente.get_model_version(NOMBRE, str(referencia))
    else:
        version = cliente.get_model_version_by_alias(NOMBRE, referencia)
    tags = getattr(version, "tags", {}) or {}
    return Version(
        etiqueta=etiqueta,
        numero=str(version.version),
        run_id=version.run_id,
        referencia=str(referencia),
        ventana_train=tags.get("ventana_train"),
    )


def cargar_modelo(version: Version):
    """Carga la versión por número con pyfunc, como la API: aplica la firma del modelo."""
    import mlflow

    return mlflow.pyfunc.load_model(version.uri)


def predecir(modelo, evaluacion: pd.DataFrame) -> np.ndarray:
    """Predicciones de un modelo sobre las variables de la firma; exige valores finitos."""
    prediccion = np.asarray(modelo.predict(evaluacion[VARIABLES]), dtype=float).ravel()
    if len(prediccion) != len(evaluacion) or not np.isfinite(prediccion).all():
        raise ValueError("El modelo devolvió predicciones faltantes o no finitas")
    return prediccion


# --- comparación completa ---------------------------------------------------------------------


def comparar(
    evaluacion: pd.DataFrame,
    referencia: pd.DataFrame,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    """Todas las tablas del A/B a partir de las predicciones de A y B sobre las mismas filas."""
    if len(pred_a) != len(evaluacion) or len(pred_b) != len(evaluacion):
        raise ValueError("A y B deben predecir exactamente las filas de la evaluación común")
    real = evaluacion[OBJETIVO].to_numpy(dtype=float)
    global_a, global_b = metricas(real, pred_a), metricas(real, pred_b)
    por_segmento = tabla_segmentos(real, pred_a, pred_b, segmentos(evaluacion, referencia))
    diaria = tabla_diaria(evaluacion, pred_a, pred_b)
    por_estacion = tabla_estaciones(evaluacion, pred_a, pred_b)
    intervalo = bootstrap_por_dias(
        evaluacion["fecha"], np.abs(pred_a - real), np.abs(pred_b - real), n=n_bootstrap
    )
    return {
        "global_a": global_a,
        "global_b": global_b,
        "por_segmento": por_segmento,
        "diaria": diaria,
        "por_estacion": por_estacion,
        "intervalo": intervalo,
        "pareado": pruebas_pareadas(diaria, por_estacion),
        "decision": decidir(global_a, global_b, intervalo, por_segmento),
    }


def tabla_ab(resultado: dict, a: Version, b: Version, evaluacion: pd.DataFrame) -> pd.DataFrame:
    """Tabla comparativa del reporte: versión, ventanas, N, métricas y diferencia contra A."""
    fechas = pd.to_datetime(evaluacion["fecha"])
    ventana_eval = f"{fechas.min().date()}/{fechas.max().date()}"
    filas = []
    for version, m in ((a, resultado["global_a"]), (b, resultado["global_b"])):
        base = resultado["global_a"]
        filas.append(
            {
                "modelo": f"{version.etiqueta} · {'champion' if version.etiqueta == 'A' else 'challenger'}",
                "version_mlflow": version.numero,
                "run_id": version.run_id,
                "ventana_train": version.ventana_train,
                "ventana_evaluacion": ventana_eval,
                "n": m["n"],
                "mae": m["mae"],
                "rmse": m["rmse"],
                "desviacion_poisson": m["desviacion_poisson"],
                "sesgo": m["sesgo"],
                "mae_relativo_pct": m["mae_relativo_pct"],
                "delta_mae_vs_a": m["mae"] - base["mae"],
                "delta_mae_vs_a_pct": diferencia(base["mae"], m["mae"])["relativa_pct"],
                "delta_rmse_vs_a_pct": diferencia(base["rmse"], m["rmse"])["relativa_pct"],
            }
        )
    return pd.DataFrame(filas)


# --- gráficas ---------------------------------------------------------------------------------


def grafica_diaria(diaria: pd.DataFrame, ruta: Path, a: Version, b: Version) -> None:
    """MAE diario de A y B (arriba) y ΔMAE por día (abajo)."""
    fig, (arriba, abajo) = plt.subplots(
        2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    arriba.plot(diaria["fecha"], diaria["mae_a"], color=COLOR_A, lw=2, label=f"A · v{a.numero}")
    arriba.plot(diaria["fecha"], diaria["mae_b"], color=COLOR_B, lw=2, label=f"B · v{b.numero}")
    arriba.set_ylabel("MAE (viajes/estación/día)")
    arriba.legend(frameon=False, loc="upper right")
    arriba.set_title("MAE diario en la evaluación común (octubre 2026)", loc="left", fontsize=11)
    colores = np.where(diaria["delta_mae"] < 0, COLOR_B, "#9aa5b1")
    abajo.bar(diaria["fecha"], diaria["delta_mae"], color=colores, width=0.8)
    abajo.axhline(0, color="#333", lw=0.8)
    abajo.set_ylabel("ΔMAE (B − A)")
    for eje in (arriba, abajo):
        eje.spines[["top", "right"]].set_visible(False)
        eje.grid(axis="y", alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def grafica_segmentos(por_segmento: pd.DataFrame, ruta: Path, a: Version, b: Version) -> None:
    """MAE de A y B por segmento, con la diferencia relativa anotada."""
    datos = por_segmento.iloc[::-1]
    y = np.arange(len(datos))
    fig, eje = plt.subplots(figsize=(9, 4.2))
    eje.barh(y + 0.2, datos["mae_a"], height=0.38, color=COLOR_A, label=f"A · v{a.numero}")
    eje.barh(y - 0.2, datos["mae_b"], height=0.38, color=COLOR_B, label=f"B · v{b.numero}")
    for posicion, fila in zip(y, datos.itertuples()):
        eje.text(
            max(fila.mae_a, fila.mae_b) + 0.15,
            posicion,
            f"{fila.delta_mae_pct:+.1f} %",
            va="center",
            fontsize=9,
        )
    eje.set_yticks(y, [s.replace("_", " ") for s in datos["segmento"]])
    eje.set_xlabel("MAE (viajes/estación/día)")
    eje.set_title("MAE por segmento y cambio de B frente a A", loc="left", fontsize=11)
    eje.legend(frameon=False, loc="lower right")
    eje.spines[["top", "right"]].set_visible(False)
    eje.set_xlim(0, datos[["mae_a", "mae_b"]].to_numpy().max() * 1.18)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def grafica_bootstrap(intervalo: dict, ruta: Path) -> None:
    """Distribución bootstrap por días de ΔMAE con el intervalo y el cero marcados."""
    fig, eje = plt.subplots(figsize=(7, 3.2))
    eje.hist(intervalo["_deltas"], bins=50, color="#8fb3d9", edgecolor="white")
    eje.axvline(0, color="#333", lw=1, label="sin diferencia")
    for clave in ("ic_inferior", "ic_superior"):
        eje.axvline(intervalo[clave], color=COLOR_B, ls="--", lw=1.2)
    eje.axvline(intervalo["delta_mae"], color=COLOR_B, lw=2, label="ΔMAE observado")
    eje.set_xlabel("ΔMAE = MAE(B) − MAE(A)")
    eje.set_ylabel("réplicas")
    eje.set_title(
        f"Bootstrap por días ({intervalo['replicas']} réplicas, IC {intervalo['nivel']:.0%})",
        loc="left",
        fontsize=11,
    )
    eje.legend(frameon=False, fontsize=8)
    eje.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# --- salida -----------------------------------------------------------------------------------


def _serializable(valor):
    if isinstance(valor, dict):
        return {k: _serializable(v) for k, v in valor.items() if not str(k).startswith("_")}
    if isinstance(valor, list | tuple):
        return [_serializable(v) for v in valor]
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, np.floating | float):
        return round(float(valor), 6)
    return valor


def guardar(
    resultado: dict, a: Version, b: Version, evaluacion: pd.DataFrame, salida: Path = SALIDA_DIR
) -> dict:
    """Escribe tablas CSV, resumen JSON y gráficas; devuelve el resumen (sin fecha: reproducible)."""
    salida.mkdir(parents=True, exist_ok=True)
    tabla = tabla_ab(resultado, a, b, evaluacion)
    tabla.to_csv(salida / "tabla_ab.csv", index=False, float_format="%.4f")
    resultado["por_segmento"].to_csv(
        salida / "ab_por_segmento.csv", index=False, float_format="%.4f"
    )
    diaria = resultado["diaria"].assign(fecha=lambda d: d["fecha"].dt.date)
    diaria.to_csv(salida / "ab_por_dia.csv", index=False, float_format="%.4f")
    resultado["por_estacion"].to_csv(
        salida / "ab_por_estacion.csv", index=False, float_format="%.4f"
    )
    grafica_diaria(resultado["diaria"], salida / "grafica_ab_mae_diario.png", a, b)
    grafica_segmentos(resultado["por_segmento"], salida / "grafica_ab_segmentos.png", a, b)
    grafica_bootstrap(resultado["intervalo"], salida / "grafica_ab_bootstrap.png")
    resumen = _serializable(
        {
            "modelo": NOMBRE,
            "tipo": "comparación offline controlada (pareada, mismas filas)",
            "versiones": {
                v.etiqueta: {
                    "version": v.numero,
                    "uri": v.uri,
                    "run_id": v.run_id,
                    "pedida_como": v.referencia,
                    "ventana_train": v.ventana_train,
                }
                for v in (a, b)
            },
            "evaluacion": {
                "ventana": "actual",
                "filas": len(evaluacion),
                "estaciones": int(evaluacion["estacion_id"].nunique()),
                "huella": huella(evaluacion),
            },
            "global": {"A": resultado["global_a"], "B": resultado["global_b"]},
            "intervalo_delta_mae": resultado["intervalo"],
            "pareado": resultado["pareado"],
            "regla": {
                "nivel_confianza": NIVEL_CONFIANZA,
                "replicas_bootstrap": resultado["intervalo"]["replicas"],
                "tolerancia_segmento_pct": TOLERANCIA_SEGMENTO_PCT,
            },
            "decision": resultado["decision"],
            "continuous_training": resultado.get("continuous_training"),
        }
    )
    (salida / "resultados_ab.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return resumen


def registrar_en_mlflow(resumen: dict, salida: Path) -> str:
    """Corrida de análisis con métricas del A/B y artefactos; no toca aliases ni versiones."""
    import mlflow

    from mibici import tracking

    a, b = resumen["global"]["A"], resumen["global"]["B"]
    params = {
        "version_a": resumen["versiones"]["A"]["version"],
        "version_b": resumen["versiones"]["B"]["version"],
        "ventana_evaluacion": "2026-10-01/2026-10-31",
        "huella_evaluacion": resumen["evaluacion"]["huella"],
        "decision": resumen["decision"]["decision"],
    }
    etiquetas = {
        "stage_of_work": "analysis",
        "model_family": "ab_offline",
        "entregable": "8-drift-ab",
    }
    with tracking.iniciar_run(
        "p1", "p1-ab-champion-vs-challenger-oct-2026", params, etiquetas
    ) as run:
        intervalo = resumen["intervalo_delta_mae"]
        tracking.registrar_metricas(
            {
                "test_mae_a": a["mae"],
                "test_mae_b": b["mae"],
                "test_rmse_a": a["rmse"],
                "test_rmse_b": b["rmse"],
                "test_delta_mae": intervalo["delta_mae"],
                "test_delta_mae_ic_inferior": intervalo["ic_inferior"],
                "test_delta_mae_ic_superior": intervalo["ic_superior"],
                "test_prob_b_mejor": intervalo["prob_b_mejor"],
            }
        )
        mlflow.log_artifacts(str(salida), "ab")
        if MANIFIESTO_MONITOREO.exists():
            mlflow.log_artifact(str(MANIFIESTO_MONITOREO), "data")
        return run.info.run_id


def generar(
    version_a: str = ALIAS_PRODUCCION,
    version_b: str = ALIAS_RETADOR,
    salida: Path = SALIDA_DIR,
    n_bootstrap: int = N_BOOTSTRAP,
    registrar: bool = True,
) -> dict:
    """Resuelve y congela las versiones, predice la evaluación común, compara y guarda todo."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    cliente = MlflowClient()
    a = resolver_version(cliente, version_a, "A")
    b = resolver_version(cliente, version_b, "B")
    if a.numero == b.numero:
        raise ValueError(f"A y B son la misma versión ({a.numero}); no hay nada que comparar")
    conjuntos = cargar_conjuntos()
    evaluacion = conjuntos["actual"]
    logger.info(f"A = v{a.numero} ({a.referencia}), B = v{b.numero} ({b.referencia})")
    modelo_a = cargar_modelo(a)
    pred_a = predecir(modelo_a, evaluacion)
    pred_b = predecir(cargar_modelo(b), evaluacion)
    resultado = comparar(evaluacion, conjuntos["referencia"], pred_a, pred_b, n_bootstrap)
    # Periodo base del champion: jul–sep 2026, su prueba original (no lo vio al entrenar)
    reciente = conjuntos["reciente"]
    base = reciente.loc[pd.to_datetime(reciente["fecha"]) >= INICIO_PRUEBA]
    resultado["continuous_training"] = politica_continuous_training(
        cargar_resumen_drift(),
        resultado["global_a"],
        metricas(base[OBJETIVO], predecir(modelo_a, base)),
        resultado["decision"]["decision"],
    )
    resumen = guardar(resultado, a, b, evaluacion, salida)
    if registrar:
        resumen["run_id_analisis"] = registrar_en_mlflow(resumen, salida)
    return resumen


@app.command()
def main(
    version_a: str = typer.Option(ALIAS_PRODUCCION, help="Versión A: número o alias"),
    version_b: str = typer.Option(ALIAS_RETADOR, help="Versión B: número o alias"),
    replicas: int = typer.Option(N_BOOTSTRAP, help="Réplicas del bootstrap por días"),
    mlflow_run: bool = typer.Option(
        True, "--mlflow/--sin-mlflow", help="Registrar una corrida de análisis en MLflow"
    ),
):
    """A/B offline de P1 en octubre de 2026 y recomendación de versión vigente."""
    resumen = generar(version_a, version_b, n_bootstrap=replicas, registrar=mlflow_run)
    a, b = resumen["global"]["A"], resumen["global"]["B"]
    ic = resumen["intervalo_delta_mae"]
    logger.info(
        "MAE A {:.4f} | B {:.4f} | ΔMAE {:+.4f} [{:+.4f}, {:+.4f}] | P(B mejor) {:.3f}",
        a["mae"],
        b["mae"],
        ic["delta_mae"],
        ic["ic_inferior"],
        ic["ic_superior"],
        ic["prob_b_mejor"],
    )
    logger.success(f"Decisión: {resumen['decision']['decision']} — evidencia en {SALIDA_DIR}")


if __name__ == "__main__":
    app()
