"""P1 · Regresión de la demanda diaria por estación (D+1): baselines y modelos candidatos.

Cada modelo es una receta de `MODELOS`; `comun.ejecutar` la entrena, la evalúa y la registra.
La partición temporal y las métricas (MAE principal, RMSE y desviación Poisson) son las mismas de
la entrega anterior, para que la comparación contra el baseline sea válida.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import FECHA_FIN, FECHA_INICIO, MINIMO_POISSON, SEMILLA, VIAJES_DIR
from mibici.dataset import cargar_estaciones
from mibici.features import particion_temporal, preparar_datos_p1
from mibici.modeling.comun import (
    Modelo,
    Resultado,
    estabilidad_mensual,
    metricas_con_prefijo,
    parametros_conjuntos,
)
from mibici.plots import figuras_demanda

PREGUNTA = "p1"
OBJETIVO = "demanda"
VARIABLES = [
    "dia_semana",
    "mes",
    "es_fin_semana",
    "dia_anio",
    "es_festivo",
    "demanda_lag_1",
    "demanda_lag_7",
    "demanda_media_7",
    "capacidad",
    "latitud",
    "longitud",
]
VARIABLES_V2 = [*VARIABLES, "demanda_media_28"]


def construir_conjuntos(datos, estaciones):
    """Une capacidad y ubicación a la demanda diaria y aplica la partición temporal de config.py."""
    datos = datos.merge(
        estaciones[["estacion_id", "capacidad", "latitud", "longitud"]],
        on="estacion_id",
        how="left",
        validate="many_to_one",
    )
    return dict(zip(("train", "val", "test"), particion_temporal(datos)))


def preparar_conjuntos():
    """Carga los datos procesados y construye train, val y test de P1 (sin descargar nada)."""
    estaciones = cargar_estaciones()
    datos = preparar_datos_p1(VIAJES_DIR, estaciones, FECHA_INICIO, FECHA_FIN)
    return construir_conjuntos(datos, estaciones)


def datos_prediccion(conjuntos):
    """Conjunto de prueba de P1 y su columna objetivo, para verificar el modelo en producción."""
    return conjuntos["test"], OBJETIVO


def predecir_promedio(entrenamiento, datos):
    """Predice P1 con medias estación × día de train y respaldo en la media global de train."""
    medias = entrenamiento.groupby(["estacion_id", "dia_semana"])["demanda"].mean()
    indice = datos.set_index(["estacion_id", "dia_semana"]).index
    return medias.reindex(indice).fillna(entrenamiento["demanda"].mean()).to_numpy()


def predecir_naive(entrenamiento, datos):
    """Predice P1 con lag semanal; respalda NaN con promedio estación × día y global de train."""
    return np.where(
        datos["demanda_lag_7"].notna(),
        datos["demanda_lag_7"],
        predecir_promedio(entrenamiento, datos),
    )


def metricas_demanda(real, prediccion):
    """Evalúa P1; recorta predicciones a 1e-6 exclusivamente para la desviación Poisson."""
    return {
        "mae": float(mean_absolute_error(real, prediccion)),
        "rmse": float(root_mean_squared_error(real, prediccion)),
        "desviacion_poisson": float(
            mean_poisson_deviance(real, np.clip(prediccion, MINIMO_POISSON, None))
        ),
    }


def evaluar(conjuntos, predecir):
    """Evalúa P1 en los tres conjuntos, por rebanadas de demanda alta/baja y su estabilidad.

    Las rebanadas se definen con la demanda media de train; la estabilidad es la desviación
    estándar del MAE mes a mes en prueba.
    """
    entrenamiento = conjuntos["train"]
    medias = entrenamiento.groupby("estacion_id")["demanda"].mean()
    altas = medias.index[medias >= medias.median()]
    metricas, predicciones = {}, {}
    for conjunto, filas in conjuntos.items():
        prediccion = np.asarray(predecir(filas), dtype=float)
        predicciones[conjunto] = prediccion
        valores = metricas_demanda(filas["demanda"], prediccion)
        mascara = filas["estacion_id"].isin(altas).to_numpy()
        for rebanada, seleccion in (("alta", mascara), ("baja", ~mascara)):
            if seleccion.any():
                valores[f"mae_{rebanada}_demanda"] = float(
                    mean_absolute_error(
                        filas["demanda"].to_numpy()[seleccion], prediccion[seleccion]
                    )
                )
        metricas.update(metricas_con_prefijo(conjunto, valores))
    prueba = conjuntos["test"]
    metricas["test_mae_std_mensual"] = estabilidad_mensual(
        prueba["fecha"], prueba["demanda"].to_numpy(), predicciones["test"], mean_absolute_error
    )
    return metricas, predicciones


def _respaldo(conjuntos, columnas_faltantes):
    """Fracción de filas que usan el respaldo de un baseline, por conjunto."""
    return {f"{c}_fraccion_respaldo": float(columnas_faltantes(f)) for c, f in conjuntos.items()}


def entrenar_naive(conjuntos, hiperparametros):
    """Baseline de referencia de P1: misma demanda que hace 7 días (naive estacional)."""
    entrenamiento = conjuntos["train"]
    metricas, predicciones = evaluar(conjuntos, lambda f: predecir_naive(entrenamiento, f))
    metricas.update(_respaldo(conjuntos, lambda f: f["demanda_lag_7"].isna().mean()))
    params = parametros_conjuntos(
        "naive", ["estacion_id", "dia_semana", "demanda_lag_7"], conjuntos
    )
    params.update(
        {
            "clip_poisson": MINIMO_POISSON,
            "respaldo": "media train estacion×dia; luego media global train",
        }
    )
    return Resultado(
        metricas,
        params,
        figuras_demanda(conjuntos, predicciones),
        entrenamiento=entrenamiento[["estacion_id", "dia_semana", "demanda_lag_7", OBJETIVO]],
        objetivo=OBJETIVO,
    )


def entrenar_promedio(conjuntos, hiperparametros):
    """Baseline alterno de P1: media de train por estación y día de la semana."""
    entrenamiento = conjuntos["train"]
    vistos = entrenamiento.set_index(["estacion_id", "dia_semana"]).index
    metricas, predicciones = evaluar(conjuntos, lambda f: predecir_promedio(entrenamiento, f))
    metricas.update(
        _respaldo(
            conjuntos,
            lambda f: (~f.set_index(["estacion_id", "dia_semana"]).index.isin(vistos)).mean(),
        )
    )
    params = parametros_conjuntos("promedio", ["estacion_id", "dia_semana"], conjuntos)
    params.update({"clip_poisson": MINIMO_POISSON, "respaldo": "media global train"})
    return Resultado(
        metricas,
        params,
        figuras_demanda(conjuntos, predicciones),
        entrenamiento=entrenamiento[["estacion_id", "dia_semana", OBJETIVO]],
        objetivo=OBJETIVO,
    )


def entrenar_hgb(conjuntos, hiperparametros):
    """Candidato de P1: gradient boosting con pérdida Poisson, adecuada para conteos."""
    entrenamiento = conjuntos["train"]
    hiperparametros = dict(hiperparametros)
    variante = hiperparametros.pop("features", "v1")
    if variante not in ("v1", "v2"):
        raise ValueError("features debe ser 'v1' o 'v2'")
    variables = VARIABLES_V2 if variante == "v2" else VARIABLES
    modelo = HistGradientBoostingRegressor(
        loss="poisson", random_state=SEMILLA, early_stopping=False, **hiperparametros
    ).fit(entrenamiento[variables], entrenamiento[OBJETIVO])
    metricas, predicciones = evaluar(conjuntos, lambda f: modelo.predict(f[variables]))
    params = parametros_conjuntos("hgb", variables, conjuntos)
    params.update(
        {
            "features": variante,
            **hiperparametros,
            "loss": "poisson",
            "early_stopping": False,
            "random_state": SEMILLA,
            "clip_poisson": MINIMO_POISSON,
        }
    )
    return Resultado(
        metricas,
        params,
        figuras_demanda(conjuntos, predicciones),
        modelo=modelo,
        ejemplo=entrenamiento[variables].head(5),
        entrenamiento=entrenamiento[variables + [OBJETIVO]],
        objetivo=OBJETIVO,
        x_prueba=conjuntos["test"][variables],
    )


def _resultado_candidato(conjuntos, modelo, nombre, variables, hiperparametros):
    """Ajusta, evalúa y empaqueta un regresor servible de P1."""
    entrenamiento = conjuntos["train"]
    modelo.fit(entrenamiento[variables], entrenamiento[OBJETIVO])
    metricas, predicciones = evaluar(conjuntos, lambda f: modelo.predict(f[variables]))
    params = parametros_conjuntos(nombre, variables, conjuntos)
    params.update({**hiperparametros, "clip_poisson": MINIMO_POISSON})
    return Resultado(
        metricas,
        params,
        figuras_demanda(conjuntos, predicciones),
        modelo=modelo,
        ejemplo=entrenamiento[variables].head(5),
        entrenamiento=entrenamiento[variables + [OBJETIVO]],
        objetivo=OBJETIVO,
        x_prueba=conjuntos["test"][variables],
    )


def entrenar_rf(conjuntos, hiperparametros):
    """Candidato de P1: bosque aleatorio con criterio Poisson para conteos."""
    modelo = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            criterion="poisson", random_state=SEMILLA, n_jobs=1, **hiperparametros
        ),
    )
    resultado = _resultado_candidato(conjuntos, modelo, "rf", VARIABLES, hiperparametros)
    resultado.params.update({"criterion": "poisson", "random_state": SEMILLA, "n_jobs": 1})
    return resultado


def entrenar_poisson(conjuntos, hiperparametros):
    """Candidato lineal de P1: GLM Poisson con imputación y escalamiento."""
    modelo = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        PoissonRegressor(**hiperparametros),
    )
    return _resultado_candidato(conjuntos, modelo, "poisson", VARIABLES, hiperparametros)


MODELOS = {
    "naive": Modelo("naive", entrenar_naive, baseline=True),
    "promedio": Modelo("naive", entrenar_promedio, baseline=True),
    "hgb": Modelo(
        "boosting",
        entrenar_hgb,
        hiperparametros=("features", "learning_rate", "max_leaf_nodes", "max_iter"),
    ),
    "rf": Modelo("tree_ensemble", entrenar_rf, hiperparametros=("n_estimators", "max_depth")),
    "poisson": Modelo("linear", entrenar_poisson, hiperparametros=("alpha", "max_iter")),
}
