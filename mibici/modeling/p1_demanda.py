"""Baselines y regresión Poisson para la demanda diaria de P1."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, root_mean_squared_error

from mibici.config import ITERACIONES_HGB, MINIMO_POISSON, REJILLA_P1, SEMILLA
from mibici.features import particion_temporal
from mibici.modeling.comun import parametros_conjuntos, registrar_resultado, seleccionar_modelo
from mibici.plots import figura_demanda

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


def seleccionar_demanda(entrenamiento, validacion):
    """Selecciona HGB de P1 por MAE de validación; ajuste fijo solo en train, sin prueba."""
    return seleccionar_modelo(
        entrenamiento,
        validacion,
        VARIABLES,
        "demanda",
        lambda p: HistGradientBoostingRegressor(
            loss="poisson",
            random_state=SEMILLA,
            max_iter=ITERACIONES_HGB,
            early_stopping=False,
            **p,
        ),
        REJILLA_P1,
        mean_absolute_error,
    )


def entrenar_p1(datos, estaciones):
    """Ejecuta los tres runs de P1, selecciona con val y evalúa prueba una sola vez por modelo."""
    datos = datos.merge(
        estaciones[["estacion_id", "capacidad", "latitud", "longitud"]],
        on="estacion_id",
        how="left",
        validate="many_to_one",
    )
    conjuntos = dict(zip(("train", "val", "test"), particion_temporal(datos)))
    entrenamiento = conjuntos["train"]
    modelo, parametros, rejilla = seleccionar_demanda(entrenamiento, conjuntos["val"])
    medias = entrenamiento.groupby("estacion_id")["demanda"].mean()
    altas = medias.index[medias >= medias.median()]
    runs = []
    for nombre in ("naive-estacional", "promedio-estacion-dia", "hgb-poisson"):
        predicciones = {}
        metricas = {}
        for conjunto, filas in conjuntos.items():
            if nombre == "hgb-poisson":
                prediccion = modelo.predict(filas[VARIABLES])
            else:
                funcion = predecir_naive if nombre == "naive-estacional" else predecir_promedio
                prediccion = funcion(entrenamiento, filas)
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
            if nombre != "hgb-poisson":
                valores["fraccion_respaldo"] = float(
                    filas["demanda_lag_7"].isna().mean()
                    if nombre == "naive-estacional"
                    else (
                        ~filas.set_index(["estacion_id", "dia_semana"]).index.isin(
                            entrenamiento.set_index(["estacion_id", "dia_semana"]).index
                        )
                    ).mean()
                )
            metricas.update({f"{conjunto}_{k}": v for k, v in valores.items()})
        params = parametros_conjuntos(
            nombre,
            VARIABLES
            if nombre == "hgb-poisson"
            else ["estacion_id", "dia_semana"]
            + (["demanda_lag_7"] if nombre == "naive-estacional" else []),
            conjuntos,
        )
        params["clip_poisson"] = MINIMO_POISSON
        if nombre != "hgb-poisson":
            params["respaldo"] = (
                "media train estacion×dia; luego media global train"
                if nombre == "naive-estacional"
                else "media global train"
            )
        if nombre == "hgb-poisson":
            params.update(
                {
                    **parametros,
                    "max_iter": ITERACIONES_HGB,
                    "loss": "poisson",
                    "early_stopping": False,
                    "random_state": SEMILLA,
                }
            )
        runs.append(
            registrar_resultado(
                "P1",
                f"p1-{nombre}",
                params,
                metricas,
                {"demanda": figura_demanda(conjuntos, predicciones)},
                {"rejilla": rejilla} if nombre == "hgb-poisson" else None,
            )
        )
    return runs
