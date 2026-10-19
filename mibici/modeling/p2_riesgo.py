"""Clasificación del proxy de riesgo operativo de P2, sin inventario observado."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mibici.config import (
    ALFA_RIESGO,
    CLASES_RIESGO,
    ITERACIONES_HGB,
    REJILLA_LOGISTICA,
    REJILLA_P2,
    SEMILLA,
)
from mibici.features import particion_temporal
from mibici.modeling.comun import parametros_conjuntos, registrar_resultado, seleccionar_modelo
from mibici.plots import figura_confusion

NUMERICAS = ["flujo_lag_1", "flujo_lag_56", "capacidad"]
CATEGORICAS = ["hora", "dia_semana"]
VARIABLES = NUMERICAS + CATEGORICAS
VARIABLES_HGB = VARIABLES + ["latitud", "longitud"]


def predecir_persistencia(datos, alpha=ALFA_RIESGO):
    """Aplica la regla P2 al lag semanal; NaN devuelve normal, clase mayoritaria de train real."""
    umbral = alpha * datos["capacidad"]
    return np.select(
        [datos["flujo_lag_56"] >= umbral, datos["flujo_lag_56"] <= -umbral],
        ["sin_bicicletas", "sin_anclajes"],
        default="normal",
    )


def metricas_riesgo(real, prediccion, puntuaciones, clases):
    """Evalúa P2 con F1 macro y AP uno contra resto; persistencia usa indicadores binarios."""
    metricas = {
        "f1_macro": float(
            f1_score(real, prediccion, labels=CLASES_RIESGO, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(real, prediccion)),
    }
    recalls = recall_score(real, prediccion, labels=CLASES_RIESGO, average=None, zero_division=0)
    metricas.update(
        {f"recall_{clase}": float(valor) for clase, valor in zip(CLASES_RIESGO, recalls)}
    )
    for clase in CLASES_RIESGO[1:]:
        indice = list(clases).index(clase) if clase in clases else None
        puntuacion = puntuaciones[:, indice] if indice is not None else np.zeros(len(real))
        binario = np.asarray(real) == clase
        metricas[f"ap_{clase}"] = (
            float(average_precision_score(binario, puntuacion)) if binario.any() else 0.0
        )
    metricas["ap_macro_riesgo"] = float(
        np.mean([metricas[f"ap_{clase}"] for clase in CLASES_RIESGO[1:]])
    )
    return metricas


def construir_logistica(parametros):
    """Construye el preprocesamiento P2 aprendido en train y logística balanceada."""
    transformador = ColumnTransformer(
        [
            (
                "numericas",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                NUMERICAS,
            ),
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ]
    )
    return Pipeline(
        [
            ("variables", transformador),
            (
                "modelo",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=SEMILLA, **parametros
                ),
            ),
        ]
    )


def seleccionar_riesgo(entrenamiento, validacion, tipo):
    """Selecciona P2 por F1 macro de val, sin recibir test ni reajustar con val."""
    es_logistica = tipo == "logistic-regression"
    construir = (
        construir_logistica
        if es_logistica
        else lambda p: HistGradientBoostingClassifier(
            random_state=SEMILLA, max_iter=ITERACIONES_HGB, early_stopping=False, **p
        )
    )
    return seleccionar_modelo(
        entrenamiento,
        validacion,
        VARIABLES if es_logistica else VARIABLES_HGB,
        "riesgo",
        construir,
        REJILLA_LOGISTICA if es_logistica else REJILLA_P2,
        lambda real, pred: f1_score(
            real, pred, labels=CLASES_RIESGO, average="macro", zero_division=0
        ),
        maximizar=True,
    )


def entrenar_p2(datos, estaciones):
    """Ejecuta cuatro runs P2 sobre particiones temporales; prueba se predice una vez por run."""
    datos = datos.merge(
        estaciones[["estacion_id", "latitud", "longitud"]],
        on="estacion_id",
        how="left",
        validate="many_to_one",
    )
    conjuntos = dict(zip(("train", "val", "test"), particion_temporal(datos, "fecha_franja")))
    runs = []
    for nombre in ("dummy", "persistencia", "logistic-regression", "hgb"):
        variables = VARIABLES_HGB if nombre == "hgb" else VARIABLES
        rejilla = None
        parametros = {}
        if nombre in ("logistic-regression", "hgb"):
            modelo, parametros, rejilla = seleccionar_riesgo(
                conjuntos["train"], conjuntos["val"], nombre
            )
            parametros.update({"random_state": SEMILLA})
            if nombre == "hgb":
                parametros.update({"max_iter": ITERACIONES_HGB, "early_stopping": False})
            else:
                parametros.update({"class_weight": "balanced", "max_iter": 1000})
        elif nombre == "dummy":
            modelo = DummyClassifier(strategy="most_frequent").fit(
                conjuntos["train"][variables], conjuntos["train"]["riesgo"]
            )
            parametros["strategy"] = "most_frequent"
        metricas, predicciones = {}, {}
        for conjunto, filas in conjuntos.items():
            if nombre == "persistencia":
                pred = predecir_persistencia(filas)
                clases = CLASES_RIESGO
                puntuaciones = np.column_stack([pred == clase for clase in clases])
            else:
                # Una sola inferencia probabilística también proporciona las clases.
                puntuaciones = modelo.predict_proba(filas[variables])
                clases = modelo.classes_
                pred = clases[np.argmax(puntuaciones, axis=1)]
            predicciones[conjunto] = pred
            metricas.update(
                {
                    f"{conjunto}_{k}": v
                    for k, v in metricas_riesgo(
                        filas["riesgo"], pred, puntuaciones, clases
                    ).items()
                }
            )
        params = parametros_conjuntos(
            nombre,
            ["flujo_lag_56", "capacidad"] if nombre == "persistencia" else variables,
            conjuntos,
        )
        params.update(
            {
                **parametros,
                "alpha": ALFA_RIESGO,
                "etiqueta_riesgo": "proxy: flujo neto vs alfa*capacidad (no es inventario observado)",
            }
        )
        if nombre == "persistencia":
            params.update(
                {
                    "respaldo": "NaN semanal: normal (mayoritaria train real)",
                    "puntuacion_ap": "indicador binario; punto PR de umbral unico",
                }
            )
        params["ap_sin_positivos"] = "0; AP no estimable sin positivos"
        runs.append(
            registrar_resultado(
                "P2",
                f"p2-{nombre}",
                params,
                metricas,
                {
                    f"confusion_{c}": figura_confusion(conjuntos[c]["riesgo"], predicciones[c], c)
                    for c in ("val", "test")
                },
                {"rejilla": rejilla} if rejilla is not None else None,
            )
        )
    return runs
