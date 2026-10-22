"""P2 · Clasificación del proxy de riesgo operativo por estación y franja, sin inventario observado.

Cada modelo es una receta de `MODELOS`; `comun.ejecutar` la entrena, la evalúa y la registra.
La etiqueta, la partición temporal y las métricas (F1 macro principal, recall por clase, AP y
balanced accuracy) son las mismas de la entrega anterior.
"""

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

from mibici.config import ALFA_RIESGO, CLASES_RIESGO, SEMILLA
from mibici.dataset import cargar_estaciones, cargar_viajes
from mibici.features import (
    flujo_por_franja,
    limpiar_viajes,
    particion_temporal,
    preparar_datos_p2,
)
from mibici.modeling.comun import (
    Modelo,
    Resultado,
    estabilidad_mensual,
    metricas_con_prefijo,
    parametros_conjuntos,
)
from mibici.plots import figura_confusion

PREGUNTA = "p2"
OBJETIVO = "riesgo"
NUMERICAS = ["flujo_lag_1", "flujo_lag_56", "capacidad"]
CATEGORICAS = ["hora", "dia_semana"]
VARIABLES = NUMERICAS + CATEGORICAS
VARIABLES_HGB = VARIABLES + ["latitud", "longitud"]
DESCRIPCION_ETIQUETA = "proxy: flujo neto vs alfa*capacidad (no es inventario observado)"


def construir_conjuntos(datos, estaciones):
    """Une la ubicación de cada estación y aplica la partición temporal por fecha de franja."""
    datos = datos.merge(
        estaciones[["estacion_id", "latitud", "longitud"]],
        on="estacion_id",
        how="left",
        validate="many_to_one",
    )
    return dict(zip(("train", "val", "test"), particion_temporal(datos, "fecha_franja")))


def preparar_conjuntos():
    """Carga los datos procesados y construye train, val y test de P2 (sin descargar nada)."""
    estaciones = cargar_estaciones()
    # Las estaciones sin capacidad válida no permiten definir el proxy P2.
    catalogo = estaciones.loc[estaciones["capacidad"] > 0]
    flujos = flujo_por_franja(limpiar_viajes(cargar_viajes(), estaciones))
    flujos = flujos.loc[flujos["estacion_id"].isin(catalogo["estacion_id"])]
    return construir_conjuntos(preparar_datos_p2(flujos, catalogo, alpha=ALFA_RIESGO), catalogo)


def datos_prediccion(conjuntos):
    """Conjunto de prueba de P2 y su columna objetivo, para verificar el modelo en producción."""
    return conjuntos["test"], OBJETIVO


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


def f1_macro(real, prediccion):
    """F1 macro sobre las tres clases de P2 (métrica principal)."""
    return f1_score(real, prediccion, labels=CLASES_RIESGO, average="macro", zero_division=0)


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


def evaluar(conjuntos, puntuar):
    """Evalúa P2 en los tres conjuntos; `puntuar` devuelve (clases predichas, puntajes, clases).

    La estabilidad es la desviación estándar del F1 macro mes a mes en prueba.
    """
    metricas, predicciones = {}, {}
    for conjunto, filas in conjuntos.items():
        prediccion, puntuaciones, clases = puntuar(filas)
        predicciones[conjunto] = prediccion
        metricas.update(
            metricas_con_prefijo(
                conjunto, metricas_riesgo(filas[OBJETIVO], prediccion, puntuaciones, clases)
            )
        )
    prueba = conjuntos["test"]
    metricas["test_f1_macro_std_mensual"] = estabilidad_mensual(
        prueba["fecha_franja"], prueba[OBJETIVO].to_numpy(), predicciones["test"], f1_macro
    )
    return metricas, predicciones


def _puntuar_estimador(modelo, variables):
    """Una sola inferencia probabilística por conjunto proporciona puntajes y clases."""

    def puntuar(filas):
        puntuaciones = modelo.predict_proba(filas[variables])
        return modelo.classes_[np.argmax(puntuaciones, axis=1)], puntuaciones, modelo.classes_

    return puntuar


def _figuras(conjuntos, predicciones):
    """Matrices de confusión de validación y prueba."""
    return {
        f"confusion_{c}": figura_confusion(conjuntos[c][OBJETIVO], predicciones[c], c)
        for c in ("val", "test")
    }


def _params(nombre, variables, conjuntos, extra):
    """Parámetros comunes de P2: variables, tamaños, alfa y descripción de la etiqueta."""
    params = parametros_conjuntos(nombre, variables, conjuntos)
    params.update(
        {
            **extra,
            "alpha": ALFA_RIESGO,
            "etiqueta_riesgo": DESCRIPCION_ETIQUETA,
            "ap_sin_positivos": "0; AP no estimable sin positivos",
        }
    )
    return params


def _resultado(conjuntos, nombre, modelo, variables, extra):
    """Evalúa un estimador ya ajustado en train y arma su Resultado registrable."""
    metricas, predicciones = evaluar(conjuntos, _puntuar_estimador(modelo, variables))
    entrenamiento = conjuntos["train"]
    return Resultado(
        metricas,
        _params(nombre, variables, conjuntos, extra),
        _figuras(conjuntos, predicciones),
        modelo=modelo,
        ejemplo=entrenamiento[variables].head(5),
        entrenamiento=entrenamiento[variables + [OBJETIVO]],
        objetivo=OBJETIVO,
        x_prueba=conjuntos["test"][variables],
    )


def entrenar_persistencia(conjuntos, hiperparametros):
    """Baseline de referencia de P2: la clase que tendría la misma franja de la semana anterior."""

    def puntuar(filas):
        prediccion = predecir_persistencia(filas)
        return (
            prediccion,
            np.column_stack([prediccion == clase for clase in CLASES_RIESGO]),
            CLASES_RIESGO,
        )

    metricas, predicciones = evaluar(conjuntos, puntuar)
    variables = ["flujo_lag_56", "capacidad"]
    params = _params(
        "persistencia",
        variables,
        conjuntos,
        {
            "respaldo": "NaN semanal: normal (mayoritaria train real)",
            "puntuacion_ap": "indicador binario; punto PR de umbral unico",
        },
    )
    return Resultado(
        metricas,
        params,
        _figuras(conjuntos, predicciones),
        entrenamiento=conjuntos["train"][variables + [OBJETIVO]],
        objetivo=OBJETIVO,
    )


def entrenar_dummy(conjuntos, hiperparametros):
    """Baseline trivial de P2: siempre la clase más frecuente de train."""
    entrenamiento = conjuntos["train"]
    modelo = DummyClassifier(strategy="most_frequent").fit(
        entrenamiento[VARIABLES], entrenamiento[OBJETIVO]
    )
    return _resultado(conjuntos, "dummy", modelo, VARIABLES, {"strategy": "most_frequent"})


def entrenar_logistica(conjuntos, hiperparametros):
    """Candidato lineal de P2: regresión logística balanceada con preprocesamiento de train."""
    entrenamiento = conjuntos["train"]
    modelo = construir_logistica(hiperparametros).fit(
        entrenamiento[VARIABLES], entrenamiento[OBJETIVO]
    )
    extra = {**hiperparametros, "class_weight": "balanced", "max_iter": 1000}
    return _resultado(
        conjuntos, "logistica", modelo, VARIABLES, {**extra, "random_state": SEMILLA}
    )


def entrenar_hgb(conjuntos, hiperparametros):
    """Candidato de P2: gradient boosting de clasificación (maneja NaN de los rezagos)."""
    entrenamiento = conjuntos["train"]
    modelo = HistGradientBoostingClassifier(
        random_state=SEMILLA, early_stopping=False, **hiperparametros
    ).fit(entrenamiento[VARIABLES_HGB], entrenamiento[OBJETIVO])
    extra = {**hiperparametros, "early_stopping": False, "random_state": SEMILLA}
    return _resultado(conjuntos, "hgb", modelo, VARIABLES_HGB, extra)


MODELOS = {
    "persistencia": Modelo("naive", entrenar_persistencia, baseline=True),
    "dummy": Modelo("naive", entrenar_dummy, baseline=True),
    "logistica": Modelo("linear", entrenar_logistica, hiperparametros=("C",)),
    "hgb": Modelo(
        "boosting",
        entrenar_hgb,
        hiperparametros=("learning_rate", "class_weight", "max_iter", "max_leaf_nodes"),
    ),
}
