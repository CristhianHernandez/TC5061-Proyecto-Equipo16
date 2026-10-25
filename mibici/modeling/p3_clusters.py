"""P3 · Agrupación de estaciones por patrón de uso, ajustada solo con perfiles de entrenamiento.

Cada modelo es una receta de `MODELOS`; `comun.ejecutar` la entrena, la evalúa y la registra.
Las métricas (silueta principal, Davies-Bouldin, Calinski-Harabasz y ARI) son las de la entrega
anterior y se calculan siempre en el espacio de referencia v1 (escala ajustada en train), así que
todas las candidatas son comparables entre sí y con el baseline aunque agrupen con otra variante
de variables. Los modelos servibles se registran como Pipeline (variables, imputación, escala y
algoritmo) para que reciban perfiles crudos y devuelvan el grupo.

Los volúmenes de cada perfil se expresan en viajes por día del periodo: train cubre 18 meses y
validación y prueba 3, así que los totales brutos no serían comparables entre periodos.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from mibici.config import (
    FECHA_FIN,
    FECHA_INICIO,
    INICIO_PRUEBA,
    INICIO_VALIDACION,
    N_SEMILLAS_P3,
    SEMILLA,
)
from mibici.dataset import cargar_estaciones, cargar_viajes
from mibici.features import limpiar_viajes, perfiles_estacion
from mibici.modeling.comun import Modelo, Resultado, parametros_conjuntos
from mibici.plots import figuras_clusters

PREGUNTA = "p3"
PERIODOS = ("train", "val", "test")
VOLUMENES = ("total_salidas", "total_llegadas", "volumen_total", "balance_neto")
COVARIANZAS = ("full", "tied", "diag", "spherical")


def matriz_perfiles(perfiles):
    """Variables P3 v1 sin municipio: perfiles, log-volúmenes, balance relativo y duración."""
    columnas = [c for c in perfiles if c.startswith(("salidas_", "llegadas_"))]
    matriz = perfiles[columnas].copy()
    for columna in ("total_salidas", "total_llegadas", "volumen_total"):
        matriz[f"log_{columna}"] = np.log1p(perfiles[columna])
    matriz["balance_relativo"] = (
        perfiles["balance_neto"].div(perfiles["volumen_total"].replace(0, np.nan)).fillna(0)
    )
    matriz["duracion_media_min"] = perfiles["duracion_media_min"]
    return matriz


def matriz_patron(perfiles):
    """Variables P3 v2: solo la forma del uso (perfiles, balance relativo y duración), sin volumen.

    Sin los log-volúmenes, los grupos no separan estaciones con mucha o poca demanda sino
    estaciones con distinto patrón horario, que es la tipología que pide planeación urbana.
    """
    return matriz_perfiles(perfiles).drop(
        columns=["log_total_salidas", "log_total_llegadas", "log_volumen_total"]
    )


VARIANTES = {"v1": matriz_perfiles, "v2": matriz_patron}


def a_tasas_diarias(perfiles, desde, hasta):
    """Convierte los volúmenes de un perfil del periodo [desde, hasta) a viajes por día."""
    dias = (pd.Timestamp(hasta) - pd.Timestamp(desde)).days
    return perfiles.assign(**{c: perfiles[c] / dias for c in VOLUMENES})


def ajustar_escala(perfiles, variante="v1"):
    """Ajusta imputación mediana y StandardScaler de P3 solo en perfiles de train."""
    transformador = make_pipeline(
        FunctionTransformer(VARIANTES[variante]),
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
    )
    return transformador, transformador.fit_transform(perfiles)


def metricas_clusters(matriz, etiquetas):
    """Evalúa separación P3; omite índices matemáticamente indefinidos (un grupo o n grupos)."""
    grupos = len(np.unique(etiquetas))
    if not 1 < grupos < len(matriz):
        return {}
    return {
        "silueta": float(silhouette_score(matriz, etiquetas)),
        "davies_bouldin": float(davies_bouldin_score(matriz, etiquetas)),
        "calinski_harabasz": float(calinski_harabasz_score(matriz, etiquetas)),
    }


def entradas(perfiles):
    """Columnas que recibe el modelo servible: el perfil de la estación sin su identificador."""
    return perfiles.drop(columns=["estacion_id"])


def construir_conjuntos(entrenamiento, validacion, prueba, estaciones):
    """Fija el universo de estaciones con train y el catálogo; periodos posteriores se intersectan."""
    perfiles = entrenamiento.merge(
        estaciones[["estacion_id"]], on="estacion_id", validate="one_to_one"
    )
    catalogo = perfiles[["estacion_id"]].merge(estaciones, on="estacion_id", validate="one_to_one")
    return {"train": perfiles, "val": validacion, "test": prueba, "catalogo": catalogo}


def preparar_conjuntos():
    """Carga los viajes procesados y construye los perfiles de train, val y test de P3."""
    estaciones = cargar_estaciones()
    viajes = limpiar_viajes(cargar_viajes(), estaciones)
    fin = pd.Timestamp(FECHA_FIN) + pd.Timedelta(days=1)
    cortes = (
        (FECHA_INICIO, INICIO_VALIDACION),
        (INICIO_VALIDACION, INICIO_PRUEBA),
        (INICIO_PRUEBA, fin),
    )
    perfiles = [
        a_tasas_diarias(perfiles_estacion(viajes, desde=desde, hasta=hasta), desde, hasta)
        for desde, hasta in cortes
    ]
    return construir_conjuntos(*perfiles, estaciones)


def datos_prediccion(conjuntos):
    """Perfiles de prueba de P3 (sin objetivo: es agrupación) para verificar el modelo."""
    return conjuntos["test"], None


def construir_pipeline(algoritmo, variante="v1"):
    """Pipeline servible de P3: variables de la variante, imputación, escala y algoritmo."""
    return make_pipeline(
        FunctionTransformer(VARIANTES[variante]),
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        algoritmo,
    )


def construir_kmeans(k, semilla=SEMILLA, variante="v1"):
    """Pipeline servible de P3 con KMeans."""
    return construir_pipeline(KMeans(n_clusters=k, n_init=10, random_state=semilla), variante)


def ari_semillas(fabricar, matriz):
    """Estabilidad P3: ARI medio entre los grupos de N_SEMILLAS_P3 inicializaciones distintas."""
    replicas = [fabricar(SEMILLA + i).fit_predict(matriz) for i in range(N_SEMILLAS_P3)]
    return float(np.mean([adjusted_rand_score(a, b) for a, b in combinations(replicas, 2)]))


def evaluar_periodos(conjuntos, etiquetas, transformar, asignar):
    """Mide en val y test la separación de los grupos de train y su consistencia (ARI periodo).

    Solo se evalúan las estaciones de train presentes en cada periodo; la escala es la de train.
    La silueta usa las etiquetas de train sobre los perfiles del periodo (¿siguen separados los
    mismos grupos?) y el ARI compara esas etiquetas con las que el modelo asigna en el periodo.
    """
    perfiles = conjuntos["train"]
    metricas, tamanos = {}, {}
    for periodo in ("val", "test"):
        comunes = (
            perfiles[["estacion_id"]]
            .assign(etiqueta_train=etiquetas)
            .merge(conjuntos[periodo], on="estacion_id", validate="one_to_one")
        )
        tamanos[f"n_comunes_{periodo}"] = len(comunes)
        if comunes.empty:
            continue
        nueva = transformar(comunes[perfiles.columns])
        separacion = metricas_clusters(nueva, comunes["etiqueta_train"].to_numpy())
        if "silueta" in separacion:
            metricas[f"{periodo}_silueta"] = separacion["silueta"]
        if asignar is not None:
            metricas[f"{periodo}_ari_periodo"] = float(
                adjusted_rand_score(
                    comunes["etiqueta_train"], asignar(comunes[perfiles.columns], nueva)
                )
            )
    return metricas, tamanos


def _params(nombre, conjuntos, k, extra, variante="v1"):
    """Parámetros comunes de P3: variables, tamaños de periodo, k y nota de métricas."""
    periodos = {p: conjuntos[p] for p in PERIODOS}
    variables = VARIANTES[variante](conjuntos["train"]).columns
    params = parametros_conjuntos(nombre, variables, periodos)
    params.update(
        {
            "n_grupos": k,
            "metricas_indefinidas": "omitidas si grupos no esta entre 2 y n-1",
            "periodos": "interseccion con estaciones train; escala fija train",
            "espacio_metricas": "v1 (escala de train)",
            "volumenes": "viajes por dia del periodo",
            **extra,
        }
    )
    return params


def entrenar_municipio(conjuntos, hiperparametros):
    """Baseline de referencia de P3: agrupar las estaciones por municipio, sin aprendizaje."""
    perfiles, catalogo = conjuntos["train"], conjuntos["catalogo"]
    referencia, matriz = ajustar_escala(perfiles)
    etiquetas = catalogo["municipio"].to_numpy()
    metricas = {f"train_{k}": v for k, v in metricas_clusters(matriz, etiquetas).items()}
    periodo, tamanos = evaluar_periodos(conjuntos, etiquetas, referencia.transform, None)
    params = _params(
        "municipio",
        conjuntos,
        len(np.unique(etiquetas)),
        {**tamanos, "ari_periodo": "omitido: etiquetas municipales fijas darian ARI=1"},
    )
    return Resultado(
        {**metricas, **periodo},
        params,
        figuras_clusters(perfiles, catalogo, etiquetas, matriz),
        entrenamiento=entradas(perfiles),
    )


def _entrenar_servible(conjuntos, nombre, fabricar, variante, k, extra):
    """Ajusta un Pipeline servible de P3 (KMeans o GMM) y lo evalúa en el espacio de referencia.

    `fabricar(semilla)` construye el algoritmo; el ARI entre semillas se mide en el espacio de la
    variante con la que el modelo agrupa.
    """
    perfiles, catalogo = conjuntos["train"], conjuntos["catalogo"]
    modelo = construir_pipeline(fabricar(SEMILLA), variante).fit(entradas(perfiles))
    etiquetas = modelo.predict(entradas(perfiles))
    referencia, matriz = ajustar_escala(perfiles)
    metricas = {f"train_{n}": v for n, v in metricas_clusters(matriz, etiquetas).items()}
    metricas["train_ari_semillas"] = ari_semillas(
        fabricar, modelo[:-1].transform(entradas(perfiles))
    )
    periodo, tamanos = evaluar_periodos(
        conjuntos,
        etiquetas,
        referencia.transform,
        lambda comunes, _: modelo.predict(entradas(comunes)),
    )
    params = _params(
        nombre,
        conjuntos,
        k,
        {**tamanos, **extra, "features": variante, "n_semillas": N_SEMILLAS_P3},
        variante,
    )
    return Resultado(
        {**metricas, **periodo},
        params,
        figuras_clusters(perfiles, catalogo, etiquetas, matriz),
        modelo=modelo,
        ejemplo=entradas(perfiles).head(5),
        entrenamiento=entradas(perfiles),
        x_prueba=entradas(conjuntos["test"][perfiles.columns]),
    )


def _variante(hiperparametros):
    """Valida la variante de variables pedida (v1 por defecto)."""
    variante = hiperparametros.get("features", "v1")
    if variante not in VARIANTES:
        raise ValueError(f"features debe ser una de {list(VARIANTES)}")
    return variante


def entrenar_kmeans(conjuntos, hiperparametros):
    """Candidato servible de P3: KMeans sobre los perfiles escalados con la media de train."""
    k = int(hiperparametros["k"])
    return _entrenar_servible(
        conjuntos,
        "kmeans",
        lambda semilla: KMeans(n_clusters=k, n_init=10, random_state=semilla),
        _variante(hiperparametros),
        k,
        {"k": k, "n_init": 10, "random_state": SEMILLA},
    )


def entrenar_gmm(conjuntos, hiperparametros):
    """Candidato servible de P3: mezcla gaussiana; asigna cada estación al componente más probable."""
    k = int(hiperparametros["n_components"])
    covarianza = hiperparametros["covariance_type"]
    if covarianza not in COVARIANZAS:
        raise ValueError(f"covariance_type debe ser una de {COVARIANZAS}")
    return _entrenar_servible(
        conjuntos,
        "gmm",
        lambda semilla: GaussianMixture(
            n_components=k, covariance_type=covarianza, n_init=5, random_state=semilla
        ),
        "v1",
        k,
        {"n_components": k, "covariance_type": covarianza, "n_init": 5, "random_state": SEMILLA},
    )


def entrenar_ward(conjuntos, hiperparametros):
    """Candidato de comparación de P3: Ward jerárquico.

    AgglomerativeClustering no tiene `predict` para estaciones nuevas, así que no se registra un
    modelo servible; los periodos se asignan al centroide de train más cercano.
    """
    perfiles, catalogo = conjuntos["train"], conjuntos["catalogo"]
    k = int(hiperparametros["k"])
    referencia, matriz = ajustar_escala(perfiles)
    etiquetas = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(matriz)
    metricas = {f"train_{n}": v for n, v in metricas_clusters(matriz, etiquetas).items()}
    grupos = np.unique(etiquetas)
    centroides = np.vstack([matriz[etiquetas == g].mean(axis=0) for g in grupos])
    periodo, tamanos = evaluar_periodos(
        conjuntos,
        etiquetas,
        referencia.transform,
        lambda _, nueva: grupos[pairwise_distances_argmin(nueva, centroides)],
    )
    params = _params(
        "ward",
        conjuntos,
        k,
        {
            **tamanos,
            **hiperparametros,
            "linkage": "ward",
            "ari_semillas": "no aplica (determinista)",
            "servible": "no: AgglomerativeClustering no tiene predict",
        },
    )
    return Resultado(
        {**metricas, **periodo},
        params,
        figuras_clusters(perfiles, catalogo, etiquetas, matriz),
        entrenamiento=entradas(perfiles),
    )


MODELOS = {
    "municipio": Modelo("naive", entrenar_municipio, baseline=True),
    "kmeans": Modelo("clustering", entrenar_kmeans, hiperparametros=("k", "features")),
    "gmm": Modelo("clustering", entrenar_gmm, hiperparametros=("n_components", "covariance_type")),
    "ward": Modelo("clustering", entrenar_ward, hiperparametros=("k",)),
}
