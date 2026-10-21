"""P3 · Agrupación de estaciones por patrón de uso, ajustada solo con perfiles de entrenamiento.

Cada modelo es una receta de `MODELOS`; `comun.ejecutar` la entrena, la evalúa y la registra.
Las métricas (silueta principal, Davies-Bouldin, Calinski-Harabasz y ARI) son las de la entrega
anterior. Los modelos servibles se registran como Pipeline (variables, imputación, escala y
algoritmo) para que reciban perfiles crudos y devuelvan el grupo.
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from mibici.config import FECHA_FIN, INICIO_PRUEBA, INICIO_VALIDACION, N_SEMILLAS_P3, SEMILLA
from mibici.dataset import cargar_estaciones, cargar_viajes
from mibici.features import limpiar_viajes, perfiles_estacion
from mibici.modeling.comun import Modelo, Resultado, parametros_conjuntos
from mibici.plots import figuras_clusters

PREGUNTA = "p3"
PERIODOS = ("train", "val", "test")


def matriz_perfiles(perfiles):
    """Construye variables P3 sin municipio: perfiles, log-volúmenes, balance relativo y duración."""
    columnas = [c for c in perfiles if c.startswith(("salidas_", "llegadas_"))]
    matriz = perfiles[columnas].copy()
    for columna in ("total_salidas", "total_llegadas", "volumen_total"):
        matriz[f"log_{columna}"] = np.log1p(perfiles[columna])
    matriz["balance_relativo"] = (
        perfiles["balance_neto"].div(perfiles["volumen_total"].replace(0, np.nan)).fillna(0)
    )
    matriz["duracion_media_min"] = perfiles["duracion_media_min"]
    return matriz


def ajustar_escala(perfiles):
    """Ajusta imputación mediana y StandardScaler de P3 solo en perfiles de train."""
    transformador = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
    )
    return transformador, transformador.fit_transform(matriz_perfiles(perfiles))


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
    return construir_conjuntos(
        perfiles_estacion(viajes),
        perfiles_estacion(viajes, desde=INICIO_VALIDACION, hasta=INICIO_PRUEBA),
        perfiles_estacion(
            viajes, desde=INICIO_PRUEBA, hasta=pd.Timestamp(FECHA_FIN) + pd.Timedelta(days=1)
        ),
        estaciones,
    )


def datos_prediccion(conjuntos):
    """Perfiles de prueba de P3 (sin objetivo: es agrupación) para verificar el modelo."""
    return conjuntos["test"], None


def construir_kmeans(k, semilla=SEMILLA):
    """Pipeline servible de P3: variables de perfil, imputación, escala y KMeans."""
    return make_pipeline(
        FunctionTransformer(matriz_perfiles),
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        KMeans(n_clusters=k, n_init=10, random_state=semilla),
    )


def evaluar_periodos(conjuntos, etiquetas, transformar, asignar):
    """Mide en val y test la separación de los grupos de train y su consistencia (ARI periodo).

    Solo se evalúan las estaciones de train presentes en cada periodo; la escala es la de train.
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
        nueva = transformar(comunes)
        separacion = metricas_clusters(nueva, comunes["etiqueta_train"].to_numpy())
        if "silueta" in separacion:
            metricas[f"{periodo}_silueta"] = separacion["silueta"]
        if asignar is not None:
            metricas[f"{periodo}_ari_periodo"] = float(
                adjusted_rand_score(comunes["etiqueta_train"], asignar(comunes, nueva))
            )
    return metricas, tamanos


def _params(nombre, conjuntos, k, extra):
    """Parámetros comunes de P3: variables, tamaños de periodo, k y nota de métricas."""
    periodos = {p: conjuntos[p] for p in PERIODOS}
    params = parametros_conjuntos(nombre, matriz_perfiles(conjuntos["train"]).columns, periodos)
    params.update(
        {
            "n_grupos": k,
            "metricas_indefinidas": "omitidas si grupos no esta entre 2 y n-1",
            "periodos": "interseccion con estaciones train; escala fija train",
            **extra,
        }
    )
    return params


def entrenar_municipio(conjuntos, hiperparametros):
    """Baseline de referencia de P3: agrupar las estaciones por municipio, sin aprendizaje."""
    perfiles, catalogo = conjuntos["train"], conjuntos["catalogo"]
    transformador, matriz = ajustar_escala(perfiles)
    etiquetas = catalogo["municipio"].to_numpy()
    metricas = {f"train_{k}": v for k, v in metricas_clusters(matriz, etiquetas).items()}
    periodo, tamanos = evaluar_periodos(
        conjuntos, etiquetas, lambda c: transformador.transform(matriz_perfiles(c)), None
    )
    params = _params(
        "municipio",
        conjuntos,
        len(np.unique(etiquetas)),
        {**tamanos, "ari_periodo": "omitido: etiquetas municipales fijas darian ARI=1"},
    )
    return Resultado(
        {**metricas, **periodo},
        params,
        figuras_clusters(perfiles, catalogo, etiquetas),
        entrenamiento=entradas(perfiles),
    )


def entrenar_kmeans(conjuntos, hiperparametros):
    """Candidato servible de P3: KMeans sobre los perfiles escalados con la media de train."""
    perfiles, catalogo = conjuntos["train"], conjuntos["catalogo"]
    k = int(hiperparametros["k"])
    modelo = construir_kmeans(k).fit(entradas(perfiles))
    preparar = modelo[:-1]
    matriz = preparar.transform(entradas(perfiles))
    etiquetas = modelo[-1].labels_
    metricas = {f"train_{n}": v for n, v in metricas_clusters(matriz, etiquetas).items()}
    replicas = [
        KMeans(n_clusters=k, n_init=10, random_state=SEMILLA + i).fit_predict(matriz)
        for i in range(N_SEMILLAS_P3)
    ]
    metricas["train_ari_semillas"] = float(
        np.mean([adjusted_rand_score(a, b) for a, b in combinations(replicas, 2)])
    )
    periodo, tamanos = evaluar_periodos(
        conjuntos,
        etiquetas,
        lambda c: preparar.transform(entradas(c[perfiles.columns])),
        lambda c, nueva: modelo[-1].predict(nueva),
    )
    params = _params(
        "kmeans",
        conjuntos,
        k,
        {
            **tamanos,
            **hiperparametros,
            "n_init": 10,
            "random_state": SEMILLA,
            "n_semillas": N_SEMILLAS_P3,
        },
    )
    return Resultado(
        {**metricas, **periodo},
        params,
        figuras_clusters(perfiles, catalogo, etiquetas),
        modelo=modelo,
        ejemplo=entradas(perfiles).head(5),
        entrenamiento=entradas(perfiles),
        x_prueba=entradas(conjuntos["test"][perfiles.columns]),
    )


def entrenar_ward(conjuntos, hiperparametros):
    """Candidato de comparación de P3: Ward jerárquico.

    AgglomerativeClustering no tiene `predict` para estaciones nuevas, así que no se registra un
    modelo servible; los periodos se asignan al centroide de train más cercano.
    """
    perfiles, catalogo = conjuntos["train"], conjuntos["catalogo"]
    k = int(hiperparametros["k"])
    transformador, matriz = ajustar_escala(perfiles)
    etiquetas = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(matriz)
    metricas = {f"train_{n}": v for n, v in metricas_clusters(matriz, etiquetas).items()}
    grupos = np.unique(etiquetas)
    centroides = np.vstack([matriz[etiquetas == g].mean(axis=0) for g in grupos])
    periodo, tamanos = evaluar_periodos(
        conjuntos,
        etiquetas,
        lambda c: transformador.transform(matriz_perfiles(c)),
        lambda c, nueva: grupos[pairwise_distances_argmin(nueva, centroides)],
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
        figuras_clusters(perfiles, catalogo, etiquetas),
        entrenamiento=entradas(perfiles),
    )


MODELOS = {
    "municipio": Modelo("naive", entrenar_municipio, baseline=True),
    "kmeans": Modelo("clustering", entrenar_kmeans, hiperparametros=("k",)),
    "ward": Modelo("clustering", entrenar_ward, hiperparametros=("k",)),
}
