"""Agrupación de estaciones P3 con selección exclusivamente en perfiles de entrenamiento."""

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
from sklearn.preprocessing import StandardScaler

from mibici.config import N_SEMILLAS_P3, RANGO_K_P3, SEMILLA
from mibici.modeling.comun import parametros_conjuntos, registrar_resultado
from mibici.plots import figuras_clusters


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


def seleccionar_clusters(matriz, tipo):
    """Elige k de P3 únicamente por silueta train; nunca recibe perfiles de val/test."""
    filas, mejor = [], None
    for k in RANGO_K_P3:
        if not 1 < k < len(matriz):
            continue
        modelo = (
            KMeans(n_clusters=k, n_init=10, random_state=SEMILLA)
            if tipo == "kmeans"
            else AgglomerativeClustering(n_clusters=k, linkage="ward")
        )
        etiquetas = modelo.fit_predict(matriz)
        metricas = metricas_clusters(matriz, etiquetas)
        if "silueta" not in metricas:
            continue
        filas.append({"k": k, "train_silueta": metricas["silueta"]})
        if mejor is None or metricas["silueta"] > mejor[0]:
            mejor = (metricas["silueta"], modelo, etiquetas)
    if mejor is None:
        raise ValueError("P3 requiere al menos tres estaciones con perfiles distintos")
    return mejor[1], mejor[2], pd.DataFrame(filas)


def entrenar_p3(entrenamiento, validacion, prueba, estaciones):
    """Ejecuta municipio, KMeans y Ward de P3; evalúa periodos después de seleccionar k train."""
    # El universo queda fijado por train y el catálogo; periodos posteriores solo se intersectan.
    perfiles = entrenamiento.merge(
        estaciones[["estacion_id"]], on="estacion_id", validate="one_to_one"
    )
    catalogo = perfiles[["estacion_id"]].merge(estaciones, on="estacion_id", validate="one_to_one")
    transformador, matriz = ajustar_escala(perfiles)
    conjuntos = {"train": perfiles, "val": validacion, "test": prueba}
    runs = []
    for nombre in ("municipio", "kmeans", "agglomerative-ward"):
        rejilla = None
        if nombre == "municipio":
            etiquetas = catalogo["municipio"].to_numpy()
            modelo = None
        else:
            modelo, etiquetas, rejilla = seleccionar_clusters(matriz, nombre)
        params = parametros_conjuntos(nombre, matriz_perfiles(perfiles).columns, conjuntos)
        params.update(
            {
                "k": len(np.unique(etiquetas)),
                "metricas_indefinidas": "omitidas si grupos no esta entre 2 y n-1",
                "periodos": "interseccion con estaciones train; escala fija train",
            }
        )
        metricas = {f"train_{k}": v for k, v in metricas_clusters(matriz, etiquetas).items()}
        if nombre == "kmeans":
            params.update({"n_init": 10, "random_state": SEMILLA, "n_semillas": N_SEMILLAS_P3})
            replicas = [
                KMeans(
                    n_clusters=modelo.n_clusters, n_init=10, random_state=SEMILLA + i
                ).fit_predict(matriz)
                for i in range(N_SEMILLAS_P3)
            ]
            metricas["train_ari_semillas"] = float(
                np.mean([adjusted_rand_score(a, b) for a, b in combinations(replicas, 2)])
            )
        else:
            params["ari_semillas"] = "no aplica (determinista)"
        if nombre == "agglomerative-ward":
            params["linkage"] = "ward"
        if nombre == "municipio":
            params["ari_periodo"] = "omitido: etiquetas municipales fijas darian ARI=1"
        grupos = np.unique(etiquetas)
        centroides = np.vstack([matriz[etiquetas == g].mean(axis=0) for g in grupos])
        for periodo, futuros in (("val", validacion), ("test", prueba)):
            comunes = (
                perfiles[["estacion_id"]]
                .assign(etiqueta_train=etiquetas)
                .merge(futuros, on="estacion_id", validate="one_to_one")
            )
            params[f"n_{periodo}"] = len(comunes)
            if comunes.empty:
                continue
            nueva = transformador.transform(matriz_perfiles(comunes))
            separacion = metricas_clusters(nueva, comunes["etiqueta_train"].to_numpy())
            if "silueta" in separacion:
                metricas[f"{periodo}_silueta"] = separacion["silueta"]
            if nombre != "municipio":
                asignadas = (
                    modelo.predict(nueva)
                    if nombre == "kmeans"
                    else grupos[pairwise_distances_argmin(nueva, centroides)]
                )
                metricas[f"{periodo}_ari_periodo"] = float(
                    adjusted_rand_score(comunes["etiqueta_train"], asignadas)
                )
        runs.append(
            registrar_resultado(
                "P3",
                f"p3-{nombre}",
                params,
                metricas,
                figuras_clusters(perfiles, catalogo, etiquetas),
                {"busqueda_k": rejilla} if rejilla is not None else None,
            )
        )
    return runs
