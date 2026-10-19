"""Figuras agregadas de demanda P1, riesgo P2 y agrupación P3, sin datos personales."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from mibici.config import CLASES_RIESGO


def figura_demanda(conjuntos, predicciones):
    """Dibuja real contra predicho y totales diarios de validación y prueba para P1."""
    figura, ejes = plt.subplots(1, 2, figsize=(12, 4))
    for nombre in ("val", "test"):
        datos = conjuntos[nombre]
        pred = predicciones[nombre]
        ejes[0].scatter(datos["demanda"], pred, s=2, alpha=0.15, label=nombre, rasterized=True)
        diario = datos[["fecha", "demanda"]].assign(prediccion=pred).groupby("fecha").sum()
        ejes[1].plot(diario.index, diario["demanda"], label=f"Real {nombre}")
        ejes[1].plot(diario.index, diario["prediccion"], label=f"Predicción {nombre}")
    limite = max(ejes[0].get_xlim()[1], ejes[0].get_ylim()[1])
    ejes[0].plot([0, limite], [0, limite], "k--", linewidth=1)
    ejes[0].set(xlabel="Demanda real", ylabel="Demanda predicha", title="P1: demanda diaria")
    ejes[1].set(xlabel="Fecha", ylabel="Viajes", title="Totales diarios de la red")
    for eje in ejes:
        eje.legend()
    figura.autofmt_xdate()
    figura.tight_layout()
    return figura


def figura_confusion(real, prediccion, conjunto):
    """Muestra matriz P2 normalizada por clase real, anotada con conteos y proporciones."""
    conteos = confusion_matrix(real, prediccion, labels=CLASES_RIESGO)
    normalizada = conteos / np.maximum(conteos.sum(axis=1, keepdims=True), 1)
    figura, eje = plt.subplots(figsize=(6, 5))
    imagen = eje.imshow(normalizada, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(CLASES_RIESGO)):
        for j in range(len(CLASES_RIESGO)):
            eje.text(
                j,
                i,
                f"{conteos[i, j]}\n{normalizada[i, j]:.1%}",
                ha="center",
                va="center",
                color="white" if normalizada[i, j] > 0.5 else "black",
            )
    eje.set(
        xticks=range(3),
        yticks=range(3),
        xticklabels=CLASES_RIESGO,
        yticklabels=CLASES_RIESGO,
        xlabel="Predicción",
        ylabel="Clase real",
        title=f"P2: confusión en {conjunto}",
    )
    eje.tick_params(axis="x", rotation=25)
    figura.colorbar(imagen, ax=eje, label="Proporción por clase real")
    figura.tight_layout()
    return figura


def figuras_clusters(perfiles, estaciones, etiquetas):
    """Dibuja mapa y perfiles horarios medios de salidas/llegadas por grupo para P3."""
    mapa, eje = plt.subplots(figsize=(7, 5))
    figura, ejes = plt.subplots(2, 2, figsize=(12, 7))
    for etiqueta in np.unique(etiquetas):
        mascara = etiquetas == etiqueta
        eje.scatter(
            estaciones.loc[mascara, "longitud"],
            estaciones.loc[mascara, "latitud"],
            label=str(etiqueta),
            s=15,
        )
        for i, tipo in enumerate(("salidas", "llegadas")):
            for j, dia in enumerate(("laboral", "fin_semana")):
                columnas = [f"{tipo}_{dia}_{h:02d}" for h in range(24)]
                ejes[i, j].plot(
                    range(24), perfiles.loc[mascara, columnas].mean(), label=str(etiqueta)
                )
                ejes[i, j].set(xlabel="Hora", ylabel="Proporción media", title=f"{tipo} · {dia}")
    eje.set(xlabel="Longitud", ylabel="Latitud", title="P3: grupos de estaciones (train)")
    eje.legend(title="Grupo")
    for subeje in ejes.flat:
        subeje.legend(title="Grupo")
    figura.suptitle("P3: perfiles horarios medios por grupo")
    figura.tight_layout()
    mapa.tight_layout()
    return {"mapa": mapa, "perfiles": figura}
