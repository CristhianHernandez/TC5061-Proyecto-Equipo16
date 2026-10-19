"""CLI DH4: carga datos procesados y ejecuta experimentos P1, P2 y P3."""

from time import perf_counter

from loguru import logger
import pandas as pd
import typer

from mibici.config import (
    ALFA_RIESGO,
    FECHA_FIN,
    FECHA_INICIO,
    INICIO_PRUEBA,
    INICIO_VALIDACION,
    VIAJES_DIR,
)
from mibici.dataset import cargar_estaciones, cargar_viajes
from mibici.features import (
    flujo_por_franja,
    limpiar_viajes,
    perfiles_estacion,
    preparar_datos_p1,
    preparar_datos_p2,
)
from mibici.modeling.p1_demanda import entrenar_p1
from mibici.modeling.p2_riesgo import entrenar_p2
from mibici.modeling.p3_clusters import entrenar_p3

app = typer.Typer()


@app.command()
def main(pregunta: str = "todas"):
    """Entrena P1/P2/P3 o todas con datos locales, sin descargas ni reajuste train+val."""
    pregunta = pregunta.upper()
    if pregunta not in ("P1", "P2", "P3", "TODAS"):
        raise typer.BadParameter("Usa P1, P2, P3 o todas", param_hint="--pregunta")
    inicio = perf_counter()
    estaciones = cargar_estaciones()
    viajes = limpiar_viajes(cargar_viajes(), estaciones)
    if pregunta in ("P1", "TODAS"):
        # Pipeline de HH2: demanda diaria con ceros, calendario, rezagos y primera actividad
        entrenar_p1(preparar_datos_p1(VIAJES_DIR, estaciones, FECHA_INICIO, FECHA_FIN), estaciones)
    if pregunta in ("P2", "TODAS"):
        # Las estaciones sin capacidad válida no permiten definir el proxy P2.
        catalogo = estaciones.loc[estaciones["capacidad"] > 0]
        flujos = flujo_por_franja(viajes)
        flujos = flujos.loc[flujos["estacion_id"].isin(catalogo["estacion_id"])]
        entrenar_p2(preparar_datos_p2(flujos, catalogo, alpha=ALFA_RIESGO), catalogo)
    if pregunta in ("P3", "TODAS"):
        entrenar_p3(
            perfiles_estacion(viajes),
            perfiles_estacion(viajes, desde=INICIO_VALIDACION, hasta=INICIO_PRUEBA),
            perfiles_estacion(
                viajes,
                desde=INICIO_PRUEBA,
                hasta=pd.Timestamp(FECHA_FIN) + pd.Timedelta(days=1),
            ),
            estaciones,
        )
    logger.success("Experimentos completados en {:.2f} segundos", perf_counter() - inicio)


if __name__ == "__main__":
    app()
