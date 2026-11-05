"""Prepara el modelo que sirve la API: exporta el champion del registry y prueba una predicción.

Con el servidor de MLflow en marcha (`make mlflow`), desde la raíz del repositorio:

    python -m mibici.serving.exportar                 # o: make export-model
    python -m mibici.serving.exportar --no-exportar   # solo prueba la copia ya exportada

Descarga `MODEL_URI` (por defecto `models:/mibici-equipo16-p1-demanda@champion`) a `MODEL_PATH`
(`model_bundle/`, fuera de Git), carga esa copia sin conexión al registry y predice la petición de
ejemplo de `docs/serving/ejemplo_entrada_p1.json`. Imprime el resultado en JSON.
"""

import json
from pathlib import Path

import typer

from mibici.config import EJEMPLO_SERVICIO, MODEL_PATH, MODEL_URI
from mibici.serving.model_loader import cargar_modelo, exportar

app = typer.Typer()


@app.command()
def main(
    destino: str = typer.Option(str(MODEL_PATH), help="Carpeta de la copia del modelo"),
    model_uri: str = typer.Option(MODEL_URI, help="models:/<nombre>@<alias> o /<versión>"),
    ejemplo: str = typer.Option(str(EJEMPLO_SERVICIO), help="JSON con una lista de registros"),
    exportar_modelo: bool = typer.Option(
        True, "--exportar/--no-exportar", help="Descargar del registry antes de probar"
    ),
):
    """Exporta el champion a una carpeta local y verifica que predice fuera de la API."""
    destino, ejemplo = Path(destino), Path(ejemplo)
    if exportar_modelo:
        metadatos = exportar(destino, model_uri)
        typer.echo(
            f"Exportado {metadatos['uri'] or metadatos['uri_version']} -> versión "
            f"{metadatos['version']} en {destino}",
            err=True,
        )
    servido = cargar_modelo(destino)  # solo la copia local, sin registry
    if servido.metadatos["origen"] != "copia_local":
        raise typer.BadParameter(f"No hay copia del modelo en {destino}")
    registros = json.loads(ejemplo.read_text(encoding="utf-8"))["registros"]
    salida = {
        "modelo": {k: servido.metadatos.get(k) for k in ("nombre", "version", "alias", "run_id")},
        "predicciones": servido.predecir(registros),
    }
    typer.echo(json.dumps(salida, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
