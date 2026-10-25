"""Entrada `train` del MLflow Project: entrena un modelo de P1, P2 o P3 y lo registra en MLflow.

Uso (siempre a través del MLflow Project):

    mlflow run . -e train --env-manager uv --experiment-name mibici-equipo16 \\
        -P question=p1 -P model=hgb -P learning_rate=0.05 -P max_leaf_nodes=63

Los parámetros vacíos se resuelven con config.py: sin `model` se entrena el baseline de la
pregunta y sin hiperparámetros se usan los de `HIPERPARAMETROS`. Con `search=true` se ejecuta la
búsqueda de `REJILLAS` como nested runs.
"""

from time import perf_counter

from loguru import logger
import typer

from mibici.modeling.comun import ejecutar

app = typer.Typer()


def limpiar(valor: str) -> str:
    """Normaliza vacíos que MLflow puede pasar como comillas literales en Windows."""
    texto = valor.strip()
    return "" if texto in ('""', "''") else texto


def convertir(valor: str):
    """Convierte el texto recibido por -P al tipo de Python: entero, flotante, None o texto."""
    texto = limpiar(valor)
    if texto.lower() in ("none", "null"):
        return None
    for tipo in (int, float):
        try:
            return tipo(texto)
        except ValueError:
            pass
    return texto


def hiperparametros_dados(**valores: str) -> dict:
    """Conserva solo los hiperparámetros que se pasaron (los vacíos se toman de config.py)."""
    return {nombre: convertir(valor) for nombre, valor in valores.items() if limpiar(valor)}


@app.command()
def main(
    question: str = typer.Option("p1", help="Pregunta: p1, p2 o p3"),
    model: str = typer.Option("", help="Modelo; vacío entrena el baseline de la pregunta"),
    stage: str = typer.Option("", help="baseline, search, candidate o analysis; vacío lo deduce"),
    search: str = typer.Option("false", help="true para una búsqueda con nested runs"),
    features: str = typer.Option("", help="Variante de variables: v1 o v2"),
    learning_rate: str = typer.Option(""),
    max_leaf_nodes: str = typer.Option(""),
    max_iter: str = typer.Option(""),
    n_estimators: str = typer.Option(""),
    max_depth: str = typer.Option(""),
    alpha: str = typer.Option("", help="Regularización de la regresión Poisson"),
    c: str = typer.Option("", "--C", help="Regularización inversa (logística)"),
    class_weight: str = typer.Option(""),
    k: str = typer.Option("", help="Número de grupos (P3)"),
    n_components: str = typer.Option(""),
    covariance_type: str = typer.Option("", help="Covarianza de GaussianMixture (P3)"),
):
    """Entrena y registra un modelo de una pregunta como corrida del experimento del equipo."""
    search = limpiar(search).lower()
    if search not in ("true", "false"):
        raise typer.BadParameter("Usa true o false", param_hint="--search")
    hiperparametros = hiperparametros_dados(
        features=features,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        max_iter=max_iter,
        n_estimators=n_estimators,
        max_depth=max_depth,
        alpha=alpha,
        C=c,
        class_weight=class_weight,
        k=k,
        n_components=n_components,
        covariance_type=covariance_type,
    )
    inicio = perf_counter()
    try:
        run_id = ejecutar(
            question,
            limpiar(model) or None,
            limpiar(stage) or None,
            hiperparametros,
            search=search == "true",
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    logger.success(f"Corrida {run_id} terminada en {perf_counter() - inicio:.1f} s")


if __name__ == "__main__":
    app()
