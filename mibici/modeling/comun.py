"""Ejecución y registro compartidos de los experimentos P1, P2 y P3 (MLflow Project).

Cada `mlflow run . -e train` entrena **un** modelo de una pregunta y produce una corrida, o una
corrida padre de búsqueda con una corrida hija por combinación de hiperparámetros. Cada corrida
guarda sus artefactos en subcarpetas:

- `data/`: manifiesto de la versión de los datos y, con `mlflow.log_input`, el conjunto de
  entrenamiento (solo variables agregadas por estación, sin datos personales);
- `config/`: configuración efectiva de la corrida y archivos del proyecto con que se ejecutó;
- `plots/`: gráficas de desempeño de la tarea;
- `metrics/`: métricas y tablas de la búsqueda;
- `model/`: el modelo con firma y ejemplo de entrada (además del LoggedModel de MLflow 3).

Los módulos de cada pregunta solo definen cómo preparar sus conjuntos y cómo entrenar y evaluar
cada modelo (`MODELOS`); todo lo demás vive aquí para que las tres preguntas registren igual.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter

from loguru import logger
import matplotlib
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from mibici import tracking
from mibici.config import (
    ALFA_RIESGO,
    BASELINES,
    EXPERIMENTO_MLFLOW,
    FECHA_FIN,
    FECHA_INICIO,
    FIGURES_DIR,
    FUENTE_DATOS,
    HIPERPARAMETROS,
    HORAS_FRANJA,
    INICIO_PRUEBA,
    INICIO_VALIDACION,
    MANIFIESTO_DATOS,
    METRICA_PRINCIPAL,
    MINIMIZAR,
    PREGUNTAS_PROYECTO,
    PROJ_ROOT,
    REJILLAS,
    SEMILLA,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULOS = {
    "p1": "mibici.modeling.p1_demanda",
    "p2": "mibici.modeling.p2_riesgo",
    "p3": "mibici.modeling.p3_clusters",
}
# `mlflow run` registra por sí mismo los parámetros del MLproject (con su valor pedido, vacío si
# no se pasó). Los valores efectivos de esos mismos hiperparámetros se registran con el prefijo hp_
# para no chocar con ellos: así la corrida conserva lo pedido y lo realmente usado.
PREFIJO_HIPERPARAMETRO = "hp_"
# Archivos del proyecto que definen cómo se ejecutó la corrida (se guardan en config/)
ARCHIVOS_CONFIGURACION = ("MLproject", "python_env.yaml", "requirements.txt", "mibici/config.py")


@dataclass
class Resultado:
    """Lo que devuelve el entrenamiento de un modelo para que se registre en MLflow."""

    metricas: dict[str, float]
    params: dict = field(default_factory=dict)
    figuras: dict = field(default_factory=dict)
    tablas: dict[str, pd.DataFrame] = field(default_factory=dict)
    modelo: object | None = None  # estimador con predict, listo para servir
    ejemplo: pd.DataFrame | None = None  # entradas de ejemplo para la firma del modelo
    entrenamiento: pd.DataFrame | None = None  # conjunto de entrenamiento para log_input
    objetivo: str | None = None
    x_prueba: pd.DataFrame | None = None  # entradas de prueba para medir la latencia


@dataclass(frozen=True)
class Modelo:
    """Receta de un modelo de una pregunta: familia, si es baseline y cómo se entrena."""

    familia: str
    entrenar: Callable[[dict, dict], Resultado]
    baseline: bool = False
    hiperparametros: tuple[str, ...] = ()


def modulo_pregunta(question: str):
    """Importa el módulo de la pregunta (p1, p2 o p3)."""
    question = question.lower()
    if question not in PREGUNTAS_PROYECTO:
        raise ValueError(f"Pregunta inválida: {question!r}; usa una de {PREGUNTAS_PROYECTO}")
    return importlib.import_module(MODULOS[question])


def metricas_con_prefijo(conjunto: str, valores: dict) -> dict[str, float]:
    """Antepone el conjunto (train, val o test) a cada métrica."""
    return {f"{conjunto}_{k}": float(v) for k, v in valores.items()}


def estabilidad_mensual(fechas, real, prediccion, metrica) -> float:
    """Desviación estándar de una métrica calculada mes a mes: mide la estabilidad temporal."""
    datos = pd.DataFrame(
        {"mes": pd.to_datetime(fechas).dt.to_period("M"), "real": real, "pred": prediccion}
    )
    valores = [metrica(g["real"], g["pred"]) for _, g in datos.groupby("mes") if len(g)]
    return float(np.std(valores)) if len(valores) > 1 else 0.0


def parametros_conjuntos(modelo, variables, conjuntos):
    """Describe variables y tamaños de partición de P1/P2/P3 para auditoría sin datos personales."""
    return {
        "modelo": modelo,
        "variables": json.dumps(list(variables)),
        "ajuste": "solo train; sin reajuste train+val",
        **{f"n_{nombre}": len(datos) for nombre, datos in conjuntos.items()},
    }


def parametros_mlproject() -> set[str]:
    """Nombres de los parámetros de la entrada `train` declarados en el MLproject."""
    import yaml

    proyecto = yaml.safe_load((PROJ_ROOT / "MLproject").read_text(encoding="utf-8"))
    return set(proyecto["entry_points"]["train"]["parameters"])


def parametros_para_registrar(params: dict) -> dict:
    """Antepone hp_ a los hiperparámetros que también son parámetros del MLproject."""
    reservados = parametros_mlproject()
    return {
        (f"{PREFIJO_HIPERPARAMETRO}{k}" if k in reservados else k): v for k, v in params.items()
    }


def resolver_hiperparametros(question: str, model: str, receta: Modelo, dados: dict) -> dict:
    """Combina los hiperparámetros por defecto de config.py con los recibidos por -P."""
    desconocidos = sorted(set(dados) - set(receta.hiperparametros))
    if desconocidos:
        raise ValueError(
            f"{question}-{model} no acepta {desconocidos}; acepta {list(receta.hiperparametros)}"
        )
    return {**HIPERPARAMETROS.get(question, {}).get(model, {}), **dados}


def resolver_etapa(receta: Modelo, stage: str | None, search: bool) -> str:
    """Deduce stage_of_work (baseline, search o candidate) y rechaza combinaciones incoherentes."""
    deducida = "baseline" if receta.baseline else ("search" if search else "candidate")
    if not stage:
        return deducida
    if stage == "analysis":
        return stage
    if stage != deducida:
        raise ValueError(f"stage={stage!r} no corresponde a este modelo; se esperaba {deducida!r}")
    return stage


def nombre_corrida(question: str, model: str, dados: dict) -> str:
    """Nombre descriptivo: pregunta, modelo e hiperparámetros elegidos explícitamente."""
    sufijo = "".join(f"-{k}{v}" for k, v in sorted(dados.items()))
    return f"{question}-{model}{sufijo}"


def configuracion_efectiva(question, model, etapa, hiperparametros, rejilla=None) -> dict:
    """Valores de config.py y del mlflow run con que se ejecutó la corrida (sin credenciales)."""
    return {
        "question": question,
        "model": model,
        "stage_of_work": etapa,
        "hiperparametros": hiperparametros,
        "rejilla_busqueda": rejilla,
        "experimento": EXPERIMENTO_MLFLOW,
        "particion": {
            "fecha_inicio": FECHA_INICIO,
            "inicio_validacion": INICIO_VALIDACION,
            "inicio_prueba": INICIO_PRUEBA,
            "fecha_fin": FECHA_FIN,
        },
        "semilla": SEMILLA,
        "alfa_riesgo": ALFA_RIESGO,
        "horas_franja": HORAS_FRANJA,
        "baseline_de_referencia": BASELINES[question],
        "metrica_principal": METRICA_PRINCIPAL[question],
        "version_datos": tracking.version_datos(MANIFIESTO_DATOS),
        "fuente_datos": FUENTE_DATOS,
        "python": platform.python_version(),
        "mlflow": mlflow.__version__,
    }


def tipos_confiables(modelo) -> list[str]:
    """Tipos que skops debe aceptar al cargar el modelo: solo scikit-learn, numpy y mibici.

    MLflow serializa los modelos de scikit-learn con skops, que rechaza por seguridad los tipos
    que no conoce. Se declaran explícitamente los de bibliotecas y código propios del proyecto; si
    aparece cualquier otro tipo se detiene el registro en lugar de confiar en él a ciegas.
    """
    from skops.io import dumps, get_untrusted_types

    tipos = get_untrusted_types(data=dumps(modelo))
    ajenos = [t for t in tipos if not t.startswith(("sklearn.", "numpy.", "mibici."))]
    if ajenos:
        raise ValueError(f"El modelo usa tipos no confiables: {ajenos}")
    return tipos


def registrar_modelo(modelo, ejemplo: pd.DataFrame) -> str:
    """Guarda el modelo con firma y ejemplo de entrada; devuelve su URI para el registry.

    MLflow 3 lo guarda como LoggedModel ligado a la corrida (visible en la pestaña de modelos);
    además se copia en la carpeta `model/` de los artefactos de la corrida. `code_paths` incluye
    el paquete mibici para que el modelo cargue aunque use funciones propias (P3).
    """
    firma = infer_signature(ejemplo, modelo.predict(ejemplo))
    opciones = {
        "signature": firma,
        "input_example": ejemplo,
        "code_paths": [str(PROJ_ROOT / "mibici")],
        "serialization_format": "skops",
        "skops_trusted_types": tipos_confiables(modelo),
    }
    info = mlflow.sklearn.log_model(modelo, name="model", **opciones)
    with TemporaryDirectory() as temporal:
        ruta = Path(temporal) / "model"
        mlflow.sklearn.save_model(modelo, ruta, **opciones)
        mlflow.log_artifacts(str(ruta), "model")
    mlflow.set_tags({"model_uri": info.model_uri, "modelo_registrable": "true"})
    return info.model_uri


def registrar_artefactos(question, resultado: Resultado, configuracion: dict, nombre: str):
    """Registra data/, config/, plots/, metrics/ y model/ de una corrida."""
    mlflow.log_artifact(str(MANIFIESTO_DATOS), "data")
    if resultado.entrenamiento is not None:
        conjunto = mlflow.data.from_pandas(
            resultado.entrenamiento,
            source=FUENTE_DATOS,
            name=f"{question}-entrenamiento",
            targets=resultado.objetivo,
        )
        mlflow.log_input(conjunto, context="training")
    mlflow.log_dict(configuracion, "config/config_efectiva.json")
    for archivo in ARCHIVOS_CONFIGURACION:
        if (PROJ_ROOT / archivo).exists():
            mlflow.log_artifact(str(PROJ_ROOT / archivo), "config")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for archivo, figura in resultado.figuras.items():
        mlflow.log_figure(figura, f"plots/{archivo}.png")
        figura.savefig(FIGURES_DIR / f"{nombre}-{archivo}.png", dpi=120, bbox_inches="tight")
        plt.close(figura)
    mlflow.log_dict(resultado.metricas, "metrics/metricas.json")
    for archivo, tabla in resultado.tablas.items():
        mlflow.log_text(tabla.to_csv(index=False), f"metrics/{archivo}.csv")
    if resultado.modelo is not None and resultado.ejemplo is not None:
        registrar_modelo(resultado.modelo, resultado.ejemplo)
    else:
        mlflow.set_tag("modelo_registrable", "false")


def medir_latencia(resultado: Resultado) -> dict[str, float]:
    """Costo de inferencia: milisegundos por cada 1000 filas del conjunto de prueba."""
    if resultado.modelo is None or resultado.x_prueba is None or resultado.x_prueba.empty:
        return {}
    inicio = perf_counter()
    resultado.modelo.predict(resultado.x_prueba)
    milisegundos = (perf_counter() - inicio) * 1000
    return {"test_latencia_ms_por_1000": milisegundos / len(resultado.x_prueba) * 1000}


def metrica_baseline(question: str) -> tuple[str, float] | None:
    """Busca la última corrida baseline de referencia de la pregunta y su métrica de prueba."""
    experimento = MlflowClient().get_experiment_by_name(EXPERIMENTO_MLFLOW)
    if experimento is None:
        return None
    clave = f"test_{METRICA_PRINCIPAL[question]}"
    runs = MlflowClient().search_runs(
        [experimento.experiment_id],
        filter_string=(
            f"tags.question = '{question}' and tags.stage_of_work = 'baseline' "
            f"and params.modelo = '{BASELINES[question]}' and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs or clave not in runs[0].data.metrics:
        return None
    return runs[0].info.run_id, runs[0].data.metrics[clave]


def mejora_contra_baseline(question: str, metricas: dict) -> dict:
    """Mejora porcentual de la métrica principal de prueba respecto al baseline de referencia."""
    principal = METRICA_PRINCIPAL[question]
    referencia = metrica_baseline(question)
    if referencia is None or f"test_{principal}" not in metricas:
        logger.warning(f"Sin baseline {BASELINES[question]!r} registrado para comparar {question}")
        return {}
    run_id, base = referencia
    valor = metricas[f"test_{principal}"]
    if base == 0:
        return {}
    mejora = (base - valor) / abs(base) if MINIMIZAR[question] else (valor - base) / abs(base)
    mlflow.set_tag("baseline_run_id", run_id)
    return {f"test_mejora_{principal}_pct": 100 * mejora}


def es_mejor(question: str, valor: float, actual: float | None) -> bool:
    """Compara dos valores de la métrica principal según su dirección."""
    if actual is None:
        return True
    return valor < actual if MINIMIZAR[question] else valor > actual


def _entrenar(receta: Modelo, conjuntos: dict, hiperparametros: dict):
    """Entrena una receta y mide su costo de entrenamiento (incluye la evaluación)."""
    inicio = perf_counter()
    resultado = receta.entrenar(conjuntos, hiperparametros)
    return resultado, perf_counter() - inicio


def _registrar_corrida(question, receta, resultado, segundos, etapa, configuracion, nombre):
    """Registra métricas, costo, mejora y artefactos de una corrida terminada."""
    metricas = {**resultado.metricas, "train_duracion_s": segundos, **medir_latencia(resultado)}
    if etapa != "baseline":
        metricas.update(mejora_contra_baseline(question, metricas))
    resultado.metricas = metricas
    tracking.registrar_metricas(metricas)
    registrar_artefactos(question, resultado, configuracion, nombre)


def ejecutar(
    question: str,
    model: str | None = None,
    stage: str | None = None,
    hiperparametros: dict | None = None,
    search: bool = False,
    conjuntos: dict | None = None,
) -> str:
    """Entrena y registra un modelo de una pregunta; devuelve el id de la corrida (o la padre).

    Sin `model` entrena el baseline de referencia. Con `search=True` organiza una búsqueda: una
    corrida padre y una hija por combinación de la rejilla de config.py; las hijas solo registran
    métricas de train y validación, y la padre registra la mejor combinación (elegida en
    validación) con sus métricas de prueba, sus artefactos y su modelo.
    """
    question = question.lower()
    modulo = modulo_pregunta(question)
    model = model or BASELINES[question]
    if model not in modulo.MODELOS:
        raise ValueError(f"{question} no tiene el modelo {model!r}; usa {list(modulo.MODELOS)}")
    receta = modulo.MODELOS[model]
    dados = dict(hiperparametros or {})
    etapa = resolver_etapa(receta, stage, search)
    if conjuntos is None:
        conjuntos = modulo.preparar_conjuntos()
    etiquetas = {"stage_of_work": etapa, "model_family": receta.familia}

    if not search:
        finales = resolver_hiperparametros(question, model, receta, dados)
        nombre = nombre_corrida(question, model, dados)
        resultado, segundos = _entrenar(receta, conjuntos, finales)
        configuracion = configuracion_efectiva(question, model, etapa, finales)
        params = parametros_para_registrar(resultado.params)
        with tracking.iniciar_run(question, nombre, params, etiquetas) as run:
            _registrar_corrida(question, receta, resultado, segundos, etapa, configuracion, nombre)
            _resumen(question, nombre, resultado.metricas)
            return run.info.run_id

    rejilla = REJILLAS.get(question, {}).get(model)
    if not rejilla:
        raise ValueError(f"No hay rejilla de búsqueda para {question}-{model} en config.py")
    base = resolver_hiperparametros(question, model, receta, dados)
    principal = f"val_{METRICA_PRINCIPAL[question]}"
    nombre = f"{question}-busqueda-{model}"
    filas, mejor = [], None
    with tracking.iniciar_run(question, nombre, {"modelo": model}, etiquetas) as padre:
        for combinacion in ParameterGrid(rejilla):
            finales = {**base, **combinacion}
            hijo = nombre_corrida(question, model, combinacion)
            resultado, segundos = _entrenar(receta, conjuntos, finales)
            params = parametros_para_registrar(resultado.params)
            with tracking.iniciar_run_anidado(hijo, params, {"search_role": "child"}):
                seleccion = {
                    k: v for k, v in resultado.metricas.items() if k.startswith(("train_", "val_"))
                }
                tracking.registrar_metricas({**seleccion, "train_duracion_s": segundos})
            for figura in resultado.figuras.values():
                plt.close(figura)
            valor = resultado.metricas.get(principal)
            filas.append({**combinacion, principal: valor, "train_duracion_s": segundos})
            if valor is not None and es_mejor(question, valor, mejor and mejor[0]):
                mejor = (valor, combinacion, resultado, segundos)
        if mejor is None:
            raise ValueError(f"Ninguna combinación produjo {principal}")
        _, combinacion, resultado, segundos = mejor
        mlflow.log_params({f"mejor_{k}": v for k, v in combinacion.items()})
        params = parametros_para_registrar(resultado.params)
        mlflow.log_params({k: v for k, v in params.items() if k != "modelo"})
        mlflow.set_tag("search_role", "parent")
        resultado.tablas = {**resultado.tablas, "busqueda": pd.DataFrame(filas)}
        configuracion = configuracion_efectiva(
            question, model, etapa, {**base, **combinacion}, rejilla
        )
        _registrar_corrida(question, receta, resultado, segundos, etapa, configuracion, nombre)
        _resumen(question, nombre, resultado.metricas)
        return padre.info.run_id


def _resumen(question: str, nombre: str, metricas: dict) -> None:
    """Muestra en la terminal la métrica principal de validación y prueba."""
    principal = METRICA_PRINCIPAL[question]
    logger.success(
        "{} | val_{}={:.6f} | test_{}={:.6f}",
        nombre,
        principal,
        metricas.get(f"val_{principal}", float("nan")),
        principal,
        metricas.get(f"test_{principal}", float("nan")),
    )
