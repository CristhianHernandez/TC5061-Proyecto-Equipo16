"""Reproducibilidad del entrenamiento: misma semilla, mismos datos y mismos hiperparámetros dan el
mismo resultado.

Se verifica en cuatro niveles:

1. **Receta**: entrenar dos veces cada candidata real de P1, P2 y P3 produce predicciones idénticas.
2. **Semilla**: cambiar la semilla sí cambia el resultado de un modelo aleatorio; si el código
   ignorara config.SEMILLA, esta prueba fallaría.
3. **Proceso**: dos intérpretes de Python independientes, con distinto PYTHONHASHSEED, producen las
   mismas predicciones (descarta dependencias del orden de diccionarios o conjuntos).
4. **Corrida de MLflow**: dos ejecuciones de `comun.ejecutar` registran las mismas métricas y el
   mismo modelo.

Tolerancia: en la misma plataforma y con las mismas versiones de dependencias, los algoritmos con
semilla fija son deterministas, así que las predicciones se comparan **exactas**. Las métricas se
comparan con una tolerancia absoluta de 1e-12, que solo absorbe diferencias de redondeo en sumas de
punto flotante. No se exige igualdad entre plataformas o versiones distintas (BLAS, sistema
operativo o scikit-learn pueden cambiar el último bit).
"""

import hashlib
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pytest

from mibici.config import PROJ_ROOT, SEMILLA
from mibici.modeling import comun

TOLERANCIA_METRICAS = 1e-12
# Métricas que miden tiempo de cómputo: varían entre ejecuciones por diseño
METRICAS_DE_TIEMPO = ("train_duracion_s", "test_latencia_ms_por_1000")

# Una candidata de cada familia aleatoria o iterativa, con tamaños pequeños para que la prueba sea rápida
CANDIDATAS = [
    pytest.param("p1", "hgb", {"max_iter": 30}, id="p1-hgb"),
    pytest.param("p1", "rf", {"n_estimators": 15}, id="p1-rf"),
    pytest.param("p2", "rf", {"n_estimators": 15, "max_depth": 5}, id="p2-rf"),
    pytest.param("p2", "hgb", {"max_iter": 30}, id="p2-hgb"),
    pytest.param("p3", "kmeans", {"k": 3}, id="p3-kmeans"),
]


def entrenar(question, model, dados, conjuntos):
    """Entrena una receta real con los hiperparámetros resueltos igual que `mlflow run . -e train`."""
    receta = comun.modulo_pregunta(question).MODELOS[model]
    finales = comun.resolver_hiperparametros(question, model, receta, dados)
    resultado = receta.entrenar(conjuntos[question], finales)
    # Las recetas generan figuras para MLflow; aquí no se registran, así que se cierran
    for figura in resultado.figuras.values():
        plt.close(figura)
    return resultado


def salida(resultado):
    """Predicción del modelo sobre prueba; en clasificación, probabilidades (más sensibles que la clase)."""
    modelo, entradas = resultado.modelo, resultado.x_prueba
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(entradas)
    return modelo.predict(entradas)


@pytest.mark.reproducibilidad
@pytest.mark.parametrize(("question", "model", "dados"), CANDIDATAS)
def test_misma_semilla_mismas_predicciones_y_metricas(question, model, dados, conjuntos):
    primero = entrenar(question, model, dados, conjuntos)
    segundo = entrenar(question, model, dados, conjuntos)

    np.testing.assert_array_equal(salida(primero), salida(segundo))
    assert primero.metricas.keys() == segundo.metricas.keys()
    for clave, valor in primero.metricas.items():
        assert segundo.metricas[clave] == pytest.approx(valor, abs=TOLERANCIA_METRICAS), clave
    assert primero.params == segundo.params


@pytest.mark.reproducibilidad
def test_la_semilla_controla_el_azar(conjuntos, monkeypatch):
    """El bosque aleatorio depende de la semilla: con otra cambia, y al volver a 16 se recupera."""
    from mibici.modeling import p2_riesgo

    dados = {"n_estimators": 15, "max_depth": 5}
    original = salida(entrenar("p2", "rf", dados, conjuntos))

    monkeypatch.setattr(p2_riesgo, "SEMILLA", SEMILLA + 1)
    otra = salida(entrenar("p2", "rf", dados, conjuntos))
    assert not np.array_equal(original, otra), "la semilla no se está usando en el entrenamiento"

    monkeypatch.setattr(p2_riesgo, "SEMILLA", SEMILLA)
    np.testing.assert_array_equal(original, salida(entrenar("p2", "rf", dados, conjuntos)))


# Entrena en un proceso nuevo y escribe la huella de las predicciones en stdout
GUION_PROCESO = """
import hashlib, pickle, sys, warnings
warnings.filterwarnings("ignore")
from mibici.modeling import comun
conjuntos = pickle.load(open(sys.argv[1], "rb"))
receta = comun.modulo_pregunta("p2").MODELOS["rf"]
dados = comun.resolver_hiperparametros("p2", "rf", receta, {"n_estimators": 15, "max_depth": 5})
resultado = receta.entrenar(conjuntos, dados)
print(hashlib.sha256(resultado.modelo.predict_proba(resultado.x_prueba).tobytes()).hexdigest())
"""


@pytest.mark.reproducibilidad
def test_reproducible_en_procesos_independientes(conjuntos, tmp_path):
    """Dos procesos con distinto PYTHONHASHSEED obtienen exactamente las mismas predicciones."""
    import pickle

    datos = tmp_path / "conjuntos_p2.pkl"
    datos.write_bytes(pickle.dumps(conjuntos["p2"]))
    huellas = []
    for semilla_hash in ("0", "12345"):
        entorno = {**os.environ, "PYTHONHASHSEED": semilla_hash, "PYTHONPATH": str(PROJ_ROOT)}
        proceso = subprocess.run(
            [sys.executable, "-c", GUION_PROCESO, str(datos)],
            capture_output=True,
            text=True,
            env=entorno,
            cwd=tmp_path,
            check=True,
        )
        huellas.append(proceso.stdout.strip().splitlines()[-1])

    en_esta_sesion = entrenar("p2", "rf", {"n_estimators": 15, "max_depth": 5}, conjuntos)
    esperada = hashlib.sha256(salida(en_esta_sesion).tobytes()).hexdigest()
    assert huellas == [esperada, esperada]


@pytest.mark.reproducibilidad
@pytest.mark.integracion
@pytest.mark.parametrize(("question", "model", "dados"), CANDIDATAS[::2])
def test_dos_corridas_de_mlflow_registran_lo_mismo(
    question, model, dados, conjuntos, mlflow_aislado
):
    """Dos `ejecutar` iguales registran las mismas métricas, parámetros y un modelo equivalente."""
    corridas = [
        mlflow.get_run(comun.ejecutar(question, model, None, dados, False, conjuntos[question]))
        for _ in range(2)
    ]
    primera, segunda = (
        {k: v for k, v in c.data.metrics.items() if k not in METRICAS_DE_TIEMPO} for c in corridas
    )
    assert primera.keys() == segunda.keys()
    for clave, valor in primera.items():
        assert segunda[clave] == pytest.approx(valor, abs=TOLERANCIA_METRICAS), clave
    assert corridas[0].data.params == corridas[1].data.params
    assert corridas[0].data.tags["version_datos"] == corridas[1].data.tags["version_datos"]

    modelos = [mlflow.pyfunc.load_model(c.data.tags["model_uri"]) for c in corridas]
    columnas = modelos[0].metadata.get_input_schema().input_names()
    entradas = conjuntos[question]["test"][columnas]
    np.testing.assert_array_equal(
        np.asarray(modelos[0].predict(entradas)), np.asarray(modelos[1].predict(entradas))
    )
