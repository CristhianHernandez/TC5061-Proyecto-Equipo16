"""Servicio model-init del stack de Compose: registra el champion de P1 sin duplicar ni degradar.

Usa un MLflow temporal (SQLite y artefactos locales); no requiere Docker ni el registry del equipo.
"""

import json

import mlflow
from mlflow import MlflowClient
import pytest
from sklearn.linear_model import LinearRegression

from mibici.config import ALIAS_PRODUCCION, ALIAS_RETADOR, EJEMPLO_SERVICIO
from mibici.serving import api, inicializar, model_loader
from mibici.serving.schemas import FIRMA_P1


@pytest.fixture
def registry(mlflow_sesion, monkeypatch, request):
    """Registry temporal compartido; cada prueba usa un nombre de modelo propio."""
    for modulo in (inicializar, model_loader):
        monkeypatch.setattr(modulo, "MLFLOW_TRACKING_URI", mlflow_sesion["uri"])
    mlflow.set_tracking_uri(mlflow_sesion["uri"])
    return {**mlflow_sesion, "nombre": f"init-{request.node.name}"[:60]}


def copia_exportada(ruta, version="2", run_id="corrida-origen"):
    """Carpeta como la que deja `make export-model`: modelo con la firma de P1 y su linaje."""
    inicializar.guardar_modelo_de_prueba(ruta)
    (ruta / "modelo_servido.json").write_text(
        json.dumps(
            {
                "version": version,
                "run_id": run_id,
                "linaje": {"git_commit": "c30cd49", "version_datos": "4d8c2a1292fc"},
            }
        ),
        encoding="utf-8",
    )
    return ruta


def test_el_modelo_de_prueba_tiene_la_firma_exacta_del_contrato(tmp_path):
    ruta = inicializar.guardar_modelo_de_prueba(tmp_path / "modelo")
    assert inicializar.firma_de(ruta) == list(FIRMA_P1)
    modelo = mlflow.pyfunc.load_model(str(ruta))
    prediccion = modelo.predict(modelo.input_example)
    assert len(prediccion) == 5 and (prediccion > 0).all()
    otra = mlflow.pyfunc.load_model(str(inicializar.guardar_modelo_de_prueba(tmp_path / "otra")))
    assert (otra.predict(otra.input_example) == prediccion).all()  # determinista


@pytest.mark.integracion
def test_sin_copia_registra_un_modelo_de_prueba_una_sola_vez(registry, tmp_path):
    nombre = registry["nombre"]
    primero = inicializar.inicializar(tmp_path / "sin-copia", nombre)
    assert primero["accion"] == "registrado" and primero["origen"] == "modelo-de-prueba"
    cliente = MlflowClient()
    assert cliente.get_model_version_by_alias(nombre, ALIAS_PRODUCCION).version == 1
    segundo = inicializar.inicializar(tmp_path / "sin-copia", nombre)
    assert segundo == {"accion": "sin_cambios", "version": "1", "origen": "modelo-de-prueba"}
    assert len(cliente.search_model_versions(f"name = '{nombre}'")) == 1


@pytest.mark.integracion
def test_la_copia_del_champion_se_registra_con_linaje_y_es_idempotente(registry, tmp_path):
    nombre = registry["nombre"]
    inicializar.inicializar(tmp_path / "sin-copia", nombre)  # registry con modelo de prueba
    bundle = copia_exportada(tmp_path / "model_bundle")
    resultado = inicializar.inicializar(bundle, nombre)
    assert resultado["accion"] == "registrado" and resultado["origen"] == "bundle"
    assert resultado["version"] == "2" and resultado["challenger"] == "1"
    cliente = MlflowClient()
    version = cliente.get_model_version_by_alias(nombre, ALIAS_PRODUCCION)
    assert version.tags["version_origen"] == "2"
    assert version.tags["run_id_origen"] == "corrida-origen"
    assert version.tags["git_commit"] == "c30cd49"
    assert cliente.get_model_version_by_alias(nombre, ALIAS_RETADOR).version == 1
    # Recrear contenedores (misma copia) no crea versiones nuevas
    assert inicializar.inicializar(bundle, nombre)["accion"] == "sin_cambios"
    # Sin copia (otra imagen, CI) no se degrada el champion real a un modelo de prueba
    sin_copia = inicializar.inicializar(tmp_path / "sin-copia", nombre)
    assert sin_copia == {"accion": "sin_cambios", "version": "2", "origen": "bundle"}
    assert len(cliente.search_model_versions(f"name = '{nombre}'")) == 2


@pytest.mark.limite
def test_una_copia_con_otro_contrato_se_rechaza_sin_registrar_nada(registry, tmp_path):
    ajena = tmp_path / "otra-firma"
    mlflow.sklearn.save_model(
        LinearRegression().fit([[1.0], [2.0]], [1.0, 2.0]),
        str(ajena),
        signature=mlflow.models.infer_signature([[1.0]], [1.0]),
    )
    with pytest.raises(ValueError, match="no coincide con el contrato"):
        inicializar.inicializar(ajena, registry["nombre"])
    assert MlflowClient().search_model_versions(f"name = '{registry['nombre']}'") == []


@pytest.mark.integracion
def test_la_api_carga_desde_el_registry_inicializado_y_predice(registry, tmp_path):
    nombre = registry["nombre"]
    inicializar.inicializar(copia_exportada(tmp_path / "model_bundle"), nombre)
    servido = model_loader.cargar_modelo(
        tmp_path / "sin-copia-local", f"models:/{nombre}@champion"
    )
    assert servido.metadatos["origen"] == "registry"
    api._validar_contrato(servido)  # la API no arrancaría si la firma difiriera
    registros = json.loads(EJEMPLO_SERVICIO.read_text(encoding="utf-8"))["registros"]
    predicciones = servido.predecir(registros)
    assert len(predicciones) == 2 and all(p > 0 for p in predicciones)
