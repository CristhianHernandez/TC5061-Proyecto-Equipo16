"""Servicio de inferencia: exportar el champion del registry, cargarlo y validar su contrato.

Usa un MLflow temporal con un modelo pequeño que tiene la misma firma que el champion de P1
(11 variables, rezagos opcionales); no requiere el servidor del equipo ni datos reales.
"""

import json

import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from typer.testing import CliRunner

from mibici.config import ALIAS_PRODUCCION, SEMILLA
from mibici.modeling import comun
from mibici.modeling.p1_demanda import VARIABLES
from mibici.serving import exportar as cli
from mibici.serving import model_loader
from mibici.serving.model_loader import ErrorEntrada, ModeloNoDisponible, ModeloServido, Variable

NOMBRE = "serving-prueba-p1"
URI_CHAMPION = f"models:/{NOMBRE}@{ALIAS_PRODUCCION}"
REZAGOS = ["demanda_lag_1", "demanda_lag_7", "demanda_media_7"]


def datos_p1(n=120, semilla=SEMILLA):
    """Variables de P1 con los dtypes del champion real; los primeros días no tienen rezagos."""
    azar = np.random.default_rng(semilla)
    fechas = pd.date_range("2025-01-01", periods=n)
    datos = pd.DataFrame(
        {
            "dia_semana": fechas.dayofweek.astype("int32"),
            "mes": fechas.month.astype("int32"),
            "es_fin_semana": (fechas.dayofweek >= 5).astype("int64"),
            "dia_anio": fechas.dayofyear.astype("int32"),
            "es_festivo": np.zeros(n, dtype="int64"),
            "demanda_lag_1": azar.poisson(30, n).astype(float),
            "demanda_lag_7": azar.poisson(30, n).astype(float),
            "demanda_media_7": azar.uniform(10, 50, n),
            "capacidad": azar.choice([15, 20, 39], n).astype("int32"),
            "latitud": azar.uniform(20.63, 20.74, n),
            "longitud": azar.uniform(-103.43, -103.30, n),
        }
    )[VARIABLES]
    datos.loc[:6, REZAGOS] = np.nan
    return datos, azar.poisson(30, n)


def registrar_version(max_iter):
    """Entrena, registra con el mismo helper de producción y devuelve (versión, modelo)."""
    datos, demanda = datos_p1()
    modelo = HistGradientBoostingRegressor(
        loss="poisson", max_iter=max_iter, random_state=SEMILLA
    ).fit(datos, demanda)
    with mlflow.start_run(run_name=f"serving-prueba-{max_iter}"):
        uri = comun.registrar_modelo(modelo, datos.head(5))
        mlflow.set_tags({"git_commit": "abc123", "version_datos": "prueba"})
    version = mlflow.register_model(uri, NOMBRE)
    MlflowClient().set_model_version_tag(NOMBRE, version.version, "git_commit", "abc123")
    return str(version.version), modelo


@pytest.fixture(scope="module")
def registry(mlflow_sesion):
    """Dos versiones registradas; champion apunta a la segunda (la más reciente)."""
    with pytest.MonkeyPatch.context() as parche:
        parche.setattr(model_loader, "MLFLOW_TRACKING_URI", mlflow_sesion["uri"])
        mlflow.set_tracking_uri(mlflow_sesion["uri"])
        mlflow.set_experiment("serving-pruebas")
        v1, _ = registrar_version(5)
        v2, modelo = registrar_version(20)
        cliente = MlflowClient()
        cliente.set_registered_model_alias(NOMBRE, ALIAS_PRODUCCION, v1)
        cliente.set_registered_model_alias(NOMBRE, ALIAS_PRODUCCION, v2)  # promoción
        yield {**mlflow_sesion, "v1": v1, "v2": v2, "modelo": modelo}


@pytest.fixture(scope="module")
def copia(registry, tmp_path_factory):
    """Champion exportado a una carpeta local, como el model_bundle/ de la imagen."""
    destino = tmp_path_factory.mktemp("bundle") / "model_bundle"
    metadatos = model_loader.exportar(destino, URI_CHAMPION)
    return destino, metadatos


@pytest.fixture(scope="module")
def servido(copia):
    return model_loader.cargar_modelo(copia[0])


def registros(n=3):
    datos, _ = datos_p1()
    return datos.iloc[10 : 10 + n].astype(object).to_dict("records")


# --- registry y exportación -------------------------------------------------------------------


@pytest.mark.integracion
def test_exporta_la_version_a_la_que_apunta_el_alias_con_su_linaje(registry, copia):
    destino, metadatos = copia
    assert (destino / "MLmodel").is_file()
    assert metadatos["nombre"] == NOMBRE
    assert metadatos["version"] == registry["v2"] != registry["v1"]
    assert metadatos["alias"] == ALIAS_PRODUCCION
    assert metadatos["uri_version"] == f"models:/{NOMBRE}/{registry['v2']}"
    assert metadatos["linaje"]["git_commit"] == "abc123"
    guardado = json.loads((destino / "modelo_servido.json").read_text(encoding="utf-8"))
    assert guardado == metadatos


@pytest.mark.integracion
def test_la_copia_local_predice_igual_que_el_modelo_registrado(registry, servido):
    assert servido.metadatos["origen"] == "copia_local"
    assert servido.metadatos["version"] == registry["v2"]
    datos, _ = datos_p1()
    esperado = registry["modelo"].predict(datos.iloc[10:13])
    assert servido.predecir(registros()) == pytest.approx(esperado.tolist(), rel=1e-12)


@pytest.mark.integracion
def test_sin_copia_local_carga_del_registry_por_nombre_y_alias(registry, tmp_path):
    en_linea = model_loader.cargar_modelo(tmp_path / "no-existe", URI_CHAMPION)
    assert en_linea.metadatos["origen"] == "registry"
    assert en_linea.metadatos["version"] == registry["v2"]
    fijo = model_loader.cargar_modelo(tmp_path / "no-existe", f"models:/{NOMBRE}/{registry['v1']}")
    assert fijo.metadatos["version"] == registry["v1"] and fijo.metadatos["alias"] is None


@pytest.mark.integracion
def test_reexportar_reemplaza_la_copia_pero_no_borra_carpetas_ajenas(registry, tmp_path):
    destino = tmp_path / "model_bundle"
    model_loader.exportar(destino, URI_CHAMPION)
    model_loader.exportar(destino, URI_CHAMPION)
    assert (destino / "MLmodel").is_file()
    ajena = tmp_path / "otra"
    ajena.mkdir()
    (ajena / "importante.txt").write_text("no borrar")
    with pytest.raises(ValueError, match="no es una copia"):
        model_loader.exportar(ajena, URI_CHAMPION)
    assert (ajena / "importante.txt").read_text() == "no borrar"


@pytest.mark.limite
def test_modelo_inexistente_o_uri_invalida_dan_un_error_claro(registry, tmp_path):
    with pytest.raises(ModeloNoDisponible, match="no se pudo cargar"):
        model_loader.cargar_modelo(tmp_path, "models:/no-existe@champion")
    with pytest.raises(ValueError, match="models:/"):
        model_loader.cargar_modelo(tmp_path, "runs:/abc/model")


@pytest.mark.integracion
def test_cli_exporta_y_predice_el_ejemplo_fuera_de_la_api(registry, tmp_path):
    ejemplo = tmp_path / "ejemplo.json"
    ejemplo.write_text(json.dumps({"registros": registros(2)}))
    argumentos = ["--destino", str(tmp_path / "bundle"), "--model-uri", URI_CHAMPION]
    resultado = CliRunner().invoke(cli.app, [*argumentos, "--ejemplo", str(ejemplo)])
    assert resultado.exit_code == 0, resultado.output
    salida = json.loads(resultado.stdout[resultado.stdout.index("{") :])
    assert salida["modelo"]["version"] == registry["v2"]
    assert len(salida["predicciones"]) == 2


# --- contrato de entrada y salida -------------------------------------------------------------


def test_contrato_sigue_la_firma_con_rezagos_opcionales(servido):
    contrato = servido.contrato()
    assert [v["nombre"] for v in contrato] == VARIABLES
    opcionales = {v["nombre"] for v in contrato if not v["requerida"]}
    assert opcionales == set(REZAGOS)
    tipos = {v["nombre"]: v["tipo"] for v in contrato}
    assert tipos["dia_semana"] == "integer" and tipos["es_festivo"] == "long"
    assert tipos["latitud"] == "double"


def test_salida_es_json_y_no_depende_del_orden_de_las_columnas(servido):
    normales = registros()
    invertidos = [dict(reversed(list(r.items()))) for r in normales]
    prediccion = servido.predecir(invertidos)
    assert prediccion == servido.predecir(normales)
    assert all(type(p) is float and p > 0 for p in prediccion)  # Poisson: demanda positiva
    json.dumps(prediccion, allow_nan=False)


def test_acepta_un_registro_suelto_y_rezagos_nulos_o_ausentes(servido):
    registro = registros(1)[0]
    assert len(servido.predecir(registro)) == 1
    sin_rezagos = {k: v for k, v in registro.items() if k not in REZAGOS}
    con_nulos = {**registro, **dict.fromkeys(REZAGOS)}
    assert servido.predecir(sin_rezagos) == servido.predecir(con_nulos)


def test_el_ejemplo_registrado_con_el_modelo_es_valido(servido):
    ejemplo = servido.ejemplo()
    assert len(ejemplo) == 5 and ejemplo[0]["demanda_lag_1"] is None
    assert len(servido.predecir(ejemplo)) == 5


@pytest.mark.limite
@pytest.mark.parametrize(
    ("cambio", "mensaje"),
    [
        ({"quitar": "capacidad"}, "faltantes: \\['capacidad'\\]"),
        ({"estacion_id": 51}, "no esperadas: \\['estacion_id'\\]"),
        ({"capacidad": "veinte"}, "capacidad: se esperaba un número"),
        ({"es_festivo": True}, "es_festivo: se esperaba un número"),
        ({"dia_semana": 2.5}, "dia_semana: se esperaba un número entero"),
        ({"latitud": None}, "latitud: es obligatoria"),
        ({"demanda_lag_1": float("inf")}, "infinitos"),
        ({"capacidad": 2**40}, "fuera del rango de int32"),
    ],
)
def test_rechaza_entradas_que_no_cumplen_el_contrato(servido, cambio, mensaje):
    registro = registros(1)[0]
    registro.pop(cambio.pop("quitar", None), None)
    registro.update(cambio)
    with pytest.raises(ErrorEntrada, match=mensaje):
        servido.predecir([registro])


@pytest.mark.limite
@pytest.mark.parametrize("entrada", [[], [42], ["texto"]])
def test_rechaza_peticiones_vacias_o_sin_objetos(servido, entrada):
    with pytest.raises(ErrorEntrada):
        servido.predecir(entrada)


@pytest.mark.limite
def test_una_salida_no_finita_del_modelo_es_error_del_servidor():
    class SalidaInvalida:
        def predict(self, datos):
            return np.full(len(datos), np.nan)

    variables = [Variable("x", "double", True)]
    with pytest.raises(RuntimeError, match="no finitos"):
        ModeloServido(SalidaInvalida(), {}, variables).predecir([{"x": 1.0}])
