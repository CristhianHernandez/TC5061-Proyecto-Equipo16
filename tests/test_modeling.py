"""Contratos DH4 con datos sintéticos y MLflow local, sin viajes de disco ni red."""

import inspect

import mlflow
import numpy as np
import pandas as pd
import pytest

from mibici import features, tracking
from mibici.config import (
    ALFA_RIESGO,
    CLASES_RIESGO,
    FECHA_FIN,
    FECHA_INICIO,
    HORAS_FRANJA,
    INICIO_PRUEBA,
    INICIO_VALIDACION,
    SEMILLA,
)
from mibici.modeling import comun
from mibici.modeling import p1_demanda as p1
from mibici.modeling import p2_riesgo as p2
from mibici.modeling import p3_clusters as p3


@pytest.fixture(scope="module")
def sinteticos():
    """Construye datos pequeños con todas las clases, periodos y perfiles de P1/P2/P3."""
    azar = np.random.default_rng(SEMILLA)
    estaciones = pd.DataFrame(
        {
            "estacion_id": range(12),
            "capacidad": [20] * 12,
            "latitud": np.linspace(20, 21, 12),
            "longitud": np.linspace(-104, -103, 12),
            "municipio": np.repeat(["GDL", "ZPN", "TLQ"], 4),
        }
    )
    fechas = [
        pd.date_range(corte, periods=12)
        for corte in (FECHA_INICIO, INICIO_VALIDACION, INICIO_PRUEBA)
    ]
    demanda = pd.DataFrame(
        [
            {
                "estacion_id": estacion,
                "fecha": fecha,
                "demanda": 1 + estacion * 2 + i % 3,
                "demanda_lag_1": 1 + estacion * 2,
                "demanda_lag_7": np.nan if i == 0 else 2 + estacion * 2,
                "demanda_media_7": 2 + estacion * 2,
            }
            for periodo in fechas
            for i, fecha in enumerate(periodo)
            for estacion in range(12)
        ]
    )
    demanda = features.agregar_features_calendario(demanda)
    riesgo = demanda[["estacion_id", "fecha", "dia_semana"]].rename(
        columns={"fecha": "fecha_franja"}
    )
    valores = np.tile([-12.0, 0.0, 12.0], len(riesgo) // 3)
    riesgo = riesgo.assign(
        capacidad=20,
        hora=np.arange(len(riesgo)) % 8 * HORAS_FRANJA,
        flujo_lag_1=valores,
        flujo_lag_56=valores,
    )
    riesgo["riesgo"] = p2.predecir_persistencia(riesgo)
    perfiles = estaciones[["estacion_id"]].copy()
    for tipo in ("salidas", "llegadas"):
        for dia in ("laboral", "fin_semana"):
            matriz = azar.uniform(0, 0.1, (12, 24))
            for i in range(12):
                matriz[i, (i // 4) * 8] += 2
            matriz /= matriz.sum(axis=1, keepdims=True)
            for h in range(24):
                perfiles[f"{tipo}_{dia}_{h:02d}"] = matriz[:, h]
    perfiles = perfiles.assign(
        total_salidas=np.arange(12) * 10 + 10,
        total_llegadas=np.arange(12) * 12 + 10,
        duracion_media_min=[np.nan] + [15.0] * 11,
    )
    perfiles = perfiles.copy()
    perfiles["volumen_total"] = perfiles["total_salidas"] + perfiles["total_llegadas"]
    perfiles["balance_neto"] = perfiles["total_salidas"] - perfiles["total_llegadas"]
    return estaciones, demanda, riesgo, perfiles


@pytest.fixture(scope="module")
def ejecuciones(tmp_path_factory, sinteticos):
    """Ejecuta dos veces los diez runs en SQLite temporal con las mismas semillas."""
    ruta = tmp_path_factory.mktemp("modelos")
    with pytest.MonkeyPatch.context() as parche:
        parche.chdir(ruta)
        manifiesto = ruta / "manifiesto.json"
        manifiesto.write_text('{"archivos": []}')
        parche.setattr(tracking, "MANIFIESTO_DATOS", manifiesto)
        parche.setattr(tracking, "MLFLOW_TRACKING_URI", f"sqlite:///{ruta / 'mlflow.db'}")
        parche.setattr(comun, "FIGURES_DIR", ruta / "figuras")
        estaciones, demanda, riesgo, perfiles = sinteticos
        resultados = []
        for _ in range(2):
            ids = p1.entrenar_p1(demanda, estaciones)
            ids += p2.entrenar_p2(riesgo, estaciones)
            ids += p3.entrenar_p3(perfiles, perfiles.copy(), perfiles.copy(), estaciones)
            resultados.append([mlflow.get_run(run) for run in ids])
        yield resultados, mlflow.MlflowClient(), ruta


def test_diez_runs_completos_metricas_tags_artefactos(ejecuciones):
    resultados, cliente, ruta = ejecuciones
    for run in resultados[0]:
        assert run.info.status == "FINISHED"
        experimento = cliente.get_experiment(run.info.experiment_id)
        assert experimento.name == "mibici-equipo16"
        assert run.data.tags["equipo"] == "16"
        pregunta = run.data.tags["pregunta"]
        assert pregunta in ("P1", "P2", "P3")
        assert all(k.startswith(("train_", "val_", "test_")) for k in run.data.metrics)
        principal = {"P1": "mae", "P2": "f1_macro", "P3": "silueta"}[pregunta]
        assert all(f"{c}_{principal}" in run.data.metrics for c in ("train", "val", "test"))
        artefactos = {a.path for a in cliente.list_artifacts(run.info.run_id)}
        assert "metricas.json" in artefactos
        assert any(a.endswith(".png") for a in artefactos)
        assert run.data.params["n_train"]
        nombre = run.data.tags["mlflow.runName"]
        if nombre in ("p1-hgb-poisson", "p2-hgb"):
            assert "learning_rate" in run.data.params
            assert "rejilla.csv" in artefactos
        if nombre == "p2-logistic-regression":
            assert "C" in run.data.params
            assert "rejilla.csv" in artefactos
        if pregunta == "P3":
            assert "k" in run.data.params
            if nombre != "p3-municipio":
                assert "busqueda_k.csv" in artefactos
            if nombre == "p3-agglomerative-ward":
                assert "train_ari_semillas" not in run.data.metrics
                assert run.data.params["ari_semillas"] == "no aplica (determinista)"
    assert len(resultados[0]) == 10
    assert len(list((ruta / "figuras").glob("*.png"))) == 17


def test_misma_semilla_metricas_identicas(ejecuciones):
    resultados, _, _ = ejecuciones
    for primero, segundo in zip(*resultados):
        assert primero.data.metrics == pytest.approx(segundo.data.metrics, abs=1e-12)
        assert primero.data.params == segundo.data.params


def test_naive_manual_respaldo_y_promedio_solo_train():
    entrenamiento = pd.DataFrame(
        {"estacion_id": [1, 1, 2], "dia_semana": [0, 0, 1], "demanda": [2, 6, 10]}
    )
    futuro = pd.DataFrame(
        {
            "estacion_id": [1, 1, 99],
            "dia_semana": [0, 0, 0],
            "demanda_lag_7": [7, np.nan, np.nan],
            "demanda": [100] * 3,
        }
    )
    assert p1.predecir_naive(entrenamiento, futuro) == pytest.approx([7, 4, 6])
    assert p1.predecir_promedio(entrenamiento, futuro) == pytest.approx([4, 4, 6])
    cambiado = futuro.assign(demanda=-999)
    np.testing.assert_array_equal(
        p1.predecir_naive(entrenamiento, futuro), p1.predecir_naive(entrenamiento, cambiado)
    )
    np.testing.assert_array_equal(
        p1.predecir_promedio(entrenamiento, futuro), p1.predecir_promedio(entrenamiento, cambiado)
    )


def test_persistencia_manual_y_ap_indicador():
    datos = pd.DataFrame({"flujo_lag_56": [10, -10, 9, -9, np.nan], "capacidad": [20] * 5})
    pred = p2.predecir_persistencia(datos)
    assert list(pred) == ["sin_bicicletas", "sin_anclajes", "normal", "normal", "normal"]
    puntuaciones = np.column_stack([pred == c for c in CLASES_RIESGO])
    metricas = p2.metricas_riesgo(pred, pred, puntuaciones, CLASES_RIESGO)
    assert metricas["f1_macro"] == 1
    assert metricas["ap_macro_riesgo"] == 1


def test_seleccion_no_recibe_prueba_y_ajustes_no_aprenden_validacion(sinteticos):
    estaciones, demanda, riesgo, _ = sinteticos
    demanda = demanda.merge(estaciones.drop(columns="municipio"), on="estacion_id")
    riesgo = riesgo.merge(estaciones[["estacion_id", "latitud", "longitud"]], on="estacion_id")
    for datos, fecha, objetivo, seleccionar, variables in (
        (demanda, "fecha", "demanda", p1.seleccionar_demanda, p1.VARIABLES),
        (
            riesgo,
            "fecha_franja",
            "riesgo",
            lambda a, b: p2.seleccionar_riesgo(a, b, "logistic-regression"),
            p2.VARIABLES,
        ),
        (
            riesgo,
            "fecha_franja",
            "riesgo",
            lambda a, b: p2.seleccionar_riesgo(a, b, "hgb"),
            p2.VARIABLES_HGB,
        ),
    ):
        train, val, test = features.particion_temporal(datos, fecha)
        modelo, params, _ = seleccionar(train, val)
        alterados = datos.copy()
        mascara_test = alterados.index.isin(test.index)
        alterados.loc[mascara_test, objetivo] = (
            999 if objetivo == "demanda" else "sin_bicicletas"
        )
        nuevo_train, nuevo_val, _ = features.particion_temporal(alterados, fecha)
        nuevo, nuevos_params, _ = seleccionar(nuevo_train, nuevo_val)
        assert params == nuevos_params
        np.testing.assert_array_equal(
            modelo.predict(train[variables]), nuevo.predict(train[variables])
        )
        # Cambiar val puede cambiar la elección, pero cada candidato sigue ajustado solo en train.
        val_cambiada = val.copy()
        val_cambiada[objetivo] = 999 if objetivo == "demanda" else "sin_anclajes"
        cambiado, _, _ = seleccionar(train, val_cambiada)
        if objetivo == "riesgo" and isinstance(cambiado, p2.Pipeline):
            imputador = (
                cambiado.named_steps["variables"]
                .named_transformers_["numericas"]
                .named_steps["simpleimputer"]
            )
            np.testing.assert_allclose(imputador.statistics_, train[p2.NUMERICAS].median())
    assert "test" not in inspect.signature(p1.seleccionar_demanda).parameters
    assert "test" not in inspect.signature(p2.seleccionar_riesgo).parameters


def test_p3_escala_k_independientes_de_periodos_futuros(sinteticos):
    _, _, _, perfiles = sinteticos
    futuros = perfiles.copy()
    escala, matriz = p3.ajustar_escala(perfiles)
    _, etiquetas, tabla = p3.seleccionar_clusters(matriz, "kmeans")
    futuros["volumen_total"] *= 10000
    futuros["duracion_media_min"] = 9999
    escala.transform(p3.matriz_perfiles(futuros))
    otra_escala, otra = p3.ajustar_escala(perfiles)
    _, otras_etiquetas, otra_tabla = p3.seleccionar_clusters(otra, "kmeans")
    np.testing.assert_array_equal(matriz, otra)
    np.testing.assert_array_equal(etiquetas, otras_etiquetas)
    pd.testing.assert_frame_equal(tabla, otra_tabla)
    np.testing.assert_array_equal(
        escala.named_steps["standardscaler"].mean_,
        otra_escala.named_steps["standardscaler"].mean_,
    )


def test_parametros_features_predeterminados_y_periodos():
    assert (
        inspect.signature(features.flujo_por_franja).parameters["horas_franja"].default
        == HORAS_FRANJA
    )
    for funcion in (features.etiquetar_riesgo, features.preparar_datos_p2):
        assert inspect.signature(funcion).parameters["alpha"].default == ALFA_RIESGO
    viajes = pd.DataFrame(
        {
            "inicio": pd.to_datetime([FECHA_INICIO, INICIO_VALIDACION, INICIO_PRUEBA]),
            "origen_id": [1, 2, 3],
            "destino_id": [4, 5, 6],
        }
    )
    viajes["fin"] = viajes["inicio"] + pd.Timedelta(minutes=15)
    train = features.perfiles_estacion(viajes)
    pd.testing.assert_frame_equal(
        train, features.perfiles_estacion(viajes, desde=FECHA_INICIO, hasta=INICIO_VALIDACION)
    )
    assert set(train.estacion_id) == {1, 4}
    assert set(
        features.perfiles_estacion(
            viajes, desde=INICIO_VALIDACION, hasta=INICIO_PRUEBA
        ).estacion_id
    ) == {2, 5}
    assert set(
        features.perfiles_estacion(
            viajes, desde=INICIO_PRUEBA, hasta=pd.Timestamp(FECHA_FIN) + pd.Timedelta(days=1)
        ).estacion_id
    ) == {3, 6}
    pd.testing.assert_frame_equal(
        features.flujo_por_franja(viajes),
        features.flujo_por_franja(viajes, horas_franja=HORAS_FRANJA),
    )
    por_hora = features.flujo_por_franja(viajes.iloc[:1], horas_franja=1)
    assert por_hora["salidas"].sum() == 1


def test_metricas_poisson_recorta_cero():
    metricas = p1.metricas_demanda([0, 1], [0, 0])
    assert metricas["mae"] == 0.5
    assert np.isfinite(metricas["desviacion_poisson"])


def test_cli_carga_una_vez_y_construye_tres_preguntas(monkeypatch, sinteticos):
    from typer.testing import CliRunner

    from mibici.modeling import train

    estaciones = sinteticos[0]
    inicio = pd.to_datetime([FECHA_INICIO, INICIO_VALIDACION, INICIO_PRUEBA])
    viajes = pd.DataFrame(
        {
            "viaje_id": [1, 2, 3],
            "inicio": inicio,
            "fin": inicio + pd.Timedelta(minutes=10),
            "origen_id": [0, 0, 0],
            "destino_id": [1, 1, 1],
        }
    )
    cargas = []
    preguntas = []
    monkeypatch.setattr(train, "cargar_estaciones", lambda: estaciones)

    def cargar():
        cargas.append(1)
        return viajes

    monkeypatch.setattr(train, "cargar_viajes", cargar)
    monkeypatch.setattr(train, "preparar_datos_p1", lambda *args: sinteticos[1])
    monkeypatch.setattr(train, "entrenar_p1", lambda *args: preguntas.append("P1"))
    monkeypatch.setattr(train, "entrenar_p2", lambda *args: preguntas.append("P2"))
    monkeypatch.setattr(train, "entrenar_p3", lambda *args: preguntas.append("P3"))
    resultado = CliRunner().invoke(train.app, ["--pregunta", "ToDaS"])
    assert resultado.exit_code == 0, resultado.output
    assert cargas == [1]
    assert preguntas == ["P1", "P2", "P3"]
    assert CliRunner().invoke(train.app, ["--pregunta", "P4"]).exit_code != 0
