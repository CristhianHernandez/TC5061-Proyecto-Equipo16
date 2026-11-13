"""A/B offline de P1: métricas, bootstrap por días, regla de decisión y política de CT.

Usa predicciones sintéticas y un registry falso; no descarga datos ni requiere MLflow.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mibici.monitoring import ab_compare as ab
from mibici.monitoring.datasets import COLUMNAS, OBJETIVO, VARIABLES


def evaluacion_sintetica(dias=14, estaciones=(1, 2, 3, 4, 9), semilla=16):
    """Ventana tipo octubre: estaciones 1–4 conocidas y la 9 nueva, una fila por estación y día."""
    azar = np.random.default_rng(semilla)
    fechas = pd.date_range("2026-10-01", periods=dias)
    filas = []
    for fecha in fechas:
        for estacion in estaciones:
            demanda = float(azar.poisson(40 if estacion < 5 else 10))
            filas.append(
                {
                    "estacion_id": estacion,
                    "fecha": fecha,
                    "dia_semana": fecha.dayofweek,
                    "mes": fecha.month,
                    "es_fin_semana": int(fecha.dayofweek >= 5),
                    "dia_anio": fecha.dayofyear,
                    "es_festivo": 0,
                    "demanda_lag_1": demanda,
                    "demanda_lag_7": demanda,
                    "demanda_media_7": demanda,
                    "capacidad": 15,
                    "latitud": 20.67,
                    "longitud": -103.35,
                    OBJETIVO: demanda,
                }
            )
    return pd.DataFrame(filas)[COLUMNAS]


def referencia_sintetica():
    """Referencia con las estaciones 1–4: 1 y 2 de demanda alta, 3 y 4 de demanda baja."""
    return pd.DataFrame(
        {"estacion_id": [1, 2, 3, 4] * 3, OBJETIVO: [60, 50, 10, 5] * 3, "fecha": "2025-01-01"}
    )


@pytest.fixture
def datos():
    evaluacion = evaluacion_sintetica()
    real = evaluacion[OBJETIVO].to_numpy(dtype=float)
    azar = np.random.default_rng(3)
    ruido = azar.normal(0, 1, len(real))
    pred_a = real + 4 + ruido  # A sobreestima 4 viajes
    pred_b = real + 2 + ruido  # B sobreestima 2: mejor en todas las filas
    return evaluacion, real, pred_a, pred_b


# --- métricas ---------------------------------------------------------------------------------


def test_metricas_de_un_caso_conocido():
    m = ab.metricas([10, 20, 30], [12, 18, 33])
    assert m["mae"] == pytest.approx(7 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert m["sesgo"] == pytest.approx(1.0)
    assert m["mae_relativo_pct"] == pytest.approx(100 * (7 / 3) / 20)
    assert m["n"] == 3


def test_desviacion_poisson_coincide_con_scikit_learn():
    from sklearn.metrics import mean_poisson_deviance

    real = np.array([0, 3, 10, 25], dtype=float)
    pred = np.array([0.5, 2.0, 12.0, 20.0])
    assert ab.metricas(real, pred)["desviacion_poisson"] == pytest.approx(
        mean_poisson_deviance(real, pred)
    )


def test_diferencia_negativa_indica_que_b_tiene_menos_error():
    assert ab.diferencia(10.0, 9.0) == {"absoluta": -1.0, "relativa_pct": -10.0}


# --- segmentos y tablas -----------------------------------------------------------------------


def test_segmentos_separan_estaciones_nuevas_y_demanda_alta(datos):
    evaluacion = datos[0]
    mascaras = ab.segmentos(evaluacion, referencia_sintetica())
    nuevas = evaluacion.loc[mascaras["estaciones_nuevas"], "estacion_id"].unique()
    altas = evaluacion.loc[mascaras["conocidas_demanda_alta"], "estacion_id"].unique()
    assert list(nuevas) == [9]
    assert sorted(altas) == [1, 2]
    assert (mascaras["estaciones_conocidas"] | mascaras["estaciones_nuevas"]).all()
    assert not (mascaras["laborable"] & mascaras["fin_de_semana"]).any()


def test_tabla_diaria_tiene_un_renglon_por_dia_y_promedia_errores(datos):
    evaluacion, real, pred_a, pred_b = datos
    diaria = ab.tabla_diaria(evaluacion, pred_a, pred_b)
    assert len(diaria) == evaluacion["fecha"].nunique()
    assert (diaria["n"] == 5).all()
    # Con días del mismo tamaño, el promedio de los MAE diarios es el MAE global
    assert diaria["mae_a"].mean() == pytest.approx(np.abs(pred_a - real).mean())


# --- comparación pareada ----------------------------------------------------------------------


def test_bootstrap_detecta_una_mejora_consistente(datos):
    evaluacion, real, pred_a, pred_b = datos
    intervalo = ab.bootstrap_por_dias(
        evaluacion["fecha"], np.abs(pred_a - real), np.abs(pred_b - real), n=500
    )
    assert intervalo["delta_mae"] == pytest.approx(
        np.abs(pred_b - real).mean() - np.abs(pred_a - real).mean()
    )
    assert intervalo["ic_inferior"] <= intervalo["delta_mae"] <= intervalo["ic_superior"]
    assert intervalo["ic_superior"] < 0
    assert intervalo["prob_b_mejor"] == 1.0
    assert intervalo["dias"] == 14


def test_bootstrap_es_reproducible_con_la_misma_semilla(datos):
    evaluacion, real, pred_a, pred_b = datos
    args = (evaluacion["fecha"], np.abs(pred_a - real), np.abs(pred_b - real))
    uno, otro = ab.bootstrap_por_dias(*args, n=200), ab.bootstrap_por_dias(*args, n=200)
    assert uno["ic_inferior"] == otro["ic_inferior"]
    assert uno["ic_superior"] == otro["ic_superior"]


def test_bootstrap_incluye_el_cero_si_los_modelos_son_equivalentes(datos):
    evaluacion, real, _, _ = datos
    azar = np.random.default_rng(5)
    pred_a = real + azar.normal(0, 3, len(real))
    pred_b = real + azar.normal(0, 3, len(real))
    intervalo = ab.bootstrap_por_dias(
        evaluacion["fecha"], np.abs(pred_a - real), np.abs(pred_b - real), n=500
    )
    assert intervalo["ic_inferior"] < 0 < intervalo["ic_superior"]


def test_comparar_exige_las_mismas_filas_para_a_y_b(datos):
    evaluacion, _, pred_a, pred_b = datos
    with pytest.raises(ValueError, match="mismas|exactamente"):
        ab.comparar(evaluacion, referencia_sintetica(), pred_a, pred_b[:-1], n_bootstrap=50)


def test_comparar_promueve_un_challenger_claramente_mejor(datos):
    evaluacion, _, pred_a, pred_b = datos
    resultado = ab.comparar(evaluacion, referencia_sintetica(), pred_a, pred_b, n_bootstrap=300)
    assert resultado["decision"]["decision"] == "promover_challenger"
    assert resultado["pareado"]["dias_b_mejor"] == resultado["pareado"]["dias"]
    assert set(resultado["por_segmento"]["segmento"]) >= {"todas", "estaciones_nuevas"}


# --- regla de decisión ------------------------------------------------------------------------


def _segmentos(**cambios):
    filas = [{"segmento": "todas", "delta_mae_pct": -2.0}]
    filas += [{"segmento": s, "delta_mae_pct": v} for s, v in cambios.items()]
    return pd.DataFrame(filas)


def _intervalo(inferior, superior):
    return {"ic_inferior": inferior, "ic_superior": superior, "nivel": 0.95}


def test_decision_conserva_champion_si_b_empeora():
    decision = ab.decidir(
        {"mae": 8.0, "rmse": 17.0}, {"mae": 8.4, "rmse": 17.5}, _intervalo(0.2, 0.6), _segmentos()
    )
    assert decision["decision"] == "conservar_champion"


def test_decision_pide_mas_datos_si_el_intervalo_incluye_cero():
    decision = ab.decidir(
        {"mae": 8.0, "rmse": 17.0},
        {"mae": 7.95, "rmse": 16.9},
        _intervalo(-0.2, 0.1),
        _segmentos(),
    )
    assert decision["decision"] == "recoger_mas_datos"


def test_decision_no_promueve_si_un_segmento_empeora_mas_de_la_tolerancia():
    decision = ab.decidir(
        {"mae": 8.0, "rmse": 17.0},
        {"mae": 7.8, "rmse": 16.5},
        _intervalo(-0.3, -0.1),
        _segmentos(estaciones_nuevas=5.0),
    )
    assert decision["decision"] == "conservar_champion"
    assert "estaciones_nuevas" in decision["guardas_fallidas"][0]


def test_decision_no_promueve_si_el_rmse_empeora():
    decision = ab.decidir(
        {"mae": 8.0, "rmse": 17.0},
        {"mae": 7.8, "rmse": 17.4},
        _intervalo(-0.3, -0.1),
        _segmentos(),
    )
    assert decision["decision"] == "conservar_champion"


# --- Continuous Training ----------------------------------------------------------------------


def _drift(objetivo=0.3, historial=3, nuevas_pct=21.0, duplicadas=0):
    return {
        "calidad": {
            "actual": {
                "columnas_faltantes": [],
                "filas_duplicadas": duplicadas,
                "claves_duplicadas": 0,
                "demanda_negativa": 0,
            }
        },
        "comparaciones": {
            "referencia_vs_actual": {
                "objetivo": {"puntaje": objetivo},
                "drift_por_grupo": {"historial de demanda": ["demanda_lag_1"] * historial},
            }
        },
        "estaciones_nuevas": {"actual": {"filas_estaciones_nuevas_pct": nuevas_pct}},
    }


def _m(mae, media):
    return {"mae": mae, "mae_relativo_pct": 100 * mae / media, "media_real": media}


ESTABLE = (_m(8.1, 43.7), _m(5.8, 30.0))  # MAE absoluto +40 %, relativo casi igual


def test_la_degradacion_se_mide_con_el_mae_relativo():
    politica = ab.politica_continuous_training(
        _drift(nuevas_pct=0.0), *ESTABLE, "conservar_champion"
    )
    degradacion = politica["disparadores"][-1]
    assert not degradacion["activo"]  # más viajes no es un peor modelo
    assert politica["aumento_mae_absoluto_pct"] > 15
    assert politica["accion"] == "investigar_drift"
    peor = ab.politica_continuous_training(
        _drift(nuevas_pct=0.0), _m(8.1, 30.0), _m(5.8, 30.0), "x"
    )
    assert peor["disparadores"][-1]["activo"]
    assert peor["accion"] == "reentrenar_candidato"


def test_politica_reentrena_con_cambio_de_demanda_e_impacto():
    politica = ab.politica_continuous_training(_drift(), *ESTABLE, "promover_challenger")
    assert politica["accion"] == "reentrenar_candidato"
    assert politica["promocion"] == "promover_challenger"


def test_politica_no_entrena_con_datos_corruptos():
    politica = ab.politica_continuous_training(_drift(duplicadas=4), *ESTABLE, "x")
    assert politica["accion"] == "corregir_pipeline_de_datos"


def test_politica_solo_investiga_drift_sin_impacto():
    politica = ab.politica_continuous_training(_drift(nuevas_pct=0.0), *ESTABLE, "x")
    assert politica["accion"] == "investigar_drift"


def test_politica_sigue_monitoreando_sin_senales():
    politica = ab.politica_continuous_training(
        _drift(objetivo=0.02, historial=0, nuevas_pct=0.0), *ESTABLE, "x"
    )
    assert politica["accion"] == "seguir_monitoreando"


# --- registry y salida ------------------------------------------------------------------------


def test_resolver_version_fija_el_numero_de_un_alias():
    cliente = SimpleNamespace(
        get_model_version_by_alias=lambda nombre, alias: SimpleNamespace(
            version=3, run_id="r3", tags={"ventana_train": "2024-10-01/2026-09-30"}
        ),
        get_model_version=lambda nombre, numero: SimpleNamespace(
            version=numero, run_id="r2", tags={}
        ),
    )
    b = ab.resolver_version(cliente, "challenger", "B")
    assert (b.numero, b.run_id, b.referencia) == ("3", "r3", "challenger")
    assert b.uri.endswith("/3")
    a = ab.resolver_version(cliente, "2", "A")
    assert a.numero == "2" and a.uri.endswith("/2")


def test_predecir_rechaza_valores_no_finitos(datos):
    evaluacion = datos[0]
    modelo = SimpleNamespace(predict=lambda x: np.full(len(x), np.nan))
    with pytest.raises(ValueError, match="no finitas"):
        ab.predecir(modelo, evaluacion)


def test_predecir_usa_solo_las_variables_de_la_firma(datos):
    evaluacion = datos[0]
    vistas = []

    def predict(x):
        vistas.append(list(x.columns))
        return np.ones(len(x))

    ab.predecir(SimpleNamespace(predict=predict), evaluacion)
    assert vistas == [VARIABLES]


def test_guardar_escribe_tablas_graficas_y_resumen_sin_fecha(datos, tmp_path):
    evaluacion, _, pred_a, pred_b = datos
    resultado = ab.comparar(evaluacion, referencia_sintetica(), pred_a, pred_b, n_bootstrap=100)
    a = ab.Version("A", "2", "r2", "champion", "2024-10-01/2026-03-31")
    b = ab.Version("B", "3", "r3", "challenger", "2024-10-01/2026-09-30")
    resumen = ab.guardar(resultado, a, b, evaluacion, tmp_path)
    esperados = {
        "tabla_ab.csv",
        "ab_por_segmento.csv",
        "ab_por_dia.csv",
        "ab_por_estacion.csv",
        "resultados_ab.json",
        "grafica_ab_mae_diario.png",
        "grafica_ab_segmentos.png",
        "grafica_ab_bootstrap.png",
    }
    assert esperados <= {p.name for p in tmp_path.iterdir()}
    tabla = pd.read_csv(tmp_path / "tabla_ab.csv")
    assert list(tabla["version_mlflow"]) == [2, 3]
    assert tabla["n"].nunique() == 1  # mismo N para A y B
    assert tabla.loc[0, "delta_mae_vs_a"] == 0
    assert "_deltas" not in resumen["intervalo_delta_mae"]
    # Una segunda corrida con los mismos datos produce el mismo JSON (reproducible)
    primero = (tmp_path / "resultados_ab.json").read_text()
    ab.guardar(resultado, a, b, evaluacion, tmp_path)
    assert (tmp_path / "resultados_ab.json").read_text() == primero
