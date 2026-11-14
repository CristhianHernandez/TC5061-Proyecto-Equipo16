"""Challenger de P1: misma receta que el champion, entrenamiento sin octubre y registro seguro.

Usa conjuntos sintéticos y un cliente de MLflow falso; no descarga datos ni requiere servidor.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mibici.monitoring import challenger_p1 as ch
from mibici.monitoring.datasets import COLUMNAS, OBJETIVO


def ventana(inicio, fin, semilla, estaciones=(1, 2, 3)):
    azar = np.random.default_rng(semilla)
    filas = []
    for fecha in pd.date_range(inicio, fin, freq="7D"):
        for estacion in estaciones:
            lag = float(azar.poisson(30))
            filas.append(
                {
                    "estacion_id": estacion,
                    "fecha": fecha,
                    "dia_semana": fecha.dayofweek,
                    "mes": fecha.month,
                    "es_fin_semana": int(fecha.dayofweek >= 5),
                    "dia_anio": fecha.dayofyear,
                    "es_festivo": 0,
                    "demanda_lag_1": lag,
                    "demanda_lag_7": lag,
                    "demanda_media_7": lag,
                    "capacidad": 15,
                    "latitud": 20.6 + estacion / 100,
                    "longitud": -103.3,
                    OBJETIVO: float(azar.poisson(lag + 1)),
                }
            )
    return pd.DataFrame(filas)[COLUMNAS]


@pytest.fixture
def conjuntos():
    return {
        "referencia": ventana("2025-01-01", "2026-03-31", 1),
        "reciente": ventana("2026-04-01", "2026-09-30", 2, estaciones=(1, 2, 3, 4)),
        "actual": ventana("2026-10-01", "2026-10-31", 3, estaciones=(1, 2, 3, 4)),
    }


def test_la_receta_es_la_del_champion_version_2():
    modelo = ch.nuevo_modelo()
    assert modelo.loss == "poisson"
    assert (modelo.learning_rate, modelo.max_iter, modelo.max_leaf_nodes) == (0.05, 100, 63)
    assert modelo.early_stopping is False


def test_partir_reciente_separa_abr_jun_de_jul_sep(conjuntos):
    abr_jun, jul_sep = ch.partir_reciente(conjuntos["reciente"])
    assert pd.to_datetime(abr_jun["fecha"]).max() < pd.Timestamp("2026-07-01")
    assert pd.to_datetime(jul_sep["fecha"]).min() >= pd.Timestamp("2026-07-01")
    assert len(abr_jun) + len(jul_sep) == len(conjuntos["reciente"])


def test_entrenar_usa_referencia_y_reciente_pero_nunca_octubre(conjuntos):
    resultado = ch.entrenar(conjuntos)
    fechas = pd.to_datetime(resultado["entrenamiento"]["fecha"])
    assert fechas.max() < pd.Timestamp("2026-10-01")
    assert len(resultado["entrenamiento"]) == len(conjuntos["referencia"]) + len(
        conjuntos["reciente"]
    )
    assert resultado["params"]["ventana_train_fin"] <= "2026-09-30"
    assert all(k.startswith(("train_", "val_")) for k in resultado["metricas"])
    assert np.isfinite(list(resultado["metricas"].values())).all()


def test_entrenar_falla_si_la_ventana_reciente_se_traslapa_con_la_evaluacion(conjuntos):
    conjuntos["reciente"] = pd.concat([conjuntos["reciente"], conjuntos["actual"].head(3)])
    with pytest.raises(ValueError, match="Fuga temporal"):
        ch.entrenar(conjuntos)


def test_entrenar_es_determinista(conjuntos):
    uno, otro = ch.entrenar(conjuntos), ch.entrenar(conjuntos)
    x = conjuntos["actual"][ch.VARIABLES]
    np.testing.assert_array_equal(uno["modelo"].predict(x), otro["modelo"].predict(x))


def _cliente_falso(champion_final="2", version_nueva="3"):
    llamadas = {"alias": [], "tags": {}}
    consultas = {"n": 0}

    def por_alias(nombre, alias):
        consultas["n"] += 1
        return SimpleNamespace(version="2" if consultas["n"] == 1 else champion_final)

    cliente = SimpleNamespace(
        get_model_version_by_alias=por_alias,
        set_model_version_tag=lambda n, v, k, val: llamadas["tags"].__setitem__(k, val),
        set_registered_model_alias=lambda n, a, v: llamadas["alias"].append((a, str(v))),
    )
    return cliente, llamadas, SimpleNamespace(version=version_nueva)


@pytest.fixture
def mlflow_falso(monkeypatch):
    @contextmanager
    def run_falso(*args, **kwargs):
        yield SimpleNamespace(info=SimpleNamespace(run_id="run-ch"))

    monkeypatch.setattr(ch.tracking, "iniciar_run", run_falso)
    monkeypatch.setattr(ch.tracking, "registrar_metricas", lambda m: None)
    monkeypatch.setattr(ch.mlflow, "log_artifact", lambda *a, **k: None)
    monkeypatch.setattr(ch.mlflow, "log_input", lambda *a, **k: None)
    monkeypatch.setattr(ch, "registrar_modelo", lambda modelo, ejemplo: "runs:/run-ch/model")
    monkeypatch.setattr(ch, "version_de_corrida", lambda nombre, run_id: None)


def test_registrar_crea_challenger_sin_mover_el_champion(conjuntos, mlflow_falso, monkeypatch):
    cliente, llamadas, nueva = _cliente_falso()
    monkeypatch.setattr(ch.mlflow, "register_model", lambda uri, nombre: nueva)
    registro = ch.registrar(ch.entrenar(conjuntos), cliente)
    assert registro == {
        "nombre": ch.NOMBRE,
        "run_id": "run-ch",
        "champion": "2",
        "challenger": "3",
    }
    assert llamadas["alias"] == [("challenger", "3")]
    assert llamadas["tags"]["rol"] == "challenger"
    assert llamadas["tags"]["champion_version_base"] == "2"


def test_registrar_falla_si_el_champion_cambia(conjuntos, mlflow_falso, monkeypatch):
    cliente, _, nueva = _cliente_falso(champion_final="9")
    monkeypatch.setattr(ch.mlflow, "register_model", lambda uri, nombre: nueva)
    with pytest.raises(RuntimeError, match="champion cambió"):
        ch.registrar(ch.entrenar(conjuntos), cliente)


def test_registrar_rechaza_una_version_igual_al_champion(conjuntos, mlflow_falso, monkeypatch):
    cliente, llamadas, _ = _cliente_falso(version_nueva="2")
    monkeypatch.setattr(
        ch.mlflow, "register_model", lambda uri, nombre: SimpleNamespace(version="2")
    )
    with pytest.raises(RuntimeError, match="coincide con el champion"):
        ch.registrar(ch.entrenar(conjuntos), cliente)
    assert llamadas["alias"] == []
