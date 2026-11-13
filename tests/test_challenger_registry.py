"""Pruebas del registro Challenger P2 sin alterar el Champion."""

from types import SimpleNamespace

import pytest

from mibici.monitoring import train_challenger


def preparar_prueba(monkeypatch, version_challenger="2", estado="READY",
                   champion_final="1"):
    llamadas = []
    consultas_champion = 0
    champion = SimpleNamespace(version="1")
    challenger = SimpleNamespace(version=version_challenger, status=estado)

    def obtener_alias(nombre, alias):
        nonlocal consultas_champion
        if alias == "champion":
            consultas_champion += 1
            version = "1" if consultas_champion == 1 else champion_final
            return SimpleNamespace(version=version)
        raise AssertionError(f"Alias inesperado: {alias}")

    def asignar_alias(nombre, alias, version):
        llamadas.append((nombre, alias, str(version)))

    cliente = SimpleNamespace(
        get_model_version_by_alias=obtener_alias,
        get_run=lambda run_id: SimpleNamespace(
            data=SimpleNamespace(tags={
                "model_uri": "runs:/test/model",
                "modelo_registrable": "true",
            })
        ),
        set_model_version_tag=lambda *args: None,
        set_registered_model_alias=asignar_alias,
    )

    monkeypatch.setattr(train_challenger, "MlflowClient", lambda: cliente)
    monkeypatch.setattr(train_challenger.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(train_challenger, "ejecutar", lambda **kwargs: "run-test")
    monkeypatch.setattr(
        train_challenger, "version_de_corrida",
        lambda nombre, run_id: challenger,
    )

    return llamadas


def test_registra_challenger_sin_modificar_champion(monkeypatch):
    llamadas = preparar_prueba(monkeypatch)
    train_challenger.main()

    assert len(llamadas) == 1
    assert llamadas[0][1:] == ("challenger", "2")


def test_rechaza_version_igual_al_champion(monkeypatch):
    llamadas = preparar_prueba(monkeypatch, version_challenger="1")

    with pytest.raises(RuntimeError, match="coincide con el Champion"):
        train_challenger.main()

    assert llamadas == []


def test_rechaza_challenger_no_listo(monkeypatch):
    llamadas = preparar_prueba(monkeypatch, estado="PENDING_REGISTRATION")

    with pytest.raises(RuntimeError, match="no esta lista"):
        train_challenger.main()

    assert llamadas == []


def test_rechaza_cambio_del_champion(monkeypatch):
    llamadas = preparar_prueba(monkeypatch, champion_final="3")

    with pytest.raises(RuntimeError, match="Champion cambio"):
        train_challenger.main()

    assert llamadas == []
