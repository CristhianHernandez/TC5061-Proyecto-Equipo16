"""Estructura del stack de Docker Compose: servicios, orden de arranque, persistencia y configuración.

Prueba estática (no necesita Docker): protege en CI las decisiones que hacen que el stack arranque
con una sola orden. La prueba de extremo a extremo está en scripts/smoke_compose.sh.
"""

from pathlib import Path
import re

import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[1]
COMPOSE = yaml.safe_load((RAIZ / "docker-compose.yml").read_text(encoding="utf-8"))
SERVICIOS = COMPOSE["services"]


def comando(servicio: str) -> str:
    return " ".join(SERVICIOS[servicio]["command"])


def test_tres_servicios_de_una_misma_imagen_construida_del_containerfile():
    assert set(SERVICIOS) == {"mlflow", "model-init", "api"}
    for servicio in SERVICIOS.values():
        assert servicio["build"] == {"context": ".", "dockerfile": "Containerfile"}
        assert servicio["image"].startswith("mibici-ops:")


def test_orden_de_arranque_por_salud_y_finalizacion():
    assert SERVICIOS["model-init"]["depends_on"] == {"mlflow": {"condition": "service_healthy"}}
    assert SERVICIOS["api"]["depends_on"] == {
        "mlflow": {"condition": "service_healthy"},
        "model-init": {"condition": "service_completed_successfully"},
    }
    assert SERVICIOS["model-init"]["restart"] == "no"
    for nombre in ("mlflow", "api"):
        assert "/health" in " ".join(SERVICIOS[nombre]["healthcheck"]["test"])


def test_mlflow_sirve_artefactos_persistentes_y_acepta_el_host_interno():
    texto = comando("mlflow")
    assert "--serve-artifacts" in texto and "--artifacts-destination=/mlflow/" in texto
    assert "sqlite:////mlflow/" in texto
    hosts = re.search(r"--allowed-hosts=(\S+)", texto).group(1).split(",")
    assert "mlflow:5000" in hosts
    assert SERVICIOS["mlflow"]["volumes"] == ["mlflow-datos:/mlflow"]
    assert "mlflow-datos" in COMPOSE["volumes"]
    assert SERVICIOS["mlflow"]["environment"]["MLFLOW_SERVER_ENABLE_JOB_EXECUTION"] == "false"


def test_la_api_carga_el_modelo_desde_el_mlflow_del_stack():
    entorno = SERVICIOS["api"]["environment"]
    assert entorno["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
    assert entorno["MODEL_PATH"] != "/app/model_bundle"  # sin copia local: obliga al registry
    assert "mibici-equipo16-p1-demanda" in entorno["MODEL_URI"]
    assert SERVICIOS["model-init"]["environment"]["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"


def test_red_propia_y_limites_de_recursos():
    assert "mibici" in COMPOSE["networks"]
    for nombre, servicio in SERVICIOS.items():
        assert servicio["networks"] == ["mibici"], nombre
    for nombre in ("mlflow", "api"):
        limites = SERVICIOS[nombre]["deploy"]["resources"]["limits"]
        assert {"cpus", "memory"} <= set(limites)


def test_cada_variable_del_compose_esta_documentada_en_env_example():
    texto = (RAIZ / "docker-compose.yml").read_text(encoding="utf-8")
    variables = set(re.findall(r"\$\{(\w+):-", texto))
    ejemplo = (RAIZ / ".env.example").read_text(encoding="utf-8")
    faltantes = sorted(
        v for v in variables if not re.search(rf"^#?\s*{v}=", ejemplo, re.MULTILINE)
    )
    assert variables and faltantes == []


def test_sin_credenciales_en_el_compose():
    texto = (RAIZ / "docker-compose.yml").read_text(encoding="utf-8").lower()
    for palabra in ("password", "secret", "token", "api_key"):
        assert palabra not in texto


@pytest.mark.parametrize(
    "fragmento",
    ["COPY model_bundl[e] ./model_bundle/", "mkdir -p /mlflow", "USER appuser"],
)
def test_el_containerfile_construye_sin_copia_del_modelo(fragmento):
    assert fragmento in (RAIZ / "Containerfile").read_text(encoding="utf-8")
