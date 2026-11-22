"""Pruebas del cálculo FinOps (scripts/estimate_inference_cost.py) sin Docker ni red."""

import copy
import json
from pathlib import Path
import sys

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))

import estimate_inference_cost as fin

TARIFAS = json.loads((RAIZ / "references" / "finops_tarifas.json").read_text(encoding="utf-8"))

# Catálogo mínimo con números redondos para verificar la aritmética a mano
TARIFAS_SIMPLES = {
    "horas_mes": 1000,
    "otros_costos": {
        "disco_gb": 10,
        "disco_gb_mes": 0.1,  # $1/mes
        "registro_imagen_gb": 0,
        "registro_imagen_gb_mes": 0,
        "transferencia_salida_gb": 0.1,
        "bytes_respuesta": 1000,
    },
    "credito_cpu_excedente_vcpu_hora": {"arm64": 0.05},
}
VM_FIJA = {
    "sku": "fija",
    "arquitectura": "arm64",
    "vcpu": 2,
    "memoria_gib": 4,
    "precio_hora": 3.6,
    "burstable": False,
}
VM_BURST = {**VM_FIJA, "sku": "burst", "burstable": True, "linea_base_por_vcpu": 0.2}


def test_costo_por_inferencia_caso_conocido():
    # 1 h a $3.60 = $3.60; con 3,600 predicciones exitosas -> $0.001 c/u y $1 por 1,000
    assert fin.costo_por_inferencia(3.6, 3600) == pytest.approx(0.001)
    assert fin.por_mil(fin.costo_por_inferencia(3.6, 3600)) == pytest.approx(1.0)


def test_cero_inferencias_exitosas_no_divide_entre_cero():
    assert fin.costo_por_inferencia(3.6, 0) is None
    assert fin.por_mil(None) is None
    assert fin.nucleo_s_por_inferencia(50.0, 180, 0) is None


def test_inferencias_negativas_se_rechazan():
    with pytest.raises(ValueError):
        fin.costo_por_inferencia(1.0, -1)


def test_costo_ventana_desglose_a_mano():
    # 180 s = 0.05 h. Cómputo 0.05 x 3.6 = 0.18; otros 0.05 x ($1 / 1000 h) = 0.00005;
    # transferencia 1,000 respuestas x 1 KB = 0.001 GB x $0.1 = 0.0001
    c = fin.costo_ventana(180, 1000, 0.1, VM_FIJA, TARIFAS_SIMPLES)
    assert c["horas_facturables"] == pytest.approx(0.05)
    assert c["computo_usd"] == pytest.approx(0.18)
    assert c["otros_usd"] == pytest.approx(0.00005)
    assert c["transferencia_usd"] == pytest.approx(0.0001)
    assert c["creditos_cpu_usd"] == 0
    assert c["costo_ventana_usd"] == pytest.approx(0.18015)


def test_creditos_solo_sobre_la_linea_base():
    # Línea base 2 x 0.2 = 0.4 núcleos. Bajo la línea base no hay cargo
    assert fin.creditos_excedentes(VM_BURST, 0.3, 10, TARIFAS_SIMPLES) == 0
    # 0.6 núcleos -> 0.2 excedentes x 10 h x $0.05 = $0.10
    assert fin.creditos_excedentes(VM_BURST, 0.6, 10, TARIFAS_SIMPLES) == pytest.approx(0.10)
    # Una instancia no burstable nunca cobra créditos
    assert fin.creditos_excedentes(VM_FIJA, 1.9, 10, TARIFAS_SIMPLES) == 0


def test_duracion_invalida():
    with pytest.raises(ValueError):
        fin.costo_ventana(0, 10, 0.1, VM_FIJA, TARIFAS_SIMPLES)


@pytest.mark.parametrize(
    ("pico", "capacidad", "util", "esperado"),
    [(5, 58.9, 0.7, 1), (60, 58.9, 0.7, 2), (60, 114.9, 0.7, 1), (0, 10, 0.7, 1)],
)
def test_instancias_necesarias(pico, capacidad, util, esperado):
    assert fin.instancias_necesarias(pico, capacidad, util) == esperado


def test_tarifas_tienen_procedencia_y_referencia_valida():
    for campo in ("proveedor", "region", "moneda", "fecha_consulta", "fuentes"):
        assert TARIFAS[campo]
    assert fin.instancia(TARIFAS, TARIFAS["instancia_referencia"])["precio_hora"] > 0
    with pytest.raises(KeyError):
        fin.instancia(TARIFAS, "no-existe")


def test_evidencia_real_se_concilia():
    """Los conteos de la ventana usados para el costo cuadran con ventana.json de Locust."""
    manifiestos = fin.cargar_manifiestos()
    nombres = [m["escenario"] for m in manifiestos]
    assert {"u05", "u20"} <= set(nombres)
    for m in manifiestos:
        ventana = json.loads(
            (fin.CARGA / m["escenario"] / "ventana.json").read_text(encoding="utf-8")
        )
        res = m["resultados_ventana"]
        assert ventana["predicciones_exitosas"] == res["predicciones_exitosas"]
        assert ventana["peticiones"] == res["peticiones_totales"]
        assert m["carga"]["registros_por_peticion"] == 1


def test_manifiesto_inconsistente_se_rechaza(tmp_path):
    original = json.loads((fin.CARGA / "u05" / "manifiesto.json").read_text(encoding="utf-8"))
    roto = copy.deepcopy(original)
    roto["resultados_ventana"]["peticiones_fallidas"] = 5  # ya no suman
    (tmp_path / "u05").mkdir()
    (tmp_path / "u05" / "manifiesto.json").write_text(json.dumps(roto), encoding="utf-8")
    with pytest.raises(ValueError, match="fallidas"):
        fin.cargar_manifiestos(tmp_path)


def test_tablas_con_la_evidencia_real():
    manifiestos = fin.cargar_manifiestos()
    ventanas = {v["escenario"]: v for v in fin.tabla_ventanas(manifiestos, TARIFAS)}
    # Con tarifa fija por hora, más inferencias en la misma ventana bajan el costo unitario
    assert ventanas["u20"]["costo_por_1000_usd"] < ventanas["u05"]["costo_por_1000_usd"]
    # OMP_NUM_THREADS=1 reduce la CPU por inferencia frente a la misma carga sin fijar hilos
    assert (
        ventanas["u120_omp1"]["nucleo_s_por_inferencia"]
        < 0.6 * ventanas["u120"]["nucleo_s_por_inferencia"]
    )
    capacidad = fin.tabla_capacidad(manifiestos, TARIFAS)
    base = next(c for c in capacidad if c["config_api"] == "1 worker, 2 CPU")
    assert base["escenario_capacidad"] == "u60"  # u120 no cumple el SLO de p95
    assert not base["es_cota_inferior"]
    proyeccion = fin.tabla_proyeccion(capacidad, TARIFAS)
    lote = [p for p in proyeccion if p["demanda"].startswith("P1 real")]
    assert min(lote, key=lambda p: p["costo_mes_usd"])["opcion"].startswith("Lote")


def test_main_escribe_salidas(tmp_path):
    fin.main(["--salida", str(tmp_path), "--sin-graficas"])
    for nombre in (
        "costo_ventana.csv",
        "sensibilidad_instancias.csv",
        "capacidad_slo.csv",
        "proyeccion_mensual.csv",
        "finops_resumen.json",
        "resumen.md",
    ):
        assert (tmp_path / nombre).stat().st_size > 0
