"""Costo por inferencia de la API de P1 a partir de las pruebas de carga (Entregable 7 · FinOps).

    python scripts/estimate_inference_cost.py                 # o: make finops
    python scripts/estimate_inference_cost.py --sin-graficas  # solo tablas

Entradas (no se edita nada a mano en las salidas):
  docs/evidencias/operacion/carga/<escenario>/manifiesto.json  ventana, conteos, latencia y recursos
  references/finops_tarifas.json                               tarifas, supuestos y escenarios de demanda

Salidas en docs/evidencias/operacion/finops/:
  costo_ventana.csv           costo de cada ventana de 180 s y costo por inferencia exitosa
  sensibilidad_instancias.csv u05 y u20 con cada instancia del catálogo: si cabe y cuánto cuesta
  capacidad_slo.csv           peticiones/s máximas que cumplen el SLO por configuración de la API
  proyeccion_mensual.csv      costo mensual y por 1,000 inferencias según demanda y despliegue
  finops_resumen.json         todo lo anterior más las entradas usadas
  resumen.md                  las tablas en Markdown para el README y el reporte
  graficas/*.png              costo por escenario y proyección mensual

Fórmula acordada por el equipo:
  costo de la ventana = horas facturables x tarifa por hora + otros costos atribuibles de la ventana
  costo por inferencia = costo de la ventana / predicciones exitosas de esa misma ventana
Cada petición de Locust lleva un registro, así que predicciones exitosas = peticiones exitosas.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess

RAIZ = Path(__file__).resolve().parents[1]
CARGA = RAIZ / "docs" / "evidencias" / "operacion" / "carga"
SALIDA = RAIZ / "docs" / "evidencias" / "operacion" / "finops"
TARIFAS = RAIZ / "references" / "finops_tarifas.json"
ESCENARIOS_OFICIALES = ("u05", "u20")


# --------------------------------------------------------------------------------------------
# Aritmética (funciones puras, probadas en tests/test_finops.py)
# --------------------------------------------------------------------------------------------


def costo_por_inferencia(costo_usd: float, exitosas: int) -> float | None:
    """Costo de la ventana entre predicciones exitosas; None si no hubo ninguna (no definido)."""
    if exitosas < 0:
        raise ValueError("las predicciones exitosas no pueden ser negativas")
    if exitosas == 0:
        return None
    return costo_usd / exitosas


def por_mil(valor: float | None) -> float | None:
    """Convierte un costo unitario en costo por 1,000 inferencias (respeta None)."""
    return None if valor is None else valor * 1000


def otros_costos_hora(tarifas: dict) -> float:
    """Disco y registro de imagen prorrateados por hora de instancia."""
    o = tarifas["otros_costos"]
    mensual = (
        o["disco_gb"] * o["disco_gb_mes"] + o["registro_imagen_gb"] * o["registro_imagen_gb_mes"]
    )
    return mensual / tarifas["horas_mes"]


def costo_transferencia(peticiones: int, tarifas: dict) -> float:
    """Salida a internet de las respuestas (GB decimales, como factura AWS)."""
    o = tarifas["otros_costos"]
    return peticiones * o["bytes_respuesta"] / 1e9 * o["transferencia_salida_gb"]


def creditos_excedentes(
    instancia: dict, nucleos_prom: float, horas: float, tarifas: dict
) -> float:
    """Créditos de CPU que cobra una instancia burstable (modo Unlimited) sobre su línea base."""
    if not instancia.get("burstable"):
        return 0.0
    linea_base = instancia["vcpu"] * instancia["linea_base_por_vcpu"]
    tarifa = tarifas["credito_cpu_excedente_vcpu_hora"][instancia["arquitectura"]]
    return max(0.0, nucleos_prom - linea_base) * horas * tarifa


def costo_ventana(
    duracion_s: float, peticiones: int, nucleos_prom: float, instancia: dict, tarifas: dict
) -> dict:
    """Desglose del costo de una ventana medida en una instancia de la nube."""
    if duracion_s <= 0:
        raise ValueError("la duración de la ventana debe ser positiva")
    horas = duracion_s / 3600
    partes = {
        "computo_usd": horas * instancia["precio_hora"],
        "creditos_cpu_usd": creditos_excedentes(instancia, nucleos_prom, horas, tarifas),
        "otros_usd": horas * otros_costos_hora(tarifas),
        "transferencia_usd": costo_transferencia(peticiones, tarifas),
    }
    return {"horas_facturables": horas, **partes, "costo_ventana_usd": sum(partes.values())}


def nucleo_s_por_inferencia(cpu_pct_prom: float, duracion_s: float, exitosas: int) -> float | None:
    """Segundos de CPU (100 % = 1 núcleo) que consume la API por predicción exitosa."""
    if exitosas == 0:
        return None
    return cpu_pct_prom / 100 * duracion_s / exitosas


def requisitos(m: dict, tarifas: dict) -> dict:
    """vCPU y memoria que necesita el stack según lo medido, con la holgura de la configuración."""
    d = tarifas["dimensionamiento"]
    r = m["recursos_ventana"]
    mem_mib = (
        r["api"]["mem_mib_max"] + r["mlflow"]["mem_mib_max"] + d["memoria_sistema_mib"]
    ) * d["holgura_memoria"]
    cpu = (r["api"]["cpu_pct_p95"] + r["mlflow"]["cpu_pct_p95"]) / 100 / d["utilizacion_cpu_max"]
    return {"memoria_gib": mem_mib / 1024, "vcpu": cpu}


def cabe(instancia: dict, req: dict) -> bool:
    return instancia["memoria_gib"] >= req["memoria_gib"] and instancia["vcpu"] >= req["vcpu"]


def cumple_slo(m: dict, slo: dict) -> bool:
    res = m["resultados_ventana"]
    return (
        res["latencia_ms"]["p95"] <= slo["p95_ms_max"]
        and res["tasa_error_pct"] <= slo["error_pct_max"]
    )


def instancias_necesarias(pico_rps: float, capacidad_rps: float, utilizacion: float) -> int:
    """Instancias para atender el pico sin pasar la utilización objetivo (mínimo 1)."""
    if capacidad_rps <= 0:
        raise ValueError("la capacidad debe ser positiva")
    return max(1, math.ceil(pico_rps / (capacidad_rps * utilizacion)))


# --------------------------------------------------------------------------------------------
# Lectura de evidencia
# --------------------------------------------------------------------------------------------


def config_api(m: dict) -> tuple[int, float, str]:
    c = m["configuracion_api"]
    return (
        int(c.get("api_workers", 1)),
        float(c.get("api_cpus_limite") or 2.0),
        str(c.get("omp_num_threads") or "sin fijar"),
    )


def etiqueta_config(m: dict) -> str:
    workers, cpus, omp = config_api(m)
    texto = f"{workers} worker{'s' if workers > 1 else ''}, {cpus:g} CPU"
    return texto + (f", OMP={omp}" if omp != "sin fijar" else "")


def cargar_manifiestos(carpeta: Path = CARGA) -> list[dict]:
    """Manifiestos de todos los escenarios, ordenados por usuarios y configuración."""
    manifiestos = []
    for ruta in sorted(carpeta.glob("*/manifiesto.json")):
        m = json.loads(ruta.read_text(encoding="utf-8"))
        res = m["resultados_ventana"]
        if res["peticiones_totales"] != res["peticiones_fallidas"] + res["predicciones_exitosas"]:
            raise ValueError(
                f"{ruta}: peticiones != fallidas + exitosas (se esperaba 1 registro por petición)"
            )
        manifiestos.append(m)
    if not manifiestos:
        raise FileNotFoundError(f"no hay manifiestos en {carpeta}")

    def orden(m):
        workers, cpus, omp = config_api(m)
        return (m["carga"]["usuarios"], omp != "sin fijar", workers, cpus)

    return sorted(manifiestos, key=orden)


def instancia(tarifas: dict, sku: str) -> dict:
    for inst in tarifas["instancias"]:
        if inst["sku"] == sku:
            return inst
    raise KeyError(f"la instancia {sku} no está en el catálogo de tarifas")


# --------------------------------------------------------------------------------------------
# Tablas
# --------------------------------------------------------------------------------------------


def tabla_ventanas(manifiestos: list[dict], tarifas: dict) -> list[dict]:
    """Costo de cada ventana medida en la instancia de referencia."""
    ref = instancia(tarifas, tarifas["instancia_referencia"])
    precio_vcpu = ref["precio_hora"] / ref["vcpu"]
    filas = []
    for m in manifiestos:
        v, res, rec = m["ventana"], m["resultados_ventana"], m["recursos_ventana"]
        exitosas = res["predicciones_exitosas"]
        nucleos = (rec["api"]["cpu_pct_prom"] + rec["mlflow"]["cpu_pct_prom"]) / 100
        costo = costo_ventana(v["duracion_s"], res["peticiones_totales"], nucleos, ref, tarifas)
        unitario = costo_por_inferencia(costo["costo_ventana_usd"], exitosas)
        nucleo_s = nucleo_s_por_inferencia(rec["api"]["cpu_pct_prom"], v["duracion_s"], exitosas)
        filas.append(
            {
                "escenario": m["escenario"],
                "usuarios": m["carga"]["usuarios"],
                "config_api": etiqueta_config(m),
                "oficial": m["escenario"] in ESCENARIOS_OFICIALES,
                "inicio_ventana_utc": v["inicio_utc"],
                "duracion_s": v["duracion_s"],
                "peticiones": res["peticiones_totales"],
                "fallidas": res["peticiones_fallidas"],
                "predicciones_exitosas": exitosas,
                "rps": res["rps_prom"],
                "p95_ms": res["latencia_ms"]["p95"],
                "cumple_slo": cumple_slo(m, tarifas["slo"]),
                "instancia": ref["sku"],
                **{k: round(val, 8) for k, val in costo.items()},
                "costo_por_inferencia_usd": unitario,
                "costo_por_1000_usd": por_mil(unitario),
                "nucleo_s_por_inferencia": nucleo_s,
                "costo_cpu_por_1000_usd": por_mil(
                    None if nucleo_s is None else nucleo_s / 3600 * precio_vcpu
                ),
            }
        )
    return filas


def tabla_sensibilidad(manifiestos: list[dict], tarifas: dict) -> list[dict]:
    """Escenarios oficiales con cada instancia: si cabe con lo medido y costo por 1,000."""
    filas = []
    for m in manifiestos:
        if m["escenario"] not in ESCENARIOS_OFICIALES:
            continue
        req = requisitos(m, tarifas)
        rec, res = m["recursos_ventana"], m["resultados_ventana"]
        nucleos = (rec["api"]["cpu_pct_prom"] + rec["mlflow"]["cpu_pct_prom"]) / 100
        for inst in tarifas["instancias"]:
            costo = costo_ventana(
                m["ventana"]["duracion_s"], res["peticiones_totales"], nucleos, inst, tarifas
            )
            unitario = costo_por_inferencia(
                costo["costo_ventana_usd"], res["predicciones_exitosas"]
            )
            filas.append(
                {
                    "escenario": m["escenario"],
                    "instancia": inst["sku"],
                    "arquitectura": inst["arquitectura"],
                    "vcpu": inst["vcpu"],
                    "memoria_gib": inst["memoria_gib"],
                    "precio_hora_usd": inst["precio_hora"],
                    "memoria_requerida_gib": round(req["memoria_gib"], 2),
                    "vcpu_requeridas": round(req["vcpu"], 2),
                    "cabe_con_lo_medido": cabe(inst, req),
                    "respeta_limites_compose": inst["memoria_gib"] >= 3 and inst["vcpu"] >= 2,
                    "creditos_cpu_usd": round(costo["creditos_cpu_usd"], 8),
                    "costo_por_1000_usd": por_mil(unitario),
                }
            )
    return filas


def tabla_capacidad(manifiestos: list[dict], tarifas: dict) -> list[dict]:
    """Por configuración de la API: mayor tasa medida que cumple el SLO y su CPU por inferencia."""
    grupos: dict[tuple, list[dict]] = {}
    for m in manifiestos:
        grupos.setdefault(config_api(m), []).append(m)
    filas = []
    for clave, ms in grupos.items():
        ok = [m for m in ms if cumple_slo(m, tarifas["slo"])]
        falla = [m for m in ms if not cumple_slo(m, tarifas["slo"])]
        mejor = max(ok, key=lambda m: m["resultados_ventana"]["rps_prom"]) if ok else None
        rec = mejor["recursos_ventana"] if mejor else None
        filas.append(
            {
                "config_api": etiqueta_config(ms[0]),
                "api_workers": clave[0],
                "api_cpus": clave[1],
                "omp_num_threads": clave[2],
                "escenarios": " ".join(m["escenario"] for m in ms),
                "escenario_capacidad": mejor["escenario"] if mejor else None,
                "capacidad_rps": mejor["resultados_ventana"]["rps_prom"] if mejor else None,
                "p95_ms": mejor["resultados_ventana"]["latencia_ms"]["p95"] if mejor else None,
                "nucleo_s_por_inferencia": nucleo_s_por_inferencia(
                    rec["api"]["cpu_pct_prom"],
                    mejor["ventana"]["duracion_s"],
                    mejor["resultados_ventana"]["predicciones_exitosas"],
                )
                if mejor
                else None,
                "nucleos_reposo_mlflow": rec["mlflow"]["cpu_pct_prom"] / 100 if mejor else None,
                "memoria_max_mib": (rec["api"]["mem_mib_max"] + rec["mlflow"]["mem_mib_max"])
                if mejor
                else None,
                # Si ningún escenario mayor del grupo falló, la capacidad real puede ser más alta
                "es_cota_inferior": mejor is not None
                and not any(f["carga"]["usuarios"] > mejor["carga"]["usuarios"] for f in falla),
            }
        )
    return filas


def _opcion_vm(nombre, esc, inst, cap, tarifas) -> dict:
    horas = tarifas["horas_mes"]
    util = tarifas["slo"]["utilizacion_objetivo"]
    n = instancias_necesarias(esc["pico_rps"], cap["capacidad_rps"], util)
    rps_prom_instancia = esc["predicciones_mes"] / (horas * 3600) / n
    nucleos = rps_prom_instancia * cap["nucleo_s_por_inferencia"] + cap["nucleos_reposo_mlflow"]
    computo = n * horas * inst["precio_hora"]
    creditos = n * creditos_excedentes(inst, nucleos, horas, tarifas)
    otros = n * horas * otros_costos_hora(tarifas) + costo_transferencia(
        esc["predicciones_mes"], tarifas
    )
    total = computo + creditos + otros
    return {
        "opcion": nombre,
        "instancia": inst["sku"],
        "instancias": n,
        "horas_facturadas": n * horas,
        "computo_usd": computo,
        "creditos_cpu_usd": creditos,
        "otros_usd": otros,
        "costo_mes_usd": total,
        "utilizacion_pct": 100
        * esc["predicciones_mes"]
        / (n * cap["capacidad_rps"] * horas * 3600),
    }


def _opcion_lote(esc, cap, tarifas) -> dict:
    f = tarifas["fargate_arm"]
    rps = f["vcpu"] * tarifas["slo"]["utilizacion_objetivo"] / cap["nucleo_s_por_inferencia"]
    horas = (
        esc["ejecuciones_mes"] * f["minutos_arranque_por_ejecucion"] / 60
        + esc["predicciones_mes"] / rps / 3600
    )
    computo = horas * (f["vcpu"] * f["vcpu_hora"] + f["memoria_gb"] * f["gb_hora"])
    o = tarifas["otros_costos"]
    otros = o["registro_imagen_gb"] * o["registro_imagen_gb_mes"] + costo_transferencia(
        esc["predicciones_mes"], tarifas
    )
    return {
        "opcion": "Lote: tarea Fargate ARM por ejecución",
        "instancia": f"Fargate {f['vcpu']} vCPU / {f['memoria_gb']} GB",
        "instancias": 1,
        "horas_facturadas": horas,
        "computo_usd": computo,
        "creditos_cpu_usd": 0.0,
        "otros_usd": otros,
        "costo_mes_usd": computo + otros,
        "utilizacion_pct": None,
    }


def tabla_proyeccion(capacidad: list[dict], tarifas: dict) -> list[dict]:
    """Costo mensual por escenario de demanda: despliegue actual, optimizado y en lote."""
    por_config = {(c["api_workers"], c["api_cpus"], c["omp_num_threads"]): c for c in capacidad}
    base = por_config.get((1, 2.0, "sin fijar"))
    optimizada = por_config.get((1, 2.0, "1")) or base
    if base is None or base["capacidad_rps"] is None:
        raise ValueError("falta un escenario de la configuración base que cumpla el SLO")
    ref = instancia(tarifas, tarifas["instancia_referencia"])
    chica = instancia(tarifas, "t4g.small")
    filas = []
    for esc in tarifas["escenarios_demanda"]:
        opciones = [
            _opcion_vm(
                f"Actual: VM 24/7 {ref['sku']}, configuración base", esc, ref, base, tarifas
            ),
            _opcion_vm(
                f"Optimizada: VM 24/7 {chica['sku']}, OMP_NUM_THREADS=1",
                esc,
                chica,
                optimizada,
                tarifas,
            ),
        ]
        if esc.get("ejecuciones_mes"):
            opciones.append(_opcion_lote(esc, optimizada, tarifas))
        for op in opciones:
            unitario = costo_por_inferencia(op["costo_mes_usd"], esc["predicciones_mes"])
            filas.append(
                {
                    "demanda": esc["nombre"],
                    "hipotetico": esc["hipotetico"],
                    "predicciones_mes": esc["predicciones_mes"],
                    "pico_rps": esc["pico_rps"],
                    **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in op.items()},
                    "costo_por_1000_usd": por_mil(unitario),
                }
            )
    return filas


# --------------------------------------------------------------------------------------------
# Salidas
# --------------------------------------------------------------------------------------------


def escribir_csv(ruta: Path, filas: list[dict]) -> None:
    with ruta.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


def _usd(v: float | None, dec: int = 4) -> str:
    return "no definido" if v is None else f"${v:,.{dec}f}"


def _usd_sig(v: float | None) -> str:
    """Tres cifras significativas para costos que van de centavos a millonésimas."""
    return "no definido" if v is None else f"${v:.3g}"


def resumen_md(ventanas, sensibilidad, capacidad, proyeccion, tarifas) -> str:
    ref = tarifas["instancia_referencia"]
    lineas = [
        "# Resumen FinOps (generado por `scripts/estimate_inference_cost.py`)",
        "",
        (
            f"Tarifas: {tarifas['proveedor']} {tarifas['region']}, {tarifas['modalidad']}, consultadas el "
            f"{tarifas['fecha_consulta']} ({tarifas['moneda']}). Instancia de referencia: `{ref}`. "
            "Fuente de cada cifra: `references/finops_tarifas.json` y los `manifiesto.json` de la carga."
        ),
        "",
        "## Costo de cada ventana de 180 s en la instancia de referencia",
        "",
        "| Escenario | Config. API | Pred. exitosas | Pet/s | p95 (ms) | SLO | Costo ventana | Costo / 1,000 | Núcleo-s / inf. | CPU / 1,000 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for f in ventanas:
        nombre = f"**{f['escenario']}**" if f["oficial"] else f["escenario"]
        lineas.append(
            f"| {nombre} | {f['config_api']} | {f['predicciones_exitosas']:,} | {f['rps']:.1f} | {f['p95_ms']:.0f} | "
            f"{'sí' if f['cumple_slo'] else 'no'} | {_usd(f['costo_ventana_usd'], 5)} | {_usd(f['costo_por_1000_usd'], 5)} | "
            f"{f['nucleo_s_por_inferencia']:.4f} | {_usd(f['costo_cpu_por_1000_usd'], 6)} |"
        )
    lineas += [
        "",
        "## Sensibilidad a la instancia (escenarios oficiales)",
        "",
        "| Escenario | Instancia | Arq. | vCPU / GiB | $/h | Cabe con lo medido | Respeta límites del Compose | Costo / 1,000 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f in sensibilidad:
        lineas.append(
            f"| {f['escenario']} | {f['instancia']} | {f['arquitectura']} | {f['vcpu']} / {f['memoria_gib']} | "
            f"{f['precio_hora_usd']:.4f} | {'sí' if f['cabe_con_lo_medido'] else 'no'} "
            f"(requiere {f['vcpu_requeridas']:.2f} vCPU, {f['memoria_requerida_gib']:.2f} GiB) | "
            f"{'sí' if f['respeta_limites_compose'] else 'no'} | {_usd(f['costo_por_1000_usd'], 5)} |"
        )
    lineas += [
        "",
        f"## Capacidad por instancia de 2 vCPU con SLO p95 ≤ {tarifas['slo']['p95_ms_max']} ms",
        "",
        "| Config. API | Escenarios | Capacidad (pet/s) | p95 (ms) | Núcleo-s / inf. | RAM máx. API + MLflow | ¿Cota inferior? |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in capacidad:
        cap = (
            "ninguno cumple"
            if c["capacidad_rps"] is None
            else f"{c['capacidad_rps']:.1f} ({c['escenario_capacidad']})"
        )
        lineas.append(
            f"| {c['config_api']} | {c['escenarios']} | {cap} | {c['p95_ms'] or '-'} | "
            f"{'-' if c['nucleo_s_por_inferencia'] is None else f'{c["nucleo_s_por_inferencia"]:.4f}'} | "
            f"{'-' if c['memoria_max_mib'] is None else f'{c["memoria_max_mib"]:.0f} MiB'} | "
            f"{'sí, no se midió más carga' if c['es_cota_inferior'] else 'no'} |"
        )
    lineas += [
        "",
        f"## Proyección mensual ({tarifas['horas_mes']} h; utilización objetivo {tarifas['slo']['utilizacion_objetivo']:.0%})",
        "",
        "| Demanda | Predicciones / mes | Opción | Instancias | Costo / mes | Costo / 1,000 | Utilización |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in proyeccion:
        util = "-" if p["utilizacion_pct"] is None else f"{p['utilizacion_pct']:.3g} %"
        lineas.append(
            f"| {p['demanda']}{' (hipótesis)' if p['hipotetico'] else ''} | {p['predicciones_mes']:,} | {p['opcion']} | "
            f"{p['instancias']} | {_usd(p['costo_mes_usd'], 2)} | {_usd_sig(p['costo_por_1000_usd'])} | {util} |"
        )
    return "\n".join(lineas) + "\n"


def graficas(ventanas: list[dict], proyeccion: list[dict], carpeta: Path) -> list[Path]:
    """Dos figuras para el reporte con el estilo de las gráficas de carga."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tinta, tinta_2, rejilla, superficie = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    azul, naranja, verde = "#2a78d6", "#eb6834", "#1baf7a"
    plt.rcParams.update(
        {
            "figure.facecolor": superficie,
            "axes.facecolor": superficie,
            "axes.edgecolor": rejilla,
            "axes.labelcolor": tinta_2,
            "axes.titlecolor": tinta,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": rejilla,
            "grid.linewidth": 0.8,
            "xtick.color": tinta_2,
            "ytick.color": tinta_2,
            "font.size": 9.5,
            "legend.frameon": False,
            "savefig.dpi": 170,
        }
    )
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = []

    # 1) Costo por 1,000 inferencias y CPU por inferencia, por escenario (dos paneles, un eje cada uno)
    etiquetas = [f"{v['escenario']}\n{v['config_api'].replace(', ', ' · ')}" for v in ventanas]
    y = list(range(len(ventanas)))[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, campo, titulo, fmt in (
        (
            a1,
            "costo_por_1000_usd",
            "Costo por 1,000 inferencias\n(USD, t4g.medium, ventana de 180 s)",
            "${:.3g}",
        ),
        (
            a2,
            "nucleo_s_por_inferencia",
            "CPU de la API por inferencia\n(núcleo-segundos)",
            "{:.3g}",
        ),
    ):
        valores = [v[campo] for v in ventanas]
        colores = [
            azul if v["oficial"] else ("#9aa7b8" if v["cumple_slo"] else naranja) for v in ventanas
        ]
        ax.barh(y, valores, color=colores, height=0.62, edgecolor=superficie, linewidth=2)
        for yi, val in zip(y, valores):
            ax.text(
                val, yi, " " + fmt.format(val), va="center", ha="left", color=tinta, fontsize=8.5
            )
        ax.set_title(titulo, loc="left")
        ax.set_xlim(0, max(valores) * 1.32)
        ax.grid(axis="y", visible=False)
    a1.set_yticks(y, etiquetas, fontsize=8.5)
    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(color=azul, label="Escenario oficial FinOps"),
            Patch(color="#9aa7b8", label="Exploratorio, cumple SLO p95 ≤ 200 ms"),
            Patch(color=naranja, label="Exploratorio, no cumple SLO"),
        ],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    ruta = carpeta / "costo_por_escenario.png"
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    rutas.append(ruta)

    # 2) Proyección mensual: costo por 1,000 por demanda y opción de despliegue (escala log)
    demandas = list(dict.fromkeys(p["demanda"] for p in proyeccion))
    opciones = list(dict.fromkeys(p["opcion"] for p in proyeccion))
    colores_op = dict(zip(opciones, ["#7d8796", azul, verde]))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ancho = 0.26
    for j, op in enumerate(opciones):
        for i, d in enumerate(demandas):
            fila = next((p for p in proyeccion if p["demanda"] == d and p["opcion"] == op), None)
            if fila is None:
                continue
            x = i + (j - (len(opciones) - 1) / 2) * ancho
            ax.bar(
                x,
                fila["costo_por_1000_usd"],
                width=ancho - 0.03,
                color=colores_op[op],
                label=op if i == 0 else None,
            )
            ax.text(
                x,
                fila["costo_por_1000_usd"] * 1.15,
                f"${fila['costo_por_1000_usd']:.3g}\n${fila['costo_mes_usd']:,.2f}/mes",
                ha="center",
                va="bottom",
                fontsize=7.6,
                color=tinta,
            )
    ax.set_yscale("log")
    ax.set_ylim(top=max(p["costo_por_1000_usd"] for p in proyeccion) * 30)
    ax.set_xticks(
        range(len(demandas)),
        [
            f"{d}\n({next(p['predicciones_mes'] for p in proyeccion if p['demanda'] == d):,} pred/mes)"
            for d in demandas
        ],
    )
    ax.set_ylabel("USD por 1,000 inferencias (escala log)")
    ax.set_title(
        "Costo por 1,000 inferencias en operación mensual según demanda y despliegue", loc="left"
    )
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    ruta = carpeta / "proyeccion_mensual.png"
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    rutas.append(ruta)
    return rutas


def _commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=RAIZ,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tarifas", type=Path, default=TARIFAS)
    p.add_argument("--carga", type=Path, default=CARGA)
    p.add_argument("--salida", type=Path, default=SALIDA)
    p.add_argument("--sin-graficas", action="store_true")
    args = p.parse_args(argv)

    tarifas = json.loads(args.tarifas.read_text(encoding="utf-8"))
    manifiestos = cargar_manifiestos(args.carga)
    ventanas = tabla_ventanas(manifiestos, tarifas)
    sensibilidad = tabla_sensibilidad(manifiestos, tarifas)
    capacidad = tabla_capacidad(manifiestos, tarifas)
    proyeccion = tabla_proyeccion(capacidad, tarifas)

    args.salida.mkdir(parents=True, exist_ok=True)
    escribir_csv(args.salida / "costo_ventana.csv", ventanas)
    escribir_csv(args.salida / "sensibilidad_instancias.csv", sensibilidad)
    escribir_csv(args.salida / "capacidad_slo.csv", capacidad)
    escribir_csv(args.salida / "proyeccion_mensual.csv", proyeccion)
    (args.salida / "resumen.md").write_text(
        resumen_md(ventanas, sensibilidad, capacidad, proyeccion, tarifas), encoding="utf-8"
    )
    resumen = {
        "git_commit": _commit(),
        "tarifas": tarifas,
        "modelo": manifiestos[0]["modelo"],
        "maquina_medicion": manifiestos[0]["maquina"],
        "ventanas": ventanas,
        "sensibilidad_instancias": sensibilidad,
        "capacidad_slo": capacidad,
        "proyeccion_mensual": proyeccion,
    }
    (args.salida / "finops_resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not args.sin_graficas:
        graficas(ventanas, proyeccion, args.salida / "graficas")

    for v in ventanas:
        if v["oficial"]:
            print(
                f"{v['escenario']}: {v['predicciones_exitosas']:,} predicciones en {v['duracion_s']:.0f} s · "
                f"costo de la ventana {_usd(v['costo_ventana_usd'], 6)} · por 1,000 {_usd(v['costo_por_1000_usd'], 5)}"
            )
    print(
        f"Tablas y gráficas en {args.salida.relative_to(RAIZ) if args.salida.is_relative_to(RAIZ) else args.salida}"
    )
    return resumen


if __name__ == "__main__":
    main()
