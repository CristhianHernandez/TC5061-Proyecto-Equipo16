"""Gráficas y tablas de las pruebas de carga a partir de la evidencia de cada escenario.

    python loadtests/graficas_carga.py                    # todos los escenarios con manifiesto
    python loadtests/graficas_carga.py u05 u20            # solo los indicados, en ese orden

Lee docs/evidencias/operacion/carga/<escenario>/{locust_stats_history.csv, recursos.csv,
manifiesto.json} y escribe en docs/evidencias/operacion/carga/:
  graficas/latencia_<grupo>.png      p50 y p95 por segundo (ventana móvil de 10 s de Locust)
  graficas/rps_<grupo>.png           peticiones por segundo
  graficas/recursos_api_<grupo>.png  CPU y RAM de la API durante la corrida
  graficas/errores.png               tasa de error de la ventana estable por escenario
  graficas/resumen.png               p95 y CPU promedio por escenario
  (<grupo> = escalamiento: 5-120 usuarios con la configuración base; optimizacion: variantes a
  120 usuarios. Ver GRUPOS.)
  resumen_escenarios.csv     una fila por escenario: ventana, conteos, latencias y recursos
  resumen_escenarios.md      la misma tabla en Markdown, para el reporte
"""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from manifiesto import leer_recursos

RAIZ = Path(__file__).resolve().parents[1]
CARGA = RAIZ / "docs" / "evidencias" / "operacion" / "carga"
GRAFICAS = CARGA / "graficas"

# Paleta categórica en orden fijo (validada para daltonismo); el color sigue al escenario.
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
VENTANA_MOVIL_S = 10
TINTA, TINTA_2, REJILLA, SUPERFICIE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "axes.edgecolor": REJILLA,
        "axes.labelcolor": TINTA_2,
        "axes.titlecolor": TINTA,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": REJILLA,
        "grid.linewidth": 0.8,
        "xtick.color": TINTA_2,
        "ytick.color": TINTA_2,
        "font.size": 10,
        "legend.frameon": False,
        "savefig.dpi": 160,
    }
)


CPUS_COMPOSE = 2.0  # límite por defecto de la API en docker-compose.yml (API_CPUS)


def _cpus(m: dict) -> float:
    return m["configuracion_api"].get("api_cpus_limite") or CPUS_COMPOSE


def etiqueta(m: dict) -> str:
    workers = m["configuracion_api"].get("api_workers", 1)
    texto = f"{m['carga']['usuarios']} usuarios · {workers} worker{'s' if workers > 1 else ''}"
    if _cpus(m) != CPUS_COMPOSE:
        texto += f" · {_cpus(m):g} CPU"
    if m["configuracion_api"].get("omp_num_threads"):
        texto += f" · OMP={m['configuracion_api']['omp_num_threads']}"
    return texto


def etiqueta_corta(m: dict) -> str:
    texto = f"{m['carga']['usuarios']}u·{m['configuracion_api'].get('api_workers', 1)}w"
    if _cpus(m) != CPUS_COMPOSE:
        texto += f"·{_cpus(m):g}cpu"
    if m["configuracion_api"].get("omp_num_threads"):
        texto += f"·omp{m['configuracion_api']['omp_num_threads']}"
    return texto


def cargar(escenario: str) -> dict:
    d = CARGA / escenario
    m = json.loads((d / "manifiesto.json").read_text())
    hist = pd.read_csv(d / "locust_stats_history.csv")
    hist = hist[hist["Name"] == "Aggregated"].copy()
    inicio = pd.Timestamp(m["ventana"]["inicio_prueba_utc"]).timestamp()
    hist["t"] = hist["Timestamp"] - inicio
    for col in ("50%", "95%"):
        # 0 = segundo sin respuestas (p. ej. justo al reiniciar las estadísticas): no es latencia
        hist[col] = pd.to_numeric(hist[col], errors="coerce").where(lambda v: v > 0)
    # Locust calcula p50/p95 y peticiones/s "actuales" con una ventana móvil de 10 s; al reiniciar
    # las estadísticas tras el calentamiento esa ventana se vacía y produce una caída artificial.
    cal = m["carga"]["calentamiento_s"]
    artefacto = (hist["t"] >= cal) & (hist["t"] < cal + VENTANA_MOVIL_S)
    hist.loc[artefacto, ["50%", "95%", "Requests/s"]] = float("nan")
    rec = leer_recursos(d / "recursos.csv")
    rec["t"] = (
        rec["timestamp"] - pd.Timestamp(m["ventana"]["inicio_prueba_utc"])
    ).dt.total_seconds()
    return {"nombre": escenario, "m": m, "hist": hist, "rec": rec}


def _eje_log(ax) -> None:
    """Escala logarítmica con marcas legibles en ms (10, 20, 50, 100...)."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(ScalarFormatter())


def _marcar_calentamiento(ax, calentamiento: float) -> None:
    ax.axvspan(0, calentamiento, color=REJILLA, alpha=0.5, lw=0, zorder=0)
    ax.text(
        calentamiento / 2,
        0.97,
        "calentamiento",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8,
        color=TINTA_2,
    )


def _etiquetas_finales(
    ax, puntos: list[tuple[float, float, str, str]], sep_pt: float = 11
) -> None:
    """Etiquetas directas al final de cada línea, separadas verticalmente si se enciman.

    La identidad de cada serie no depende solo del color (la paleta tiene tonos claros).
    """
    puntos = [p for p in puntos if pd.notna(p[1])]
    if not puntos:
        return
    ax.figure.canvas.draw()
    a_pantalla = ax.transData.transform
    ys = sorted(
        ((a_pantalla((x, y))[1], x, y, texto, color) for x, y, texto, color in puntos),
        key=lambda p: p[0],
    )
    sep_px = sep_pt * ax.figure.dpi / 72
    colocadas: list[float] = []
    for y_px, *_ in ys:
        colocadas.append(max(y_px, colocadas[-1] + sep_px) if colocadas else y_px)
    for y_obj, (y_px, x, y, texto, color) in zip(colocadas, ys):
        ax.plot([x], [y], "o", ms=5, color=color, mec=SUPERFICIE, mew=1.5)
        ax.annotate(
            texto,
            (x, y),
            xytext=(6, (y_obj - y_px) * 72 / ax.figure.dpi),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=TINTA,
        )


def _serie_por_escenario(datos, columna, titulo, ylabel, archivo, log=False, suavizar=1):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    puntos = []
    for i, e in enumerate(datos):
        h = e["hist"]
        y = h[columna].rolling(suavizar, min_periods=1).mean() if suavizar > 1 else h[columna]
        ax.plot(h["t"], y, lw=2, color=e["color"], label=etiqueta(e["m"]))
        validos = y.dropna()
        if len(validos):
            puntos.append(
                (h["t"].loc[validos.index[-1]], validos.iloc[-1], etiqueta(e["m"]), e["color"])
            )
    _marcar_calentamiento(ax, datos[0]["m"]["carga"]["calentamiento_s"])
    if log:
        _eje_log(ax)
    ax.set(title=titulo, xlabel="segundos desde el inicio de la prueba", ylabel=ylabel)
    ax.margins(x=0.02)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.18)  # espacio para etiquetas directas
    _etiquetas_finales(ax, puntos)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(GRAFICAS / archivo)
    plt.close(fig)


def _puntos_finales(datos, col: str) -> list[tuple[float, float, str, str]]:
    puntos = []
    for i, e in enumerate(datos):
        validos = e["hist"][col].dropna()
        if len(validos):
            corta = etiqueta_corta(e["m"])
            puntos.append(
                (e["hist"]["t"].loc[validos.index[-1]], validos.iloc[-1], corta, e["color"])
            )
    return puntos


def grafica_latencia(datos, archivo: str = "latencia.png") -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, col, nombre in zip(ejes, ("50%", "95%"), ("p50", "p95")):
        for i, e in enumerate(datos):
            h = e["hist"]
            ax.plot(h["t"], h[col], lw=2, color=e["color"], label=etiqueta(e["m"]))
        _marcar_calentamiento(ax, datos[0]["m"]["carga"]["calentamiento_s"])
        _eje_log(ax)
        ax.set(title=f"Latencia {nombre} de POST /predict", xlabel="segundos desde el inicio")
        ax.set_xlim(right=ax.get_xlim()[1] * 1.12)
    for ax, col in zip(ejes, ("50%", "95%")):
        _etiquetas_finales(ax, _puntos_finales(datos, col))
    ejes[0].set_ylabel("ms (escala logarítmica, ventana móvil de 10 s)")
    ejes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(GRAFICAS / archivo)
    plt.close(fig)


def grafica_errores(datos) -> None:
    """Tasa de error de la ventana estable por escenario, con el conteo explícito."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    nombres = [etiqueta(e["m"]) for e in datos]
    tasas = [e["m"]["resultados_ventana"]["tasa_error_pct"] or 0 for e in datos]
    ax.barh(nombres, tasas, color=[e["color"] for e in datos], height=0.55)
    tope = max(1.0, max(tasas) * 1.3)
    for i, e in enumerate(datos):
        r = e["m"]["resultados_ventana"]
        ax.text(
            tasas[i] + tope * 0.01,
            i,
            f"{tasas[i]:.2f} %  ({r['peticiones_fallidas']:,} fallidas de "
            f"{r['peticiones_totales']:,} peticiones)",
            va="center",
            fontsize=9,
            color=TINTA,
        )
    ax.set_xlim(0, tope)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set(
        title="Tasa de error en la ventana estable (HTTP ≠ 200 o respuesta inválida)",
        xlabel="% de peticiones fallidas",
    )
    fig.tight_layout()
    fig.savefig(GRAFICAS / "errores.png")
    plt.close(fig)


def grafica_recursos(datos, archivo: str = "recursos_api.png") -> None:
    fig, (cpu, mem) = plt.subplots(1, 2, figsize=(11, 4.6))
    for i, e in enumerate(datos):
        api = e["rec"][e["rec"]["servicio"] == "api"]
        cpu.plot(api["t"], api["cpu_pct"], lw=2, color=e["color"], label=etiqueta(e["m"]))
        mem.plot(api["t"], api["mem_mib"], lw=2, color=e["color"], label=etiqueta(e["m"]))
    limites = sorted({100 * _cpus(e["m"]) for e in datos})
    for limite_cpu in limites:
        cpu.axhline(limite_cpu, color=TINTA_2, lw=1, ls="--")
        cpu.text(
            0.99,
            limite_cpu,
            f"límite del contenedor ({limite_cpu:.0f} %)",
            transform=cpu.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            color=TINTA_2,
        )
    for ax in (cpu, mem):
        _marcar_calentamiento(ax, datos[0]["m"]["carga"]["calentamiento_s"])
        ax.set_xlabel("segundos desde el inicio")
    cpu.set(title="CPU de la API (docker stats)", ylabel="% de CPU (100 % = 1 núcleo)")
    cpu.set_ylim(bottom=0, top=max(limites[-1] * 1.15, cpu.get_ylim()[1]))
    mem.set(title="Memoria de la API (docker stats)", ylabel="MiB")
    mem.set_ylim(bottom=0)
    cpu.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(GRAFICAS / archivo)
    plt.close(fig)


def tabla_resumen(datos) -> pd.DataFrame:
    filas = []
    for e in datos:
        m = e["m"]
        r, rec = m["resultados_ventana"], m["recursos_ventana"]
        filas.append(
            {
                "escenario": e["nombre"],
                "usuarios": m["carga"]["usuarios"],
                "spawn_rate": m["carga"]["spawn_rate_usuarios_s"],
                "api_workers": m["configuracion_api"].get("api_workers"),
                "api_cpus": _cpus(m),
                "omp_num_threads": m["configuracion_api"].get("omp_num_threads") or "sin fijar",
                "inicio_ventana_utc": m["ventana"]["inicio_utc"],
                "fin_ventana_utc": m["ventana"]["fin_utc"],
                "duracion_s": m["ventana"]["duracion_s"],
                "peticiones": r["peticiones_totales"],
                "fallidas": r["peticiones_fallidas"],
                "predicciones_exitosas": r["predicciones_exitosas"],
                "error_pct": r["tasa_error_pct"],
                "rps": r["rps_prom"],
                "p50_ms": r["latencia_ms"]["p50"],
                "p95_ms": r["latencia_ms"]["p95"],
                "p99_ms": r["latencia_ms"]["p99"],
                "api_cpu_prom_pct": rec["api"]["cpu_pct_prom"],
                "api_cpu_p95_pct": rec["api"]["cpu_pct_p95"],
                "api_mem_max_mib": rec["api"]["mem_mib_max"],
                "mlflow_cpu_prom_pct": rec["mlflow"]["cpu_pct_prom"],
                "mlflow_mem_max_mib": rec["mlflow"]["mem_mib_max"],
            }
        )
    return pd.DataFrame(filas)


def _markdown(tabla: pd.DataFrame) -> str:
    """Tabla Markdown sin depender de tabulate."""
    filas = [list(map(str, tabla.columns)), ["---"] * len(tabla.columns)]
    filas += [[str(v) for v in fila] for fila in tabla.itertuples(index=False)]
    return "\n".join("| " + " | ".join(f) + " |" for f in filas) + "\n"


def grafica_resumen(datos) -> None:
    """p95 y CPU de la API en la ventana estable por escenario (paneles separados, sin doble eje)."""
    fig, (lat, cpu) = plt.subplots(1, 2, figsize=(11, 0.55 * len(datos) + 1.6), sharey=True)
    nombres = [etiqueta(e["m"]) for e in datos]
    colores = [e["color"] for e in datos]
    p95 = [e["m"]["resultados_ventana"]["latencia_ms"]["p95"] for e in datos]
    uso = [e["m"]["recursos_ventana"]["api"]["cpu_pct_prom"] for e in datos]
    for ax, valores, fmt, titulo in (
        (lat, p95, "{:.0f} ms", "Latencia p95 en la ventana estable"),
        (cpu, uso, "{:.0f} %", "CPU promedio de la API (100 % = 1 núcleo)"),
    ):
        ax.barh(nombres, valores, color=colores, height=0.6)
        for i, v in enumerate(valores):
            ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8, color=TINTA)
        ax.set_xlim(0, max(valores) * 1.2)
        ax.grid(axis="y", visible=False)
        ax.set_title(titulo)
    lat.invert_yaxis()
    fig.tight_layout()
    fig.savefig(GRAFICAS / "resumen.png")
    plt.close(fig)


# Familias de gráficas: el escalamiento con la configuración base y las variantes a 120 usuarios.
GRUPOS = {
    "escalamiento": ["u05", "u20", "u60", "u120"],
    "optimizacion": ["u120", "u120_w2", "u120_w2_cpu4", "u120_omp1"],
}


def main(escenarios: list[str]) -> None:
    if not escenarios:
        escenarios = sorted(
            (p.parent.name for p in CARGA.glob("*/manifiesto.json")),
            key=lambda n: (int("".join(c for c in n.split("_")[0] if c.isdigit()) or 0), n),
        )
    GRAFICAS.mkdir(parents=True, exist_ok=True)
    datos = [cargar(e) for e in escenarios]
    for i, e in enumerate(datos):  # el color sigue al escenario en todas las gráficas
        e["color"] = COLORES[i % len(COLORES)]
    por_nombre = {e["nombre"]: e for e in datos}
    for grupo, nombres in GRUPOS.items():
        sub = [por_nombre[n] for n in nombres if n in por_nombre]
        if not sub:
            continue
        grafica_latencia(sub, f"latencia_{grupo}.png")
        _serie_por_escenario(
            sub,
            "Requests/s",
            "Peticiones por segundo atendidas",
            "peticiones/s",
            f"rps_{grupo}.png",
            suavizar=5,
        )
        grafica_recursos(sub, f"recursos_api_{grupo}.png")
    grafica_errores(datos)
    grafica_resumen(datos)
    tabla = tabla_resumen(datos)
    tabla.to_csv(CARGA / "resumen_escenarios.csv", index=False)
    (CARGA / "resumen_escenarios.md").write_text(_markdown(tabla))
    print(
        tabla[
            [
                "escenario",
                "peticiones",
                "fallidas",
                "rps",
                "p50_ms",
                "p95_ms",
                "api_cpu_prom_pct",
                "api_mem_max_mib",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main(sys.argv[1:])
