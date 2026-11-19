"""Manifiesto de una corrida de carga: commit, modelo, máquina, ventana y conteos (insumo de FinOps).

    python loadtests/manifiesto.py docs/evidencias/operacion/carga/u05

Lee `ventana.json`, `locust_stats.csv`, `recursos.csv`, `health.json` y `modelo_registry.json` de
la carpeta de la corrida y escribe `manifiesto.json`. Lo invoca `run_escenario.sh` al terminar.
"""

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import pandas as pd

_UNIDADES = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "kB": 1e3, "MB": 1e6, "GB": 1e9}


def _mib(texto: str) -> float:
    """'197.6MiB' -> 197.6 (MiB)."""
    m = re.fullmatch(r"\s*([\d.]+)\s*([A-Za-z]+)\s*", texto)
    if not m:
        return float("nan")
    return float(m.group(1)) * _UNIDADES.get(m.group(2), float("nan")) / 2**20


def leer_recursos(ruta: Path) -> pd.DataFrame:
    """recursos.csv de capturar_recursos.sh -> timestamp, contenedor, cpu_pct, mem_mib, limite_mib."""
    df = pd.read_csv(ruta)
    df["timestamp"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["cpu_pct"] = df["cpu_pct"].astype(str).str.rstrip("%").astype(float)
    usada_limite = df["memoria"].astype(str).str.split("/", n=1, expand=True)
    df["mem_mib"] = usada_limite[0].map(_mib)
    df["limite_mib"] = usada_limite[1].map(_mib)
    df["servicio"] = df["contenedor"].str.replace(r"^mibici-|-\d+$", "", regex=True)
    return df[["timestamp", "contenedor", "servicio", "cpu_pct", "mem_mib", "limite_mib"]]


def resumen_recursos(recursos: pd.DataFrame, inicio: str, fin: str) -> dict:
    """Promedio, p95 y máximo de CPU y RAM por servicio dentro de la ventana [inicio, fin]."""
    ventana = recursos[
        (recursos["timestamp"] >= pd.Timestamp(inicio)) & (recursos["timestamp"] <= pd.Timestamp(fin))
    ]
    salida = {}
    for servicio, g in ventana.groupby("servicio"):
        salida[servicio] = {
            "muestras": int(len(g)),
            "cpu_pct_prom": round(g["cpu_pct"].mean(), 1),
            "cpu_pct_p95": round(g["cpu_pct"].quantile(0.95), 1),
            "cpu_pct_max": round(g["cpu_pct"].max(), 1),
            "mem_mib_prom": round(g["mem_mib"].mean(), 1),
            "mem_mib_max": round(g["mem_mib"].max(), 1),
            "limite_mem_mib": round(g["limite_mib"].iloc[-1], 0),
        }
    return salida


def _cmd(*args: str) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _config_api(docker: str) -> dict:
    """Workers y límites reales del contenedor de la API (no los supuestos del Compose)."""
    datos = _cmd(docker, "inspect", "mibici-api-1")
    if not datos:
        return {}
    info = json.loads(datos)[0]
    cmd = info["Config"]["Cmd"] or []
    workers = next((c.split("=", 1)[1] for c in cmd if c.startswith("--workers=")), "1")
    host = info["HostConfig"]
    return {
        "api_workers": int(workers),
        "api_cpus_limite": (host.get("NanoCpus") or 0) / 1e9 or None,
        "api_mem_limite_mib": (host.get("Memory") or 0) / 2**20 or None,
        "imagen": info["Config"]["Image"],
    }


def _maquina(docker: str) -> dict:
    vm = _cmd(docker, "info", "--format", "{{.NCPU}}|{{.MemTotal}}|{{.ServerVersion}}").split("|")
    chip = _cmd("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor()
    nucleos = _cmd("sysctl", "-n", "hw.ncpu") or str(os.cpu_count())
    memoria = _cmd("sysctl", "-n", "hw.memsize")
    return {
        "host": f"{platform.system()} {platform.mac_ver()[0] or platform.release()} {platform.machine()}",
        "cpu": chip,
        "nucleos_host": int(nucleos) if nucleos.isdigit() else None,
        "ram_host_gib": round(int(memoria) / 2**30, 1) if memoria.isdigit() else None,
        "motor_contenedores": f"{docker} {vm[2]}" if len(vm) == 3 else docker,
        "vm_nucleos": int(vm[0]) if len(vm) == 3 and vm[0].isdigit() else None,
        "vm_ram_gib": round(int(vm[1]) / 2**30, 2) if len(vm) == 3 and vm[1].isdigit() else None,
        "locust_mismo_host": True,
    }


def construir(directorio: Path, docker: str = "docker") -> dict:
    ventana = json.loads((directorio / "ventana.json").read_text())
    stats = pd.read_csv(directorio / "locust_stats.csv")
    agregado = stats[stats["Name"] == "Aggregated"].iloc[0]
    registro = json.loads((directorio / "modelo_registry.json").read_text())["model_version"]
    etiquetas = {t["key"]: t["value"] for t in registro.get("tags", [])}
    salud = json.loads((directorio / "health.json").read_text())
    inicio, fin = ventana["inicio_ventana_utc"], ventana["fin_ventana_utc"]
    duracion = (pd.Timestamp(fin) - pd.Timestamp(inicio)).total_seconds()
    recursos = leer_recursos(directorio / "recursos.csv")

    return {
        "escenario": directorio.name,
        "generado_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "git_commit": _cmd("git", "rev-parse", "--short", "HEAD"),
        "git_sucio": bool(_cmd("git", "status", "--porcelain", "--untracked-files=no")),
        "modelo": {
            "nombre": registro["name"],
            "version_stack": registro["version"],
            "origen": etiquetas.get("origen"),
            "version_origen": etiquetas.get("version_origen"),
            "run_id_origen": etiquetas.get("run_id_origen"),
            "run_id_stack": salud.get("modelo", {}).get("run_id"),
        },
        "carga": {
            "usuarios": int(os.environ.get("USUARIOS", "0")) or None,
            "spawn_rate_usuarios_s": float(os.environ.get("SPAWN", "0")) or None,
            "espera_s": ventana["espera_s"],
            "calentamiento_s": ventana["calentamiento_s"],
            "registros_por_peticion": ventana["registros_por_peticion"],
            "endpoint": "POST /predict",
        },
        "ventana": {
            "inicio_prueba_utc": ventana["inicio_prueba_utc"],
            "inicio_utc": inicio,
            "fin_utc": fin,
            "duracion_s": duracion,
        },
        "resultados_ventana": {
            "peticiones_totales": ventana["peticiones"],
            "peticiones_fallidas": ventana["fallidas"],
            "predicciones_exitosas": ventana["predicciones_exitosas"],
            "tasa_error_pct": round(100 * ventana["fallidas"] / ventana["peticiones"], 3)
            if ventana["peticiones"]
            else None,
            "rps_prom": round(ventana["peticiones"] / duracion, 2) if duracion else None,
            "predicciones_por_s": round(ventana["predicciones_exitosas"] / duracion, 2)
            if duracion
            else None,
            "latencia_ms": {
                "p50": float(agregado["50%"]),
                "p95": float(agregado["95%"]),
                "p99": float(agregado["99%"]),
                "promedio": round(float(agregado["Average Response Time"]), 1),
                "max": float(agregado["Max Response Time"]),
            },
        },
        "recursos_ventana": resumen_recursos(recursos, inicio, fin),
        "configuracion_api": _config_api(docker),
        "maquina": _maquina(docker),
    }


if __name__ == "__main__":
    carpeta = Path(sys.argv[1])
    manifiesto = construir(carpeta, os.environ.get("DOCKER", "docker"))
    (carpeta / "manifiesto.json").write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    r = manifiesto["resultados_ventana"]
    print(
        f"{carpeta.name}: {r['peticiones_totales']} peticiones, {r['peticiones_fallidas']} fallidas, "
        f"{r['predicciones_exitosas']} predicciones en {manifiesto['ventana']['duracion_s']:.0f} s"
    )
