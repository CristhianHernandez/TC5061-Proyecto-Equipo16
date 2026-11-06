"""Petición HTTP real a Uvicorn con el modelo configurado; no entrena ni cambia aliases.

Desde la raíz, con MODEL_PATH o MLFLOW_TRACKING_URI preparados:
python docs/evidencias/actividad8/api/comprobar_local.py
El puerto de loopback es efímero y el servidor se cierra al finalizar.
"""

import importlib.metadata
import json
from pathlib import Path
import platform
import socket
from threading import Thread
import time

import httpx
import uvicorn

from mibici import config
from mibici.serving.api import app


def comprobar():
    payload = json.loads(config.EJEMPLO_SERVICIO.read_text(encoding="utf-8"))
    with socket.socket() as conexion:
        conexion.bind(("127.0.0.1", 0))
        conexion.listen()
        puerto = conexion.getsockname()[1]
        servidor = uvicorn.Server(
            uvicorn.Config(
                app, host="127.0.0.1", port=puerto, log_level="warning", access_log=False
            )
        )
        hilo = Thread(target=servidor.run, kwargs={"sockets": [conexion]}, daemon=True)
        hilo.start()
        try:
            limite = time.monotonic() + 30
            while not servidor.started:
                if not hilo.is_alive() or time.monotonic() >= limite:
                    raise RuntimeError("Uvicorn no pudo iniciar dentro del plazo")
                time.sleep(0.05)
            with httpx.Client(
                base_url=f"http://127.0.0.1:{puerto}", timeout=10, trust_env=False
            ) as cliente:
                salud = cliente.get("/health")
                if salud.status_code != 200:
                    raise RuntimeError("No hay modelo listo: revise el bundle o el Registry")
                respuesta = cliente.post("/predict", json=payload)
                respuesta.raise_for_status()
                esperado = app.state.modelo.predecir(payload["registros"])
                assert respuesta.json()["predicciones"] == esperado
                invalido = {
                    "registros": [
                        {k: v for k, v in payload["registros"][0].items() if k != "capacidad"}
                    ]
                }
                negativa = cliente.post("/predict", json=invalido)
                assert negativa.status_code == 422
                swagger = cliente.get("/docs")
                assert swagger.status_code == 200
                return {
                    "prueba": "HTTP real por loopback a Uvicorn; no es una prueba de contenedor",
                    "python": platform.python_version(),
                    "versiones": {
                        p: importlib.metadata.version(p)
                        for p in ("fastapi", "starlette", "pydantic", "httpx", "uvicorn", "mlflow")
                    },
                    "origen_modelo": app.state.modelo.metadatos.get("origen"),
                    "firma": app.state.modelo.contrato(),
                    "health": {"status": salud.status_code, "json": salud.json()},
                    "predict": {
                        "status": respuesta.status_code,
                        "entrada": payload,
                        "json": respuesta.json(),
                    },
                    "coincide_con_loader": True,
                    "entrada_invalida": {"status": negativa.status_code, "json": negativa.json()},
                    "docs": {"status": swagger.status_code},
                }
        finally:
            servidor.should_exit = True
            hilo.join(timeout=10)
            if hilo.is_alive():
                raise RuntimeError("Uvicorn no terminó correctamente")


if __name__ == "__main__":
    evidencia = comprobar()
    destino = Path(__file__).with_name("respuesta_local.json")
    destino.write_text(
        json.dumps(evidencia, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("HTTP: health=200, predict=200, entrada inválida=422, docs=200")
    print(json.dumps(evidencia["predict"]["json"], ensure_ascii=False))
