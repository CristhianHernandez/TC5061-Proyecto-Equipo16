"""Prueba de carga de la API de P1 (Entregable 7, PR #2 de LH3).

Cada usuario virtual envía `POST /predict` con **un registro por petición**, tomado en orden fijo de
`loadtests/payloads_p1.json` (24 registros reales de oct-2026). Así cada petición exitosa equivale a
una predicción y FinOps puede dividir el costo entre peticiones exitosas.

Una respuesta cuenta como exitosa solo si es HTTP 200, trae JSON con `predicciones` = lista de un
`float` finito > 0 y el modelo esperado. Un 200 con otra estructura se registra como **fallido**.

Calentamiento: durante los primeros `LOCUST_CALENTAMIENTO` segundos (por defecto 60) se envía carga
normal, pero al terminar se reinician las estadísticas. Por eso `<prefijo>_stats.csv` y
`<prefijo>_failures.csv` cubren **solo la ventana estable**, mientras que `_stats_history.csv`
conserva toda la corrida (los acumulados vuelven a cero en el reinicio). Al terminar se escribe
`LOCUST_VENTANA_JSON` con los límites de la ventana y los conteos propios de esa ventana.

    locust -f loadtests/locustfile.py --host http://127.0.0.1:8000 --headless \
        -u 5 -r 1 --run-time 4m --csv docs/evidencias/operacion/carga/u05/locust \
        --html docs/evidencias/operacion/carga/u05/locust.html

Normalmente se lanza con `loadtests/run_escenario.sh`, que además captura CPU/RAM y el manifiesto.
"""

from datetime import UTC, datetime
import itertools
import json
import os
from pathlib import Path
import threading

import gevent
from locust import HttpUser, between, events, task
from validacion import validar_respuesta  # locust agrega loadtests/ al sys.path

PAYLOADS = Path(
    os.environ.get("LOCUST_PAYLOADS", Path(__file__).with_name("payloads_p1.json"))
)
CALENTAMIENTO_S = float(os.environ.get("LOCUST_CALENTAMIENTO", "60"))
ESPERA_MIN = float(os.environ.get("LOCUST_ESPERA_MIN", "0.5"))
ESPERA_MAX = float(os.environ.get("LOCUST_ESPERA_MAX", "1.5"))
MODELO_ESPERADO = os.environ.get("LOCUST_MODELO", "mibici-equipo16-p1-demanda")
VENTANA_JSON = os.environ.get("LOCUST_VENTANA_JSON")

REGISTROS = json.loads(PAYLOADS.read_text(encoding="utf-8"))["registros"]
_rotacion = itertools.cycle(REGISTROS)  # mezcla fija y compartida entre usuarios
_candado = threading.Lock()


def _ahora() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class Ventana:
    """Conteos de la ventana estable, independientes de las estadísticas de Locust."""

    def __init__(self) -> None:
        self.inicio_prueba: str | None = None
        self.inicio: str | None = None
        self.fin: str | None = None
        self.en_ventana = False
        self.peticiones = 0
        self.fallidas = 0
        self.predicciones_exitosas = 0

    def contar(self, exitosa: bool, predicciones: int) -> None:
        if not self.en_ventana:
            return
        with _candado:
            self.peticiones += 1
            if exitosa:
                self.predicciones_exitosas += predicciones
            else:
                self.fallidas += 1


VENTANA = Ventana()


class UsuarioPrediccion(HttpUser):
    wait_time = between(ESPERA_MIN, ESPERA_MAX)

    @task
    def predecir(self) -> None:
        with _candado:
            registro = next(_rotacion)
        with self.client.post(
            "/predict", json={"registros": [registro]}, name="POST /predict", catch_response=True
        ) as respuesta:
            if respuesta.status_code != 200:
                respuesta.failure(f"HTTP {respuesta.status_code}")
                VENTANA.contar(False, 0)
                return
            try:
                motivo = validar_respuesta(respuesta.json(), 1, MODELO_ESPERADO)
            except ValueError:
                motivo = "JSON inválido"
            if motivo:
                respuesta.failure(motivo)
                VENTANA.contar(False, 0)
            else:
                respuesta.success()
                VENTANA.contar(True, 1)


def _abrir_ventana(environment) -> None:
    gevent.sleep(CALENTAMIENTO_S)
    environment.runner.stats.reset_all()  # _stats.csv queda solo con la ventana estable
    VENTANA.inicio = _ahora()
    VENTANA.en_ventana = True


@events.test_start.add_listener
def _al_iniciar(environment, **_kwargs) -> None:
    VENTANA.inicio_prueba = _ahora()
    if CALENTAMIENTO_S > 0:
        gevent.spawn(_abrir_ventana, environment)
    else:
        VENTANA.inicio, VENTANA.en_ventana = VENTANA.inicio_prueba, True


@events.test_stop.add_listener
def _al_terminar(environment, **_kwargs) -> None:
    VENTANA.en_ventana = False
    VENTANA.fin = _ahora()
    if not VENTANA_JSON:
        return
    total = environment.runner.stats.total
    resumen = {
        "inicio_prueba_utc": VENTANA.inicio_prueba,
        "inicio_ventana_utc": VENTANA.inicio,
        "fin_ventana_utc": VENTANA.fin,
        "calentamiento_s": CALENTAMIENTO_S,
        "espera_s": [ESPERA_MIN, ESPERA_MAX],
        "registros_por_peticion": 1,
        "pool_payloads": str(PAYLOADS.name),
        "peticiones": VENTANA.peticiones,
        "fallidas": VENTANA.fallidas,
        "predicciones_exitosas": VENTANA.predicciones_exitosas,
        # Mismas cifras vistas por Locust (pueden diferir en las peticiones en vuelo al reiniciar)
        "locust_peticiones": total.num_requests,
        "locust_fallidas": total.num_failures,
    }
    Path(VENTANA_JSON).write_text(json.dumps(resumen, indent=2), encoding="utf-8")
