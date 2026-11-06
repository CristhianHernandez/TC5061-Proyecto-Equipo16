"""API P1: uvicorn mibici.serving.api:app --host 127.0.0.1 --port 8000.

La importación no carga modelos. Cada proceso carga una vez durante lifespan;
si falla, /health y /predict responden 503 y /docs sigue disponible.
"""

from collections.abc import Callable
from contextlib import asynccontextmanager
import numbers

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
import numpy as np
from starlette.concurrency import run_in_threadpool

from mibici.serving.model_loader import (
    ErrorEntrada,
    ModeloNoDisponible,
    ModeloServido,
    cargar_modelo,
)
from mibici.serving.schemas import (
    FIRMA_P1,
    ModeloPublico,
    PeticionPrediccion,
    RespuestaPrediccion,
    RespuestaSalud,
)

ERROR_NO_DISPONIBLE = "Modelo no disponible. Consulte /health."
ERROR_ENTRADA = "La entrada no cumple el contrato del modelo."
ERROR_PREDICCION = "No se pudo generar una predicción válida."


def _modelo_publico(servido: ModeloServido) -> ModeloPublico:
    return ModeloPublico.model_validate(
        {campo: servido.metadatos.get(campo) for campo in ModeloPublico.model_fields}
    )


def _validar_contrato(servido: ModeloServido) -> None:
    esperado = [
        {"nombre": nombre, "tipo": tipo, "requerida": requerida}
        for nombre, tipo, requerida in FIRMA_P1
    ]
    if servido.contrato() != esperado:
        raise ModeloNoDisponible("La firma cargada no coincide con el contrato HTTP P1")


def _salida_json(salida, cantidad: int) -> list[float]:
    # dtype=object conserva booleanos en listas mixtas antes de convertir a float.
    valores = np.asarray(salida, dtype=object)
    if valores.ndim != 1 or len(valores) != cantidad:
        raise ValueError("El modelo debe devolver un valor por registro, en una dimensión")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real) for v in valores):
        raise ValueError("El modelo debe devolver números reales")
    return [float(v) for v in valores]  # El response_model valida finitud y positividad.


def crear_app(cargador: Callable[[], ModeloServido] | None = None) -> FastAPI:
    """Inyección del loader para pruebas; en producción siempre usa cargar_modelo de CH1."""

    @asynccontextmanager
    async def lifespan(aplicacion: FastAPI):
        aplicacion.state.modelo = None
        aplicacion.state.modelo_publico = None
        try:
            servido = await run_in_threadpool(cargador or cargar_modelo)
            _validar_contrato(servido)
            publico = _modelo_publico(servido)
            aplicacion.state.modelo = servido
            aplicacion.state.modelo_publico = publico
        except Exception as error:  # noqa: BLE001 - aislar fallos del loader en readiness
            # No se interpolan mensajes que puedan contener URIs, credenciales o rutas.
            logger.warning("Modelo P1 no disponible al iniciar ({})", type(error).__name__)
        try:
            yield
        finally:
            aplicacion.state.modelo = None
            aplicacion.state.modelo_publico = None

    aplicacion = FastAPI(
        title="MiBici · Predicción de demanda P1",
        description="Viajes esperados por estación y día. Los rezagos provienen del pasado.",
        lifespan=lifespan,
    )
    aplicacion.state.modelo = None
    aplicacion.state.modelo_publico = None

    @aplicacion.exception_handler(RequestValidationError)
    async def entrada_invalida(request: Request, error: RequestValidationError):
        # Omitir input/ctx evita reemitir payloads y serializar NaN/inf en los errores.
        detalles = [
            {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in error.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": detalles})

    @aplicacion.get(
        "/health", response_model=RespuestaSalud, responses={503: {"model": RespuestaSalud}}
    )
    def health(request: Request, response: Response):
        publico = request.app.state.modelo_publico
        if request.app.state.modelo is None or publico is None:
            response.status_code = 503
            return RespuestaSalud(estado="no_disponible")
        return RespuestaSalud(estado="listo", modelo=publico)

    @aplicacion.post(
        "/predict",
        response_model=RespuestaPrediccion,
        responses={
            422: {"description": "Entrada inválida"},
            500: {"description": "Error de inferencia o salida inválida"},
            503: {"description": "Modelo no disponible"},
        },
    )
    def predict(peticion: PeticionPrediccion, request: Request):
        servido = request.app.state.modelo
        publico = request.app.state.modelo_publico
        if servido is None or publico is None:
            raise HTTPException(status_code=503, detail=ERROR_NO_DISPONIBLE)
        registros = [registro.model_dump() for registro in peticion.registros]
        try:
            salida = servido.predecir(registros)
            return RespuestaPrediccion(
                predicciones=_salida_json(salida, len(registros)),
                modelo=publico,
            )
        except ErrorEntrada:
            raise HTTPException(status_code=422, detail=ERROR_ENTRADA) from None
        except ModeloNoDisponible:
            request.app.state.modelo = None
            request.app.state.modelo_publico = None
            raise HTTPException(status_code=503, detail=ERROR_NO_DISPONIBLE) from None
        except Exception as error:  # noqa: BLE001 - frontera HTTP sin excepciones internas
            logger.warning("Fallo de inferencia P1 ({})", type(error).__name__)
            raise HTTPException(status_code=500, detail=ERROR_PREDICCION) from None

    return aplicacion


app = crear_app()
