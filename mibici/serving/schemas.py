"""Contrato HTTP de P1, basado en docs/serving/contrato_p1.md de CH1.

Los límites observados en train no son límites de dominio. Se validan calendario,
conteos no negativos, capacidad positiva y coordenadas geográficas válidas.
"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Firma del champion features-v1: también se verifica al cargar para no servir
# silenciosamente un modelo con otro contrato (por ejemplo, features-v2).
FIRMA_P1 = (
    ("dia_semana", "integer", True),
    ("mes", "integer", True),
    ("es_fin_semana", "long", True),
    ("dia_anio", "integer", True),
    ("es_festivo", "long", True),
    ("demanda_lag_1", "double", False),
    ("demanda_lag_7", "double", False),
    ("demanda_media_7", "double", False),
    ("capacidad", "integer", True),
    ("latitud", "double", True),
    ("longitud", "double", True),
)

Conteo = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Prediccion = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class EsquemaEstricto(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class RegistroP1(EsquemaEstricto):
    """Variables del día a predecir; los rezagos solo usan días anteriores."""

    dia_semana: int = Field(ge=0, le=6)
    mes: int = Field(ge=1, le=12)
    es_fin_semana: int = Field(ge=0, le=1)
    dia_anio: int = Field(ge=1, le=366)
    es_festivo: int = Field(ge=0, le=1)
    demanda_lag_1: Conteo | None = None
    demanda_lag_7: Conteo | None = None
    demanda_media_7: Conteo | None = None
    capacidad: int = Field(gt=0, le=2**31 - 1)
    latitud: float = Field(ge=-90, le=90)
    longitud: float = Field(ge=-180, le=180)

    @model_validator(mode="after")
    def fin_semana_coherente(self) -> Self:
        if self.es_fin_semana != int(self.dia_semana >= 5):
            raise ValueError("es_fin_semana debe corresponder a dia_semana")
        return self


class PeticionPrediccion(EsquemaEstricto):
    registros: list[RegistroP1] = Field(min_length=1)


class ModeloPublico(EsquemaEstricto):
    """Lista permitida de metadatos: nunca exporta URI, rutas ni tags del Registry."""

    nombre: str | None = None
    version: str | None = None
    alias: str | None = None
    run_id: str | None = None


class RespuestaPrediccion(EsquemaEstricto):
    predicciones: list[Prediccion] = Field(min_length=1)
    modelo: ModeloPublico


class RespuestaSalud(EsquemaEstricto):
    estado: Literal["listo", "no_disponible"]
    modelo: ModeloPublico | None = None
