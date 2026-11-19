"""Criterio de éxito de una respuesta de /predict en la prueba de carga (sin dependencias de Locust)."""

import math

MODELO_ESPERADO = "mibici-equipo16-p1-demanda"


def validar_respuesta(
    datos: object, esperadas: int, modelo: str = MODELO_ESPERADO
) -> str | None:
    """Devuelve el motivo del fallo o None si la respuesta 200 cumple el contrato.

    Exitosa = objeto JSON con `predicciones`, una lista de `esperadas` floats finitos > 0, y
    `modelo.nombre` igual al modelo esperado. Cualquier otra forma cuenta como fallida.
    """
    if not isinstance(datos, dict):
        return "la respuesta no es un objeto JSON"
    predicciones = datos.get("predicciones")
    if not isinstance(predicciones, list) or len(predicciones) != esperadas:
        return f"se esperaban {esperadas} predicciones"
    for valor in predicciones:
        if isinstance(valor, bool) or not isinstance(valor, (int, float)):
            return "predicción no numérica"
        if not math.isfinite(valor) or valor <= 0:
            return "predicción no es un float finito > 0"
    info_modelo = datos.get("modelo")
    if not isinstance(info_modelo, dict) or info_modelo.get("nombre") != modelo:
        return "modelo distinto al esperado"
    return None
