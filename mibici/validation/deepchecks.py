"""Contratos de ingesta, integridad temporal y alertas de drift, sin IO ni entrenamiento.

Requiere requirements-dev.txt. Se ejecutan checks individuales: las suites generales de
DeepChecks 0.19.1 tienen incompatibilidades conocidas con pandas 3 y scikit-learn 1.9.
Los contratos conservan los tipos originales; los checks estadísticos reciben copias
convertidas con mibici.compat. Ningún resultado incluye muestras de viajes.
"""

import os

import numpy as np
import pandas as pd
from pandas.api import types as tipos

from mibici import config
from mibici.compat import a_tipos_numpy, importar_deepchecks

# DeepChecks consulta su versión por red al importar si no se desactiva explícitamente.
os.environ["DISABLE_DEEPCHECKS_ANONYMOUS_TELEMETRY"] = "1"
os.environ["DISABLE_LATEST_VERSION_CHECK"] = "1"
importar_deepchecks()

from deepchecks.core import CheckResult, ConditionResult
from deepchecks.core.condition import ConditionCategory
from deepchecks.tabular import Dataset, SingleDatasetCheck
from deepchecks.tabular.checks import (
    DataDuplicates,
    FeatureDrift,
    MixedDataTypes,
    PercentOfNulls,
)

ESQUEMA_VIAJES = {
    "viaje_id": "entero",
    "inicio": "fecha",
    "fin": "fecha",
    "origen_id": "entero",
    "destino_id": "entero",
}
ESQUEMA_ESTACIONES = {
    "estacion_id": "entero",
    "nombre": "texto",
    "municipio": "texto",
    "latitud": "real",
    "longitud": "real",
    "capacidad": "entero",
    "activa": "booleano",
}


def _sin_incidencias(valores):
    fallos = {regla: n for regla, n in valores.items() if n}
    return ConditionResult(
        ConditionCategory.FAIL if fallos else ConditionCategory.PASS,
        details=str(fallos) if fallos else "Sin incidencias",
    )


class EsquemaColumnas(SingleDatasetCheck):
    """Valida tipos semánticos antes de normalizar nullable para los checks nativos."""

    def __init__(self, esquema, **kwargs):
        super().__init__(**kwargs)
        self.esquema = esquema
        self.add_condition("Columnas requeridas y tipos válidos", _sin_incidencias)

    def run_logic(self, context, dataset_kind):
        datos = context.get_data_by_kind(dataset_kind).data
        errores = {}
        for columna, esperado in self.esquema.items():
            if columna not in datos:
                errores[f"falta:{columna}"] = 1
                continue
            serie = datos[columna]
            dtype = serie.dtype
            valido = {
                "entero": tipos.is_integer_dtype(dtype) and not tipos.is_bool_dtype(dtype),
                "real": tipos.is_float_dtype(dtype),
                "booleano": tipos.is_bool_dtype(dtype),
                "fecha": tipos.is_datetime64_any_dtype(dtype)
                and getattr(dtype, "tz", None) is None,
                "texto": all(isinstance(v, str) for v in serie.dropna()),
            }[esperado]
            errores[f"tipo:{columna}"] = int(not valido)
        return CheckResult(errores)


class IntegridadViajes(SingleDatasetCheck):
    """IDs únicos, referencias y duración conforme a features.limpiar_viajes."""

    def __init__(self, estaciones, **kwargs):
        super().__init__(**kwargs)
        self.estaciones = estaciones
        self.add_condition("Viajes íntegros", _sin_incidencias)

    def run_logic(self, context, dataset_kind):
        datos = context.get_data_by_kind(dataset_kind).data
        ids = self.estaciones["estacion_id"].dropna()
        # La conversión solo permite informar fechas inválidas; el esquema rechaza strings.
        inicio = pd.to_datetime(datos["inicio"], format="mixed", errors="coerce", utc=True)
        fin = pd.to_datetime(datos["fin"], format="mixed", errors="coerce", utc=True)
        minutos = (fin - inicio).dt.total_seconds() / 60
        desde = pd.Timestamp(config.FECHA_INICIO, tz="UTC")
        hasta = pd.Timestamp(config.FECHA_FIN, tz="UTC") + pd.Timedelta(days=1)
        return CheckResult(
            {
                "filas_vacias": int(datos.empty),
                "id_repetido": int(datos["viaje_id"].duplicated().sum()),
                "origen_desconocido": int((~datos["origen_id"].isin(ids)).sum()),
                "destino_desconocido": int((~datos["destino_id"].isin(ids)).sum()),
                "fecha_invalida": int((inicio.isna() | fin.isna()).sum()),
                "inicio_fuera_periodo": int((inicio.lt(desde) | inicio.ge(hasta)).sum()),
                "duracion_invalida": int((minutos.le(0) | minutos.gt(1440)).sum()),
            }
        )


class IntegridadEstaciones(SingleDatasetCheck):
    """Catálogo único; capacidad cero permitida en ingesta, excluida para P2."""

    def __init__(self, para_p2=False, **kwargs):
        super().__init__(**kwargs)
        self.para_p2 = para_p2
        self.add_condition("Catálogo íntegro y capacidad válida", _sin_incidencias)

    def run_logic(self, context, dataset_kind):
        datos = context.get_data_by_kind(dataset_kind).data
        capacidad = pd.to_numeric(datos["capacidad"], errors="coerce")
        invalida = (
            capacidad.isna() | ~np.isfinite(capacidad) | capacidad.lt(0) | capacidad.mod(1).ne(0)
        )
        if self.para_p2:
            invalida |= capacidad.eq(0)
        return CheckResult(
            {
                "filas_vacias": int(datos.empty),
                "id_repetido": int(datos["estacion_id"].duplicated().sum()),
                "capacidad_invalida": int(invalida.sum()),
                "latitud_invalida": int(
                    (~pd.to_numeric(datos["latitud"], errors="coerce").between(-90, 90)).sum()
                ),
                "longitud_invalida": int(
                    (~pd.to_numeric(datos["longitud"], errors="coerce").between(-180, 180)).sum()
                ),
            }
        )


def _dataset(datos):
    # cat_features explícito evita inferencia heurística, especialmente con muestras pequeñas.
    return Dataset(datos.copy(), cat_features=[])


def _validar(datos, esquema, integridad):
    if datos.empty:
        raise ValueError("Se requieren observaciones para validar calidad")
    if not datos.columns.is_unique:
        raise ValueError("Los nombres de columnas deben ser únicos")
    original = _dataset(datos)
    resultado = {"esquema": EsquemaColumnas(esquema).run(original, with_display=False)}
    if not set(esquema).issubset(datos):
        return resultado  # El fallo de esquema informa las columnas ausentes.
    resultado["integridad"] = integridad.run(original, with_display=False)
    normalizado = _dataset(a_tipos_numpy(datos[list(esquema)]))
    resultado["nulos"] = (
        PercentOfNulls(n_samples=len(datos))
        .add_condition_percent_of_nulls_not_greater_than(0)
        .run(normalizado, with_display=False)
    )
    resultado["duplicados"] = (
        DataDuplicates(n_samples=len(datos))
        .add_condition("Sin filas duplicadas", lambda proporcion: proporcion == 0)
        .run(normalizado, with_display=False)
    )
    resultado["tipos_mixtos"] = (
        MixedDataTypes(
            columns=[c for c, tipo in esquema.items() if tipo != "fecha"], n_samples=len(datos)
        )
        .add_condition("Sin mezcla de números y texto", lambda v: not any(v.values()))
        .run(normalizado, with_display=False)
    )
    return resultado


def validar_viajes(viajes, estaciones):
    """Devuelve CheckResults con condiciones; no limpia ni modifica las tablas recibidas."""
    if "estacion_id" not in estaciones:
        raise ValueError("Falta estacion_id en el catálogo")
    return _validar(viajes, ESQUEMA_VIAJES, IntegridadViajes(estaciones))


def validar_estaciones(estaciones, *, para_p2=False):
    """Use para_p2=True únicamente sobre el catálogo filtrado que alimentará P2."""
    return _validar(estaciones, ESQUEMA_ESTACIONES, IntegridadEstaciones(para_p2))


class IntegridadTemporal(SingleDatasetCheck):
    """Comprueba cada fila contra su intervalo asignado y la unicidad de las claves."""

    def __init__(self, columna_fecha, claves, columna_fin=None, **kwargs):
        super().__init__(**kwargs)
        self.columna_fecha = columna_fecha
        self.claves = claves
        self.columna_fin = columna_fin
        self.add_condition("Particiones sin fuga temporal", _sin_incidencias)

    def run_logic(self, context, dataset_kind):
        datos = context.get_data_by_kind(dataset_kind).data
        fechas = pd.to_datetime(datos[self.columna_fecha], errors="coerce", utc=True)
        cortes = pd.to_datetime(
            [config.FECHA_INICIO, config.INICIO_VALIDACION, config.INICIO_PRUEBA], utc=True
        ).tolist() + [pd.Timestamp(config.FECHA_FIN, tz="UTC") + pd.Timedelta(days=1)]
        errores = {"fecha_invalida": int(fechas.isna().sum())}
        for i, nombre in enumerate(("train", "val", "test")):
            mascara = datos["_particion"].eq(nombre)
            errores[f"vacia:{nombre}"] = int(not mascara.any())
            errores[f"fuera_corte:{nombre}"] = int(
                (mascara & (fechas.lt(cortes[i]) | fechas.ge(cortes[i + 1]))).sum()
            )
            if self.columna_fin:
                fin = pd.to_datetime(datos[self.columna_fin], errors="coerce", utc=True)
                errores[f"fin_fuera_corte:{nombre}"] = int(
                    (mascara & (fin.isna() | fin.ge(cortes[i + 1]) | fin.lt(fechas))).sum()
                )
        errores["clave_repetida"] = int(datos.duplicated(list(self.claves)).sum())
        errores["clave_nula"] = int(datos[list(self.claves)].isna().any(axis=1).sum())
        return CheckResult(errores)


def validar_particiones(conjuntos, columna_fecha="fecha", *, claves=None, columna_fin=None):
    """Valida train/val/test de P1/P2 o viajes; cortes inclusivos/exclusivos de config.

    Para viajes use claves=('viaje_id',), columna_fecha='inicio', columna_fin='fin'.
    Una misma estación puede aparecer en varios conjuntos en fechas diferentes.
    Esto verifica asignación y claves; la causalidad de features requiere pruebas propias.
    """
    if set(conjuntos) != {"train", "val", "test"}:
        raise ValueError("Se requieren exactamente train, val y test")
    claves = tuple(claves) if claves is not None else ("estacion_id", columna_fecha)
    requeridas = {columna_fecha, *claves}
    if columna_fin:
        requeridas.add(columna_fin)
    partes = []
    for nombre in ("train", "val", "test"):
        datos = conjuntos[nombre]
        if not requeridas.issubset(datos):
            raise ValueError(f"Faltan columnas temporales en {nombre}: {sorted(requeridas)}")
        columnas_fecha = [columna_fecha] + ([columna_fin] if columna_fin else [])
        for columna in columnas_fecha:
            if not tipos.is_datetime64_any_dtype(datos[columna].dtype):
                raise ValueError(f"{columna} debe tener tipo datetime en {nombre}")
            if getattr(datos[columna].dtype, "tz", None) is not None:
                raise ValueError("Las fechas deben usar la hora local sin zona, como la ingesta")
        partes.append(datos[sorted(requeridas)].assign(_particion=nombre))
    datos = pd.concat(partes, ignore_index=True)
    return IntegridadTemporal(columna_fecha, claves, columna_fin).run(
        _dataset(datos), with_display=False
    )


def evaluar_drift(train, test, columnas, *, umbral=0.2):
    """FeatureDrift con condición WARN: desplazamiento estadístico, no fallo de integridad.

    Selección explícita de predictores numéricos: nunca incluir IDs, fechas ni objetivos.
    No ajusta modelos ni modifica los cortes o variables a partir del conjunto de prueba.
    """
    columnas = list(columnas)
    if not columnas or len(set(columnas)) != len(columnas):
        raise ValueError("Seleccione columnas únicas para drift")
    if not 0 <= umbral <= 1:
        raise ValueError("umbral debe estar entre 0 y 1")
    for datos in (train, test):
        if datos.empty or not set(columnas).issubset(datos):
            raise ValueError("Drift requiere observaciones y las columnas seleccionadas")
        if not all(tipos.is_numeric_dtype(datos[c]) for c in columnas):
            raise ValueError("Drift requiere predictores numéricos")
        if not np.isfinite(a_tipos_numpy(datos[columnas]).to_numpy(dtype=float)).all():
            raise ValueError("Resuelva nulos e infinitos antes de evaluar drift")
    check = FeatureDrift(
        columns=columnas, n_samples=max(len(train), len(test)), random_state=config.SEMILLA
    )

    def alerta(valores):
        columnas_alerta = [c for c, v in valores.items() if v["Drift score"] > umbral]
        return ConditionResult(
            ConditionCategory.WARN if columnas_alerta else ConditionCategory.PASS,
            details=f"Revisar drift en: {columnas_alerta}; no implica fallo del pipeline",
        )

    check.add_condition("Drift como alerta estadística", alerta)
    return check.run(
        _dataset(a_tipos_numpy(train[columnas])),
        _dataset(a_tipos_numpy(test[columnas])),
        with_display=False,
    )


def resumir_resultados(resultados):
    """Resumen JSON de condiciones y métricas agregadas, sin displays ni muestras."""

    def convertir(valor):
        if isinstance(valor, pd.DataFrame):
            return convertir(valor.to_dict(orient="index"))
        if isinstance(valor, dict):
            return {
                str(k): convertir(v) for k, v in valor.items() if not str(k).endswith("_examples")
            }
        if isinstance(valor, (list, tuple)):
            return [convertir(v) for v in valor]
        if isinstance(valor, np.generic):
            return valor.item()
        return valor

    return {
        nombre: {
            "check": type(resultado.check).__name__,
            "valor": convertir(resultado.value),
            "condiciones": [
                {"estado": c.category.name, "detalle": c.details}
                for c in resultado.conditions_results
            ],
        }
        for nombre, resultado in resultados.items()
    }
