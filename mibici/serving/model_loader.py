"""Carga del modelo en producción y predicción con su contrato de entrada (semana 8).

El servicio usa el champion de P1 del Model Registry y lo obtiene de una de dos fuentes:

1. **Copia local** (`MODEL_PATH`, por defecto `model_bundle/`): la carpeta MLflow que `exportar`
   descarga del registry por nombre y alias. Se incorpora a la imagen del contenedor, así la
   inferencia no depende del servidor de MLflow ni de sus artefactos. Es la opción recomendada.
2. **Registry en línea** (`MODEL_URI`, por defecto `models:/mibici-equipo16-p1-demanda@champion`)
   en el servidor de `MLFLOW_TRACKING_URI`, cuando no hay copia local.

En ambos casos el alias se resuelve a una versión concreta y se registra en los metadatos, para que
el servicio informe exactamente qué versión responde. `ModeloServido.predecir` aplica la firma
registrada: columnas en el orden del modelo, tipos y obligatoriedad. Solo las variables opcionales
de la firma (los rezagos de demanda) admiten nulos.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import numbers
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import numpy as np
import pandas as pd

from mibici.config import METADATOS_MODELO, MLFLOW_TRACKING_URI, MODEL_PATH, MODEL_URI

# Tipos de columna de la firma de MLflow y su dtype de pandas
TIPOS_FIRMA = {
    "integer": "int32",
    "long": "int64",
    "float": "float32",
    "double": "float64",
}
LIMITES_ENTEROS = {"int32": np.iinfo(np.int32), "int64": np.iinfo(np.int64)}


class ErrorEntrada(ValueError):
    """Los datos recibidos no cumplen el contrato de entrada del modelo."""


class ModeloNoDisponible(RuntimeError):
    """No hay copia local del modelo y tampoco se pudo cargar desde el registry."""


@dataclass(frozen=True)
class Variable:
    """Una columna de la firma del modelo: nombre, tipo de MLflow y si admite nulos."""

    nombre: str
    tipo: str
    requerida: bool


@dataclass
class ModeloServido:
    """Modelo pyfunc cargado, sus metadatos de linaje y su contrato de entrada."""

    modelo: object
    metadatos: dict
    variables: list[Variable]

    @property
    def columnas(self) -> list[str]:
        return [v.nombre for v in self.variables]

    def contrato(self) -> list[dict]:
        """Variables de entrada en el orden del modelo, para documentar y validar la API."""
        return [
            {"nombre": v.nombre, "tipo": v.tipo, "requerida": v.requerida} for v in self.variables
        ]

    def ejemplo(self) -> list[dict]:
        """Ejemplo de entrada registrado con el modelo; los nulos se devuelven como None."""
        ejemplo = getattr(self.modelo, "input_example", None)
        if ejemplo is None:
            return []
        datos = pd.DataFrame(ejemplo)[self.columnas].astype(object)
        return datos.where(datos.notna(), None).to_dict("records")

    def predecir(self, registros) -> list[float]:
        """Predice uno o varios registros y devuelve números JSON (float finitos), en orden."""
        entrada = preparar_entrada(registros, self.variables)
        salida = np.asarray(self.modelo.predict(entrada), dtype=float).ravel()
        if len(salida) != len(entrada):
            raise RuntimeError(f"El modelo devolvió {len(salida)} valores para {len(entrada)}")
        if not np.isfinite(salida).all():
            raise RuntimeError("El modelo devolvió valores no finitos")
        return [float(valor) for valor in salida]


def variables_de_firma(modelo) -> list[Variable]:
    """Lee la firma de entrada del modelo; sin firma no hay contrato que validar."""
    esquema = modelo.metadata.get_input_schema()
    if esquema is None or not esquema.has_input_names():
        raise ModeloNoDisponible("El modelo no tiene firma de entrada con nombres de columna")
    return [Variable(c.name, c.type.name, c.required) for c in esquema.inputs]


def _invalidos(valores: pd.Series) -> list:
    """Índices con valores que no son números reales (texto, booleanos, listas...)."""
    return [
        i
        for i, x in valores.items()
        if not (x is None or x is pd.NA)
        and (isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real))
    ]


def _convertir(valores: pd.Series, variable: Variable) -> pd.Series:
    """Convierte una columna al tipo de la firma o explica por qué no cumple el contrato."""
    nombre, dtype = variable.nombre, TIPOS_FIRMA.get(variable.tipo)
    if dtype is None:
        raise ErrorEntrada(f"{nombre}: tipo de firma no soportado ({variable.tipo})")
    if pd.api.types.is_bool_dtype(valores):
        raise ErrorEntrada(f"{nombre}: se esperaba un número, no un booleano")
    # pandas 3 guarda el texto con dtype str, no object: se revisa todo lo que no sea numérico
    if not pd.api.types.is_numeric_dtype(valores) and (malos := _invalidos(valores)):
        raise ErrorEntrada(f"{nombre}: se esperaba un número en los registros {malos}")
    numeros = pd.to_numeric(valores.astype(object).where(valores.notna(), np.nan))
    numeros = numeros.astype("float64")
    nulos = numeros.isna()
    if nulos.any() and (variable.requerida or dtype in LIMITES_ENTEROS):
        raise ErrorEntrada(
            f"{nombre}: es obligatoria y falta en los registros {list(nulos[nulos].index)}"
        )
    if np.isinf(numeros).any():
        raise ErrorEntrada(f"{nombre}: no admite valores infinitos")
    if dtype in LIMITES_ENTEROS:
        limites = LIMITES_ENTEROS[dtype]
        if (numeros % 1 != 0).any():
            raise ErrorEntrada(f"{nombre}: se esperaba un número entero")
        if (numeros < limites.min).any() or (numeros > limites.max).any():
            raise ErrorEntrada(f"{nombre}: fuera del rango de {dtype}")
    return numeros.astype(dtype)


def preparar_entrada(registros, variables: list[Variable]) -> pd.DataFrame:
    """Valida registros (dict, lista de dicts o DataFrame) y los ordena como la firma del modelo.

    Rechaza columnas faltantes o no esperadas, valores no numéricos, nulos en variables
    obligatorias, infinitos y decimales en variables enteras. Las variables opcionales que no
    vienen se completan con nulos, igual que en el entrenamiento.
    """
    if isinstance(registros, dict):
        registros = [registros]
    if isinstance(registros, pd.DataFrame):
        datos = registros.reset_index(drop=True)
    else:
        registros = list(registros)
        if not all(isinstance(r, dict) for r in registros):
            raise ErrorEntrada("Cada registro debe ser un objeto con las variables del modelo")
        datos = pd.DataFrame.from_records(registros)
    if datos.empty:
        raise ErrorEntrada("Se requiere al menos un registro")
    esperadas = [v.nombre for v in variables]
    faltantes = [v.nombre for v in variables if v.requerida and v.nombre not in datos]
    sobrantes = [c for c in datos.columns if c not in esperadas]
    if faltantes or sobrantes:
        raise ErrorEntrada(
            f"Variables faltantes: {faltantes}; variables no esperadas: {sobrantes}"
        )
    columnas = {
        v.nombre: _convertir(
            datos[v.nombre] if v.nombre in datos else pd.Series(np.nan, index=datos.index), v
        )
        for v in variables
    }
    return pd.DataFrame(columnas)[esperadas]


def resolver_version(model_uri: str):
    """Versión del registry a la que apunta `models:/<nombre>@<alias>` o `models:/<nombre>/<n>`."""
    referencia = model_uri.removeprefix("models:/")
    if referencia == model_uri or not referencia:
        raise ValueError(f"Se esperaba una URI models:/<nombre>@<alias> o /<versión>: {model_uri}")
    cliente = MlflowClient()
    if "@" in referencia:
        nombre, alias = referencia.split("@", 1)
        return cliente.get_model_version_by_alias(nombre, alias), alias
    nombre, _, version = referencia.rpartition("/")
    if not nombre or not version.isdigit():
        raise ValueError(f"Versión inválida en {model_uri}")
    return cliente.get_model_version(nombre, version), None


def metadatos_version(version, alias: str | None) -> dict:
    """Nombre, versión, alias y linaje (corrida, commit, versión de datos) de una versión."""
    return {
        "nombre": version.name,
        "version": str(version.version),
        "alias": alias,
        "uri": f"models:/{version.name}@{alias}" if alias else None,
        "uri_version": f"models:/{version.name}/{version.version}",
        "run_id": version.run_id,
        "linaje": dict(version.tags),
    }


def _metadatos_copia(ruta: Path, modelo) -> dict:
    archivo = ruta / METADATOS_MODELO
    metadatos = json.loads(archivo.read_text(encoding="utf-8")) if archivo.is_file() else {}
    metadatos.setdefault("run_id", modelo.metadata.run_id)
    return {**metadatos, "origen": "copia_local"}


def cargar_modelo(
    model_path: Path | str | None = None, model_uri: str | None = None
) -> ModeloServido:
    """Carga la copia local si existe; si no, el modelo del registry. Ver el docstring del módulo."""
    ruta = Path(model_path) if model_path is not None else MODEL_PATH
    if (ruta / "MLmodel").is_file():
        modelo = mlflow.pyfunc.load_model(str(ruta))
        metadatos = _metadatos_copia(ruta, modelo)
    else:
        uri = model_uri or MODEL_URI
        logger.info(f"No hay copia del modelo en {ruta}; se carga {uri} del registry")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        try:
            version, alias = resolver_version(uri)
            modelo = mlflow.pyfunc.load_model(f"models:/{version.name}/{version.version}")
        except (MlflowException, OSError) as error:
            raise ModeloNoDisponible(
                f"No hay copia del modelo en {ruta} y no se pudo cargar {uri}"
            ) from error
        metadatos = {**metadatos_version(version, alias), "origen": "registry"}
    servido = ModeloServido(modelo, metadatos, variables_de_firma(modelo))
    logger.info(
        f"Modelo {metadatos.get('nombre')} v{metadatos.get('version')} cargado "
        f"({metadatos['origen']})"
    )
    return servido


def _es_copia_previa(destino: Path) -> bool:
    """Solo se reemplaza una carpeta vacía o una copia anterior del modelo."""
    return not any(destino.iterdir()) or (destino / "MLmodel").is_file()


def exportar(destino: Path | str | None = None, model_uri: str | None = None) -> dict:
    """Descarga del registry la versión a la que apunta el alias y la deja lista para la imagen.

    Fija la versión resuelta (no el alias) para que un cambio posterior del champion no altere en
    silencio una imagen ya construida; escribe el linaje en `modelo_servido.json` y verifica que la
    copia carga y predice su ejemplo de entrada antes de reemplazar la anterior.
    """
    destino = Path(destino) if destino is not None else MODEL_PATH
    if destino.exists() and not _es_copia_previa(destino):
        raise ValueError(f"{destino} existe y no es una copia del modelo; elija otra carpeta")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    version, alias = resolver_version(model_uri or MODEL_URI)
    metadatos = metadatos_version(version, alias)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destino.parent) as temporal:
        copia = Path(temporal) / "modelo"
        mlflow.artifacts.download_artifacts(
            artifact_uri=metadatos["uri_version"], dst_path=str(copia)
        )
        metadatos["exportado_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
        (copia / METADATOS_MODELO).write_text(
            json.dumps(metadatos, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        servido = cargar_modelo(copia)
        ejemplo = servido.ejemplo()
        if ejemplo:
            servido.predecir(ejemplo)
        if destino.exists():
            shutil.rmtree(destino)
        shutil.move(str(copia), str(destino))
    return metadatos
