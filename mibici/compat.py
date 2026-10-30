"""Compatibilidad de herramientas de validación con las versiones fijas del proyecto.

deepchecks 0.19.1 (última versión publicada) se escribió para NumPy 1 y scikit-learn < 1.6. Con las
versiones del proyecto (NumPy 2.5, scikit-learn 1.9, pandas 3.0 y Python 3.13) falla al importarse
por dos nombres que ya no existen. `importar_deepchecks` los restituye sin cambiar ningún cálculo:

- alias de NumPy 1 retirados en NumPy 2 (`np.Inf`, `np.NINF`, `np.NaN`), que se agregan apuntando a
  los valores actuales (`np.inf`, `-np.inf`, `np.nan`);
- el scorer `max_error` que scikit-learn retiró de su registro; la función `max_error` sigue
  existiendo y se registra con ella.

Verificado con datos sintéticos: funcionan los chequeos de integridad (nulos, nulos mixtos,
duplicados, tipos mixtos, valor único, etiquetas en conflicto, inconsistencias de texto) y de drift
(FeatureDrift, LabelDrift). Dentro de las suites completas fallan algunos chequeos por cambios de
pandas 3 (`OutlierSampleDetection`, `TrainTestSamplesMix`) y de scikit-learn (`MultivariateDrift`):
conviene ejecutar los chequeos individuales.

Tampoco acepta los tipos *nullable* de pandas (`Int32`, `boolean`, `string` con `pd.NA`): compara
contra `pd.NA` y pandas lo considera ambiguo. `a_tipos_numpy` convierte esas columnas a tipos de
NumPy (`float64` con `NaN`, `object`) antes de construir el `Dataset`.
"""

import pandas as pd

# Alias de NumPy 1 que deepchecks 0.19.1 usa en sus valores por defecto
_ALIAS_NUMPY = {"Inf": "inf", "NINF": None, "NaN": "nan"}


def preparar_deepchecks() -> None:
    """Restituye los nombres que deepchecks 0.19.1 necesita al importarse (idempotente)."""
    import numpy as np
    from sklearn.metrics import _scorer, make_scorer, max_error

    for alias, actual in _ALIAS_NUMPY.items():
        if alias not in np.__dict__:
            setattr(np, alias, -np.inf if actual is None else getattr(np, actual))
    if "max_error" not in _scorer._SCORERS:
        _scorer._SCORERS["max_error"] = make_scorer(max_error, greater_is_better=False)


def importar_deepchecks():
    """Importa deepchecks después de aplicar los ajustes de compatibilidad y devuelve el módulo.

    Uso:
        from mibici.compat import importar_deepchecks
        importar_deepchecks()
        from deepchecks.tabular import Dataset
    """
    preparar_deepchecks()
    import deepchecks

    return deepchecks


def a_tipos_numpy(datos: pd.DataFrame) -> pd.DataFrame:
    """Copia de `datos` con las columnas nullable de pandas convertidas a tipos de NumPy.

    Enteros y booleanos nullable pasan a `float64` (los faltantes quedan como `NaN`); textos
    nullable pasan a `object` (los faltantes quedan como `None`). Las demás columnas no cambian.
    """
    convertidos = datos.copy()
    for columna, tipo in datos.dtypes.items():
        if isinstance(tipo, pd.api.extensions.ExtensionDtype) and tipo.na_value is pd.NA:
            numerico = pd.api.types.is_numeric_dtype(tipo) or pd.api.types.is_bool_dtype(tipo)
            serie = datos[columna]
            convertidos[columna] = (
                serie.astype("float64") if numerico else serie.astype(object).where(serie.notna())
            )
    return convertidos
