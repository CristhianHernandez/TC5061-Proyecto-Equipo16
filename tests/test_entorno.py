"""El entorno de calidad (requirements-dev.txt) funciona con las versiones fijas del proyecto.

Si las herramientas de requirements-dev.txt no están instaladas, estas pruebas se omiten con un
aviso en lugar de romper la suite: el resto de las pruebas solo necesita requirements.txt.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from mibici.compat import importar_deepchecks, preparar_deepchecks


def requiere(paquete):
    """Omite la prueba si falta una herramienta de requirements-dev.txt."""
    if importlib.util.find_spec(paquete) is None:
        pytest.skip(f"{paquete} no está instalado: python -m pip install -r requirements-dev.txt")


def test_ajustes_de_compatibilidad_no_cambian_valores_y_son_idempotentes():
    from sklearn.metrics import get_scorer

    preparar_deepchecks()
    preparar_deepchecks()
    assert np.Inf == np.inf and np.NINF == -np.inf and np.isnan(np.NaN)
    # max_error como scorer: más error es peor, así que el puntaje es el error con signo negativo
    scorer = get_scorer("max_error")
    assert scorer._score_func([1, 2], [1, 5]) == 3


@pytest.mark.parametrize("paquete", ["bandit", "pip_audit"])
def test_herramientas_de_seguridad_instaladas(paquete):
    requiere(paquete)


@pytest.mark.deepchecks
def test_deepchecks_importa_y_detecta_problemas_en_datos_de_ingesta(viajes_sinteticos):
    requiere("deepchecks")
    importar_deepchecks()
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.checks import DataDuplicates, PercentOfNulls

    limpios = Dataset(viajes_sinteticos, cat_features=["origen_id", "destino_id"])
    assert DataDuplicates().run(limpios).value == 0

    # Un destino faltante (NaN en float64, no pd.NA) y 4 filas duplicadas
    alterados = viajes_sinteticos.astype({"destino_id": "float64"})
    alterados.loc[0, "destino_id"] = np.nan
    alterados = pd.concat([alterados, alterados.iloc[:4]], ignore_index=True)
    dataset = Dataset(alterados, cat_features=["origen_id", "destino_id"])
    assert DataDuplicates().run(dataset).value == pytest.approx(4 / len(alterados))
    nulos = PercentOfNulls().run(dataset).value
    assert nulos.loc["destino_id"].iloc[0] > 0


@pytest.mark.deepchecks
def test_tipos_nullable_de_pandas_se_convierten_antes_de_deepchecks(viajes_sinteticos):
    """deepchecks 0.19.1 falla con pd.NA; a_tipos_numpy lo convierte en NaN sin perder valores."""
    requiere("deepchecks")
    importar_deepchecks()
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.checks import PercentOfNulls

    from mibici.compat import a_tipos_numpy

    nullable = viajes_sinteticos.astype({"destino_id": "Int32"})
    nullable.loc[0, "destino_id"] = pd.NA
    convertidos = a_tipos_numpy(nullable)
    assert convertidos["destino_id"].dtype == "float64"
    assert convertidos["destino_id"].isna().sum() == 1
    pd.testing.assert_series_equal(
        convertidos["destino_id"].iloc[1:], nullable["destino_id"].iloc[1:].astype("float64")
    )
    nulos = PercentOfNulls().run(Dataset(convertidos, cat_features=["origen_id"])).value
    assert nulos.loc["destino_id"].iloc[0] > 0
