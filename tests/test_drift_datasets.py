"""Conjuntos del monitoreo de P1: ventanas sin fuga, mismo esquema y rezagos causales.

Usa viajes y catálogos sintéticos; no descarga nada ni requiere MLflow.
"""

import numpy as np
import pandas as pd
import pytest

from mibici import dataset
from mibici.config import FECHA_FIN, FECHA_INICIO, INICIO_VALIDACION, PERIODOS_MONITOREO
from mibici.features import preparar_datos_p1
from mibici.monitoring import datasets as mon

ACTUAL_CORTO = {**PERIODOS_MONITOREO, "actual": ("2026-10-01", "2026-10-03")}


def viajes(dias, estaciones=(1, 2, 3), por_dia=3, semilla=16):
    """Viajes válidos de 10 minutos que salen de cada estación hacia la siguiente."""
    azar = np.random.default_rng(semilla)
    filas = []
    for dia in pd.to_datetime(dias):
        for origen in estaciones:
            for _ in range(int(azar.integers(1, por_dia + 1))):
                inicio = dia + pd.Timedelta(hours=int(azar.integers(6, 22)))
                destino = estaciones[(estaciones.index(origen) + 1) % len(estaciones)]
                filas.append((inicio, inicio + pd.Timedelta(minutes=10), origen, destino))
    datos = pd.DataFrame(filas, columns=["inicio", "fin", "origen_id", "destino_id"])
    datos.insert(0, "viaje_id", np.arange(1, len(datos) + 1, dtype="int64"))
    return datos.astype({"origen_id": "int32", "destino_id": "int32"})


def catalogo(capacidades=(15, 20, 12)):
    n = len(capacidades)
    return pd.DataFrame(
        {
            "estacion_id": pd.array(range(1, n + 1), dtype="int32"),
            "latitud": np.linspace(20.64, 20.74, n),
            "longitud": np.linspace(-103.42, -103.30, n),
            "capacidad": pd.array(capacidades, dtype="int32"),
        }
    )


@pytest.fixture
def archivos_mensuales(tmp_path):
    """Septiembre (historial) y los primeros días de octubre (mes de monitoreo) en Parquet."""
    historial = tmp_path / "viajes_2026_09.parquet"
    actual = tmp_path / "viajes_2026_10.parquet"
    viajes(pd.date_range("2026-09-01", "2026-09-30")).to_parquet(historial, index=False)
    viajes(pd.date_range("2026-10-01", "2026-10-03"), semilla=7).to_parquet(actual, index=False)
    return historial, actual


# --- ventanas y reglas sin fuga ---------------------------------------------------------------


def test_los_periodos_del_proyecto_son_consecutivos_y_la_referencia_es_el_train():
    fechas = mon.validar_periodos()
    assert fechas["referencia"] == (
        pd.Timestamp(FECHA_INICIO),
        pd.Timestamp(INICIO_VALIDACION) - pd.Timedelta(days=1),
    )
    assert fechas["reciente"][1] == pd.Timestamp(FECHA_FIN)
    assert fechas["reciente"][1] < fechas["actual"][0]


@pytest.mark.limite
@pytest.mark.parametrize(
    ("cambio", "mensaje"),
    [
        ({"actual": ("2026-09-15", "2026-10-31")}, "Fuga temporal"),
        ({"reciente": ("2026-03-01", "2026-09-30")}, "Fuga temporal"),
        ({"actual": ("2026-10-31", "2026-10-01")}, "posterior al fin"),
        (
            {"referencia": ("2024-10-01", "2026-06-30"), "reciente": ("2026-07-01", "2026-09-30")},
            "entrenamiento del champion",
        ),
    ],
)
def test_periodos_con_traslape_o_mal_definidos_se_rechazan(cambio, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        mon.validar_periodos({**PERIODOS_MONITOREO, **cambio})


@pytest.mark.limite
def test_faltar_una_ventana_o_cambiar_su_orden_se_rechaza():
    with pytest.raises(ValueError, match="ventanas"):
        mon.validar_periodos({k: PERIODOS_MONITOREO[k] for k in ("referencia", "actual")})
    invertido = {k: PERIODOS_MONITOREO[k] for k in ("actual", "reciente", "referencia")}
    with pytest.raises(ValueError, match="ventanas"):
        mon.validar_periodos(invertido)


@pytest.mark.limite
def test_validar_sin_fuga_bloquea_entrenamiento_que_llega_a_la_evaluacion():
    entrenamiento = pd.DataFrame({"fecha": pd.to_datetime(["2026-09-29", "2026-09-30"])})
    evaluacion = pd.DataFrame({"fecha": pd.to_datetime(["2026-10-01"])})
    mon.validar_sin_fuga(entrenamiento, evaluacion)
    with pytest.raises(ValueError, match="Fuga temporal"):
        mon.validar_sin_fuga(
            entrenamiento, pd.DataFrame({"fecha": pd.to_datetime(["2026-09-30"])})
        )
    with pytest.raises(ValueError, match="vacíos"):
        mon.validar_sin_fuga(entrenamiento.iloc[:0], evaluacion)


# --- referencia, reciente y actual ------------------------------------------------------------


def test_referencia_es_el_train_del_champion_y_reciente_val_mas_test(conjuntos):
    p1 = conjuntos["p1"]
    referencia, reciente = mon.construir_referencia_y_reciente(p1)
    assert list(referencia.columns) == mon.COLUMNAS
    assert len(referencia) == len(p1["train"])
    assert len(reciente) == len(p1["val"]) + len(p1["test"])
    assert referencia["fecha"].max() < reciente["fecha"].min()
    esperado = p1["train"][mon.COLUMNAS].sort_values(mon.COLUMNAS_ID).reset_index(drop=True)
    pd.testing.assert_frame_equal(referencia, esperado)


def test_actual_usa_el_pipeline_de_p1_y_rezagos_del_mes_anterior(archivos_mensuales):
    historial, actual = archivos_mensuales
    estaciones = catalogo()
    datos = mon.construir_actual([historial], [actual], estaciones, ACTUAL_CORTO)
    assert list(datos.columns) == mon.COLUMNAS
    assert datos["fecha"].min() == pd.Timestamp("2026-10-01")
    assert datos["fecha"].max() == pd.Timestamp("2026-10-03")
    assert len(datos) == 3 * 3  # 3 estaciones × 3 días
    # El rezago del 1 de octubre es la demanda observada el 30 de septiembre
    septiembre = pd.read_parquet(historial)
    salidas_30 = septiembre.loc[septiembre["inicio"].dt.day == 30].groupby("origen_id").size()
    primero = datos.loc[datos["fecha"] == "2026-10-01"].set_index("estacion_id")
    assert primero["demanda_lag_1"].to_dict() == salidas_30.astype(float).to_dict()
    # Variables de estación tomadas del catálogo vigente
    assert primero.loc[1, "capacidad"] == 15 and primero["demanda_media_7"].notna().all()


def test_las_variables_de_un_dia_no_dependen_de_dias_posteriores(archivos_mensuales, tmp_path):
    historial, actual = archivos_mensuales
    base = mon.construir_actual([historial], [actual], catalogo(), ACTUAL_CORTO)
    alterado = pd.read_parquet(actual)
    extra = viajes(["2026-10-03"], por_dia=20, semilla=99)  # mucha más demanda el día 3
    extra["viaje_id"] += 10_000
    ruta = tmp_path / "otro" / "viajes_2026_10.parquet"
    ruta.parent.mkdir()
    pd.concat([alterado, extra]).to_parquet(ruta, index=False)
    nuevo = mon.construir_actual([historial], [ruta], catalogo(), ACTUAL_CORTO)
    hasta_2 = base["fecha"] <= "2026-10-02"
    pd.testing.assert_frame_equal(base[hasta_2], nuevo[hasta_2])
    assert not base.equals(nuevo)


def test_los_tres_conjuntos_comparten_columnas_y_tipos(conjuntos, archivos_mensuales):
    historial, actual = archivos_mensuales
    referencia, reciente = mon.construir_referencia_y_reciente(conjuntos["p1"])
    nuevo = mon.construir_actual([historial], [actual], catalogo(), ACTUAL_CORTO)
    assert list(referencia.columns) == list(reciente.columns) == list(nuevo.columns)
    assert set(mon.CATEGORICAS) | set(mon.NUMERICAS) == set(mon.VARIABLES)
    for columna in ("dia_semana", "capacidad", "latitud", "demanda_lag_1"):
        assert pd.api.types.is_numeric_dtype(nuevo[columna])
        assert pd.api.types.is_numeric_dtype(referencia[columna])


def test_preparar_datos_p1_acepta_carpeta_o_lista_de_archivos(archivos_mensuales):
    historial, actual = archivos_mensuales
    desde_carpeta = preparar_datos_p1(historial.parent, catalogo(), "2026-09-01", "2026-10-03")
    desde_lista = preparar_datos_p1([actual, historial], catalogo(), "2026-09-01", "2026-10-03")
    pd.testing.assert_frame_equal(desde_carpeta, desde_lista)


# --- calidad, catálogo, huella y carga --------------------------------------------------------


def test_calidad_cuenta_duplicados_fuera_de_catalogo_y_duraciones_invalidas(tmp_path):
    datos = viajes(["2026-10-01"], por_dia=1)
    malos = datos.iloc[[0, 0, 0]].copy()
    malos.iloc[1, malos.columns.get_loc("viaje_id")] = 900
    malos.iloc[1, malos.columns.get_loc("origen_id")] = 77  # estación inexistente
    malos.iloc[2, malos.columns.get_loc("viaje_id")] = 901
    malos.iloc[2, malos.columns.get_loc("fin")] = malos.iloc[2]["inicio"]  # duración cero
    ruta = tmp_path / "viajes_2026_10.parquet"
    pd.concat([datos, malos]).to_parquet(ruta, index=False)
    calidad = mon.calidad_viajes([ruta], catalogo())
    assert calidad == {
        "viajes": len(datos) + 3,
        "filas_duplicadas": 1,
        "fuera_de_catalogo": 1,
        "duracion_invalida": 1,
        "viajes_limpios": len(datos),
    }


def test_cambios_de_catalogo_detectan_altas_bajas_y_capacidad():
    anterior = catalogo((15, 20, 12))
    vigente = catalogo((15, 25, 12, 30)).iloc[[0, 1, 3]]  # baja la 3, alta la 4, cambia la 2
    cambios = mon.cambios_catalogo(anterior, vigente)
    assert cambios["altas"] == [4] and cambios["bajas"] == [3]
    assert cambios["cambios_capacidad"] == [{"estacion_id": 2, "antes": 20, "ahora": 25}]


def test_la_huella_es_determinista_y_detecta_cambios(conjuntos):
    referencia, _ = mon.construir_referencia_y_reciente(conjuntos["p1"])
    assert mon.huella(referencia) == mon.huella(referencia.copy())
    alterada = referencia.copy()
    alterada.loc[0, "demanda"] += 1
    assert mon.huella(alterada) != mon.huella(referencia)


@pytest.mark.limite
def test_cargar_sin_generar_indica_el_comando(tmp_path, monkeypatch):
    monkeypatch.setattr(mon, "MONITOREO_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="make monitoring-data"):
        mon.cargar_conjuntos()


def test_el_challenger_solo_recibe_datos_anteriores_a_la_evaluacion(conjuntos, archivos_mensuales):
    referencia, reciente = mon.construir_referencia_y_reciente(conjuntos["p1"])
    historial, actual = archivos_mensuales
    nuevo = mon.construir_actual([historial], [actual], catalogo(), ACTUAL_CORTO)
    entrenamiento = mon.datos_entrenamiento_challenger(
        {"referencia": referencia, "reciente": reciente, "actual": nuevo}
    )
    assert len(entrenamiento) == len(referencia) + len(reciente)
    with pytest.raises(ValueError, match="Fuga temporal"):
        mon.datos_entrenamiento_challenger(
            {"referencia": referencia, "reciente": nuevo, "actual": nuevo}
        )


def test_procesar_estaciones_puede_escribir_un_catalogo_aparte(tmp_path, monkeypatch):
    original = tmp_path / "estaciones.parquet"
    monkeypatch.setattr(dataset, "ESTACIONES_PARQUET", original)
    csv = tmp_path / "nomenclatura.csv"
    pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["(GDL-001) Uno", "ex-(ZPN-002) Dos"],
            "latitude": [20.6, 20.7],
            "longitude": [-103.3, -103.4],
            "dpcapacity": [15, 20],
        }
    ).to_csv(csv, index=False)
    destino = tmp_path / "monitoreo" / "estaciones_2026_10.parquet"
    dataset.procesar_estaciones(csv, destino)
    assert destino.exists() and not original.exists()
    assert pd.read_parquet(destino)["activa"].tolist() == [True, False]
