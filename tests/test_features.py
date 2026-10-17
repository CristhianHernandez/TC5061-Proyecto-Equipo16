import pandas as pd

from mibici.features import (
    agregar_lag_7,
    agregar_media_movil_7,
    demanda_diaria,
    filtrar_por_primera_actividad,
    agregar_features_calendario,
    agregar_lag_demanda,

)


def test_demanda_diaria():
    viajes = pd.DataFrame({
        "origen_id": [2, 2, 3],
        "inicio": pd.to_datetime([
            "2024-10-01 08:00:00",
            "2024-10-01 12:00:00",
            "2024-10-02 09:00:00",
        ]),
    })

    estaciones = pd.DataFrame({
        "estacion_id": [2, 3],
    })

    resultado = demanda_diaria(
        viajes=viajes,
        estaciones=estaciones,
        fecha_inicio="2024-10-01",
        fecha_fin="2024-10-03",
    )

    assert len(resultado) == 6
    assert resultado["demanda"].sum() == 3

    # Estación 2: dos viajes el primer día
    demanda_estacion_2 = resultado.loc[
        (resultado["estacion_id"] == 2)
        & (resultado["fecha"] == pd.Timestamp("2024-10-01")),
        "demanda",
    ].iloc[0]

    assert demanda_estacion_2 == 2

    # Estación 3: cero viajes el primer día
    demanda_estacion_3 = resultado.loc[
        (resultado["estacion_id"] == 3)
        & (resultado["fecha"] == pd.Timestamp("2024-10-01")),
        "demanda",
    ].iloc[0]

    assert demanda_estacion_3 == 0

def test_filtrar_por_primera_actividad():
    viajes = pd.DataFrame({
        "origen_id": [2, 3],
        "inicio": pd.to_datetime([
            "2024-12-01 08:00:00",
            "2024-12-16 09:00:00",
        ]),
    })

    estaciones = pd.DataFrame({
        "estacion_id": [2, 3],
    })

    actividad = pd.DataFrame({
        "estacion_id": [2, 3],
        "primera_actividad": pd.to_datetime([
            "2024-12-01",
            "2024-12-16",
        ]),
    })

    demanda = demanda_diaria(
        viajes,
        estaciones,
        "2024-12-01",
        "2024-12-31",
    )

    resultado = filtrar_por_primera_actividad(
        demanda,
        actividad,
    )

    # Estación 2: 31 días
    # Estación 3: 16 días
    assert len(resultado) == 47

    assert resultado["demanda"].sum() == 2

    estacion_3 = resultado[
        resultado["estacion_id"] == 3
    ]

    assert estacion_3["fecha"].min() == pd.Timestamp(
        "2024-12-16"
    )

def test_agregar_features_calendario():
    demanda = pd.DataFrame({
        "estacion_id": [2, 2],
        "fecha": pd.to_datetime([
            "2024-10-04",  # Viernes
            "2024-10-05",  # Sábado
        ]),
        "demanda": [50, 30],
    })

    resultado = agregar_features_calendario(demanda)

    assert resultado["dia_semana"].tolist() == [4, 5]
    assert resultado["mes"].tolist() == [10, 10]
    assert resultado["es_fin_semana"].tolist() == [0, 1]
    assert resultado["dia_anio"].tolist() == [278, 279]
    assert resultado["demanda"].tolist() == [50, 30]

def test_agregar_lag_demanda():
    demanda = pd.DataFrame({
        "estacion_id": [2, 2, 2],
        "fecha": pd.to_datetime([
            "2024-10-01",
            "2024-10-02",
            "2024-10-03",
        ]),
        "demanda": [50, 30, 40],
    })

    resultado = agregar_lag_demanda(demanda)

    assert pd.isna(resultado["demanda_lag_1"].iloc[0])
    assert resultado["demanda_lag_1"].iloc[1] == 50
    assert resultado["demanda_lag_1"].iloc[2] == 30

def test_lag_con_fechas_faltantes():
    demanda = pd.DataFrame({
        "estacion_id": [2, 2, 2],
        "fecha": pd.to_datetime([
            "2024-10-01",
            "2024-10-02",
            "2024-10-05",
        ]),
        "demanda": [50, 30, 40],
    })

    resultado = agregar_lag_demanda(demanda)

    assert pd.isna(resultado["demanda_lag_1"].iloc[0])

    assert resultado["demanda_lag_1"].iloc[1] == 50

    # El 4 de octubre no existe en los datos
    assert pd.isna(resultado["demanda_lag_1"].iloc[2])

def test_agregar_lag_7():
    demanda = pd.DataFrame({
        "estacion_id": [2, 2, 2],
        "fecha": pd.to_datetime([
            "2024-10-01",
            "2024-10-08",
            "2024-10-09",
        ]),
        "demanda": [50, 65, 45],
    })

    resultado = agregar_lag_7(demanda)

    assert pd.isna(resultado["demanda_lag_7"].iloc[0])
    assert resultado["demanda_lag_7"].iloc[1] == 50

    # No existe registro del 2 de octubre
    assert pd.isna(resultado["demanda_lag_7"].iloc[2])

def test_agregar_media_movil_7():
    demanda = pd.DataFrame({
        "estacion_id": [2] * 8,
        "fecha": pd.date_range(
            "2024-10-01",
            periods=8,
            freq="D",
        ),
        "demanda": [10, 20, 30, 40, 50, 60, 70, 100],
    })

    resultado = agregar_media_movil_7(demanda)

    # No hay suficiente historial durante los primeros 7 días
    assert resultado["demanda_media_7"].iloc[:7].isna().all()

    # Para el día 8: promedio de 10, 20, ..., 70
    assert resultado["demanda_media_7"].iloc[7] == 40

def test_media_movil_con_fechas_faltantes():
    demanda = pd.DataFrame({
        "estacion_id": [2] * 7,
        "fecha": pd.to_datetime([
            "2024-10-01",
            "2024-10-02",
            "2024-10-03",
            "2024-10-04",
            "2024-10-05",
            "2024-10-07",
            "2024-10-08",
        ]),
        "demanda": [10, 20, 30, 40, 50, 70, 80],
    })

    resultado = agregar_media_movil_7(demanda)

    # Falta el 6 de octubre.
    # No existen siete días completos anteriores al 8.
    assert pd.isna(resultado["demanda_media_7"].iloc[-1])

def test_media_movil_varias_estaciones():
    demanda = pd.DataFrame({
        "estacion_id": [2] * 8 + [3] * 8,
        "fecha": list(pd.date_range(
            "2024-10-01", periods=8
        )) * 2,
        "demanda": [10] * 8 + [100] * 8,
    })

    resultado = agregar_media_movil_7(demanda)

    estacion_2 = resultado[
        resultado["estacion_id"] == 2
    ]

    estacion_3 = resultado[
        resultado["estacion_id"] == 3
    ]

    assert estacion_2["demanda_media_7"].iloc[-1] == 10
    assert estacion_3["demanda_media_7"].iloc[-1] == 100