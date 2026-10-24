import pandas as pd

from mibici.features import (
    agregar_features_calendario,
    agregar_lag_7,
    agregar_lag_demanda,
    agregar_media_movil_7,
    agregar_media_movil_28,
    demanda_diaria,
    filtrar_por_primera_actividad,
)


def test_demanda_diaria():
    viajes = pd.DataFrame(
        {
            "origen_id": [2, 2, 3],
            "inicio": pd.to_datetime(
                [
                    "2024-10-01 08:00:00",
                    "2024-10-01 12:00:00",
                    "2024-10-02 09:00:00",
                ]
            ),
        }
    )

    estaciones = pd.DataFrame(
        {
            "estacion_id": [2, 3],
        }
    )

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
        (resultado["estacion_id"] == 2) & (resultado["fecha"] == pd.Timestamp("2024-10-01")),
        "demanda",
    ].iloc[0]

    assert demanda_estacion_2 == 2

    # Estación 3: cero viajes el primer día
    demanda_estacion_3 = resultado.loc[
        (resultado["estacion_id"] == 3) & (resultado["fecha"] == pd.Timestamp("2024-10-01")),
        "demanda",
    ].iloc[0]

    assert demanda_estacion_3 == 0


def test_filtrar_por_primera_actividad():
    viajes = pd.DataFrame(
        {
            "origen_id": [2, 3],
            "inicio": pd.to_datetime(
                [
                    "2024-12-01 08:00:00",
                    "2024-12-16 09:00:00",
                ]
            ),
        }
    )

    estaciones = pd.DataFrame(
        {
            "estacion_id": [2, 3],
        }
    )

    actividad = pd.DataFrame(
        {
            "estacion_id": [2, 3],
            "primera_actividad": pd.to_datetime(
                [
                    "2024-12-01",
                    "2024-12-16",
                ]
            ),
        }
    )

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

    estacion_3 = resultado[resultado["estacion_id"] == 3]

    assert estacion_3["fecha"].min() == pd.Timestamp("2024-12-16")


def test_agregar_features_calendario():
    demanda = pd.DataFrame(
        {
            "estacion_id": [2, 2],
            "fecha": pd.to_datetime(
                [
                    "2024-10-04",  # Viernes
                    "2024-10-05",  # Sábado
                ]
            ),
            "demanda": [50, 30],
        }
    )

    resultado = agregar_features_calendario(demanda)

    assert resultado["dia_semana"].tolist() == [4, 5]
    assert resultado["mes"].tolist() == [10, 10]
    assert resultado["es_fin_semana"].tolist() == [0, 1]
    assert resultado["dia_anio"].tolist() == [278, 279]
    assert resultado["demanda"].tolist() == [50, 30]


def test_agregar_lag_demanda():
    demanda = pd.DataFrame(
        {
            "estacion_id": [2, 2, 2],
            "fecha": pd.to_datetime(
                [
                    "2024-10-01",
                    "2024-10-02",
                    "2024-10-03",
                ]
            ),
            "demanda": [50, 30, 40],
        }
    )

    resultado = agregar_lag_demanda(demanda)

    assert pd.isna(resultado["demanda_lag_1"].iloc[0])
    assert resultado["demanda_lag_1"].iloc[1] == 50
    assert resultado["demanda_lag_1"].iloc[2] == 30


def test_lag_con_fechas_faltantes():
    demanda = pd.DataFrame(
        {
            "estacion_id": [2, 2, 2],
            "fecha": pd.to_datetime(
                [
                    "2024-10-01",
                    "2024-10-02",
                    "2024-10-05",
                ]
            ),
            "demanda": [50, 30, 40],
        }
    )

    resultado = agregar_lag_demanda(demanda)

    assert pd.isna(resultado["demanda_lag_1"].iloc[0])

    assert resultado["demanda_lag_1"].iloc[1] == 50

    # El 4 de octubre no existe en los datos
    assert pd.isna(resultado["demanda_lag_1"].iloc[2])


def test_agregar_lag_7():
    demanda = pd.DataFrame(
        {
            "estacion_id": [2, 2, 2],
            "fecha": pd.to_datetime(
                [
                    "2024-10-01",
                    "2024-10-08",
                    "2024-10-09",
                ]
            ),
            "demanda": [50, 65, 45],
        }
    )

    resultado = agregar_lag_7(demanda)

    assert pd.isna(resultado["demanda_lag_7"].iloc[0])
    assert resultado["demanda_lag_7"].iloc[1] == 50

    # No existe registro del 2 de octubre
    assert pd.isna(resultado["demanda_lag_7"].iloc[2])


def test_agregar_media_movil_7():
    demanda = pd.DataFrame(
        {
            "estacion_id": [2] * 8,
            "fecha": pd.date_range(
                "2024-10-01",
                periods=8,
                freq="D",
            ),
            "demanda": [10, 20, 30, 40, 50, 60, 70, 100],
        }
    )

    resultado = agregar_media_movil_7(demanda)

    # No hay suficiente historial durante los primeros 7 días
    assert resultado["demanda_media_7"].iloc[:7].isna().all()

    # Para el día 8: promedio de 10, 20, ..., 70
    assert resultado["demanda_media_7"].iloc[7] == 40


def test_media_movil_con_fechas_faltantes():
    demanda = pd.DataFrame(
        {
            "estacion_id": [2] * 7,
            "fecha": pd.to_datetime(
                [
                    "2024-10-01",
                    "2024-10-02",
                    "2024-10-03",
                    "2024-10-04",
                    "2024-10-05",
                    "2024-10-07",
                    "2024-10-08",
                ]
            ),
            "demanda": [10, 20, 30, 40, 50, 70, 80],
        }
    )

    resultado = agregar_media_movil_7(demanda)

    # Falta el 6 de octubre.
    # No existen siete días completos anteriores al 8.
    assert pd.isna(resultado["demanda_media_7"].iloc[-1])


def test_media_movil_varias_estaciones():
    demanda = pd.DataFrame(
        {
            "estacion_id": [2] * 8 + [3] * 8,
            "fecha": list(pd.date_range("2024-10-01", periods=8)) * 2,
            "demanda": [10] * 8 + [100] * 8,
        }
    )

    resultado = agregar_media_movil_7(demanda)

    estacion_2 = resultado[resultado["estacion_id"] == 2]

    estacion_3 = resultado[resultado["estacion_id"] == 3]

    assert estacion_2["demanda_media_7"].iloc[-1] == 10
    assert estacion_3["demanda_media_7"].iloc[-1] == 100


def test_media_movil_28_excluye_dia_actual_y_futuro():
    demanda = pd.DataFrame(
        {
            "estacion_id": [2] * 30,
            "fecha": pd.date_range("2024-10-01", periods=30, freq="D"),
            "demanda": list(range(1, 30)) + [9999],
        }
    )

    resultado = agregar_media_movil_28(demanda)

    assert resultado["demanda_media_28"].iloc[:28].isna().all()
    assert resultado["demanda_media_28"].iloc[28] == 14.5
    assert resultado["demanda_media_28"].iloc[29] == 15.5

    alterada = demanda.copy()
    alterada.loc[29, "demanda"] = -9999
    assert agregar_media_movil_28(alterada)["demanda_media_28"].iloc[29] == 15.5


def test_limpiar_viajes():
    import pandas as pd

    from mibici.features import limpiar_viajes

    estaciones = pd.DataFrame(
        {
            "estacion_id": [1, 2, 3],
        }
    )

    viajes = pd.DataFrame(
        {
            "viaje_id": [1, 1, 2, 3, 4, 5],
            "origen_id": [1, 1, 1, 99, 2, 2],
            "destino_id": [2, 2, 3, 2, 3, 3],
            "inicio": pd.to_datetime(
                [
                    "2026-01-01 08:00",
                    "2026-01-01 08:00",
                    "2026-01-01 09:00",
                    "2026-01-01 10:00",
                    "2026-01-01 11:00",
                    "2026-01-01 12:00",
                ]
            ),
            "fin": pd.to_datetime(
                [
                    "2026-01-01 08:15",  # Válido
                    "2026-01-01 08:15",  # Duplicado
                    "2026-01-01 09:00",  # Duración cero
                    "2026-01-01 10:15",  # Estación desconocida
                    "2026-01-03 11:01",  # Más de 24 horas
                    "2026-01-01 12:30",  # Válido
                ]
            ),
        }
    )

    resultado = limpiar_viajes(viajes, estaciones)

    assert len(resultado) == 2
    assert resultado["viaje_id"].tolist() == [1, 5]
    assert len(viajes) == 6  # No modifica el original


def test_limpiar_viajes_limites_duracion():
    import pandas as pd

    from mibici.features import limpiar_viajes

    estaciones = pd.DataFrame({"estacion_id": [1, 2]})

    inicio = pd.Timestamp("2026-01-01 08:00")

    viajes = pd.DataFrame(
        {
            "viaje_id": [1, 2, 3],
            "origen_id": [1, 1, 1],
            "destino_id": [2, 2, 2],
            "inicio": [inicio, inicio, inicio],
            "fin": [
                inicio - pd.Timedelta(minutes=1),
                inicio + pd.Timedelta(hours=24),
                inicio + pd.Timedelta(hours=24, seconds=1),
            ],
        }
    )

    resultado = limpiar_viajes(viajes, estaciones)

    assert resultado["viaje_id"].tolist() == [2]


def test_particion_temporal():
    import pandas as pd

    from mibici.features import particion_temporal

    datos = pd.DataFrame(
        {
            "fecha": pd.to_datetime(
                [
                    "2024-09-30",  # Fuera del periodo
                    "2024-10-01",  # Entrenamiento
                    "2026-03-31",  # Entrenamiento
                    "2026-04-01",  # Validación
                    "2026-06-30",  # Validación
                    "2026-07-01",  # Prueba
                    "2026-09-30",  # Prueba
                    "2026-10-01",  # Fuera del periodo
                ]
            ),
            "valor": range(8),
        }
    )

    entrenamiento, validacion, prueba = particion_temporal(datos)

    assert entrenamiento["valor"].tolist() == [1, 2]
    assert validacion["valor"].tolist() == [3, 4]
    assert prueba["valor"].tolist() == [5, 6]

    # Ningún registro debe aparecer en dos conjuntos
    assert set(entrenamiento.index).isdisjoint(validacion.index)
    assert set(entrenamiento.index).isdisjoint(prueba.index)
    assert set(validacion.index).isdisjoint(prueba.index)

    # La función no modifica los datos originales
    assert len(datos) == 8


def test_flujo_por_franja():
    import pandas as pd

    from mibici.features import flujo_por_franja

    viajes = pd.DataFrame(
        {
            "inicio": pd.to_datetime(
                [
                    "2026-04-01 08:50",
                    "2026-04-01 09:15",
                ]
            ),
            "fin": pd.to_datetime(
                [
                    "2026-04-01 09:10",
                    "2026-04-01 09:30",
                ]
            ),
            "origen_id": [101, 102],
            "destino_id": [102, 101],
        }
    )

    resultado = flujo_por_franja(viajes)

    assert len(resultado) == 4

    esperado = {
        (101, "2026-04-01 06:00"): (1, 0, 1),
        (101, "2026-04-01 09:00"): (0, 1, -1),
        (102, "2026-04-01 06:00"): (0, 0, 0),
        (102, "2026-04-01 09:00"): (1, 1, 0),
    }

    assert len(resultado) == len(esperado)

    for _, fila in resultado.iterrows():
        clave = (
            fila["estacion_id"],
            fila["fecha_franja"].strftime("%Y-%m-%d %H:%M"),
        )

        assert clave in esperado
        assert (
            fila["salidas"],
            fila["llegadas"],
            fila["flujo_neto"],
        ) == esperado[clave]


def test_etiquetar_riesgo():
    import pandas as pd

    from mibici.features import etiquetar_riesgo

    flujos = pd.DataFrame(
        {
            "estacion_id": [101, 101, 101, 101, 101],
            "flujo_neto": [10, -10, 0, 9, -9],
        }
    )

    estaciones = pd.DataFrame(
        {
            "estacion_id": [101],
            "capacidad": [20],
        }
    )

    resultado = etiquetar_riesgo(
        flujos,
        estaciones,
        alpha=0.5,
    )

    assert resultado["riesgo"].tolist() == [
        "sin_bicicletas",
        "sin_anclajes",
        "normal",
        "normal",
        "normal",
    ]

    assert resultado["capacidad"].tolist() == [20] * 5

    # Los DataFrames originales no deben modificarse
    assert "riesgo" not in flujos.columns
    assert "riesgo" not in estaciones.columns


def test_agregar_rezagos_flujo():
    import pandas as pd

    from mibici.features import agregar_rezagos_flujo

    flujos = pd.DataFrame(
        {
            "estacion_id": [101, 101, 101, 101, 102],
            "fecha_franja": pd.to_datetime(
                [
                    "2026-04-01 06:00",
                    "2026-04-01 09:00",
                    "2026-04-01 15:00",
                    "2026-04-08 06:00",
                    "2026-04-01 09:00",
                ]
            ),
            "flujo_neto": [5, -3, 8, 12, 20],
        }
    )

    resultado = agregar_rezagos_flujo(flujos)

    estacion_101 = resultado.loc[resultado["estacion_id"] == 101].set_index("fecha_franja")

    # Rezago de 3 horas
    assert estacion_101.loc["2026-04-01 09:00", "flujo_lag_1"] == 5

    # No debe utilizar la fila anterior si faltan franjas
    assert pd.isna(estacion_101.loc["2026-04-01 15:00", "flujo_lag_1"])

    # Rezago de la misma franja de la semana anterior
    assert estacion_101.loc["2026-04-08 06:00", "flujo_lag_56"] == 5

    # No debe mezclar información de otras estaciones
    estacion_102 = resultado.loc[resultado["estacion_id"] == 102]

    assert pd.isna(estacion_102["flujo_lag_1"].iloc[0])

    # Los datos originales permanecen intactos
    assert "flujo_lag_1" not in flujos.columns


def test_preparar_datos_p2_sin_fuga():
    import pandas as pd

    from mibici.features import preparar_datos_p2

    flujos = pd.DataFrame(
        {
            "estacion_id": [101, 101, 101],
            "fecha_franja": pd.to_datetime(
                [
                    "2026-04-01 06:00",
                    "2026-04-01 09:00",
                    "2026-04-01 12:00",
                ]
            ),
            "salidas": [5, 15, 2],
            "llegadas": [3, 1, 2],
            "flujo_neto": [2, 14, 0],
        }
    )

    estaciones = pd.DataFrame(
        {
            "estacion_id": [101],
            "capacidad": [20],
        }
    )

    resultado = preparar_datos_p2(flujos, estaciones)

    # A las 09:00 solo puede utilizar el flujo de las 06:00
    fila_09 = resultado.loc[resultado["fecha_franja"] == pd.Timestamp("2026-04-01 09:00")].iloc[0]

    assert fila_09["flujo_lag_1"] == 2
    assert fila_09["riesgo"] == "sin_bicicletas"

    # A las 12:00 ya puede utilizar el flujo de las 09:00
    fila_12 = resultado.loc[resultado["fecha_franja"] == pd.Timestamp("2026-04-01 12:00")].iloc[0]

    assert fila_12["flujo_lag_1"] == 14
    assert fila_12["riesgo"] == "normal"

    # El flujo contemporáneo no debe aparecer como predictor
    assert "flujo_neto" not in resultado.columns
    assert "salidas" not in resultado.columns
    assert "llegadas" not in resultado.columns

    # El primer periodo no dispone de flujo anterior
    assert pd.isna(resultado["flujo_lag_1"].iloc[0])


def test_perfiles_estacion():
    import pandas as pd

    from mibici.features import perfiles_estacion

    viajes = pd.DataFrame(
        {
            "inicio": pd.to_datetime(
                [
                    "2026-03-30 08:00",  # Lunes, entrenamiento
                    "2026-03-30 08:30",  # Lunes, entrenamiento
                    "2026-03-29 10:00",  # Domingo, entrenamiento
                    "2026-04-01 08:00",  # Validación: excluir
                ]
            ),
            "fin": pd.to_datetime(
                [
                    "2026-03-30 08:15",
                    "2026-03-30 08:45",
                    "2026-03-29 10:15",
                    "2026-04-01 08:15",
                ]
            ),
            "origen_id": [101, 101, 101, 102],
            "destino_id": [102, 102, 102, 101],
        }
    )

    resultado = perfiles_estacion(viajes)

    # Una fila por estación, con 96 variables horarias
    assert len(resultado) == 2
    assert resultado.shape[1] == 102

    estacion_101 = resultado.set_index("estacion_id").loc[101]

    # Las salidas laborales ocurren a las 08:00
    assert estacion_101["salidas_laboral_08"] == 1.0

    # Las salidas de fin de semana ocurren a las 10:00
    assert estacion_101["salidas_fin_semana_10"] == 1.0

    # No hay salidas laborales a las 09:00
    assert estacion_101["salidas_laboral_09"] == 0.0

    # La estación 102 recibe los viajes de entrenamiento
    estacion_102 = resultado.set_index("estacion_id").loc[102]

    assert estacion_102["llegadas_laboral_08"] == 1.0
    assert estacion_102["llegadas_fin_semana_10"] == 1.0

    # El viaje de validación no debe incorporarse
    assert estacion_102["salidas_laboral_08"] == 0.0

    # Indicadores agregados de P3
    assert estacion_101["total_salidas"] == 3
    assert estacion_101["total_llegadas"] == 0
    assert estacion_101["volumen_total"] == 3
    assert estacion_101["balance_neto"] == 3
    assert estacion_101["duracion_media_min"] == 15

    assert estacion_102["total_salidas"] == 0
    assert estacion_102["total_llegadas"] == 3
    assert estacion_102["volumen_total"] == 3
    assert estacion_102["balance_neto"] == -3
    assert pd.isna(estacion_102["duracion_media_min"])


def test_perfiles_estacion_sin_fuga_temporal():
    import pandas as pd

    from mibici.features import perfiles_estacion

    viajes = pd.DataFrame(
        {
            "inicio": pd.to_datetime(
                [
                    "2026-03-31 08:00",
                    "2026-03-31 23:55",
                    "2026-04-01 09:00",
                ]
            ),
            "fin": pd.to_datetime(
                [
                    "2026-03-31 08:15",
                    "2026-04-01 00:10",
                    "2026-04-01 09:15",
                ]
            ),
            "origen_id": [101, 101, 101],
            "destino_id": [102, 103, 104],
        }
    )

    resultado = perfiles_estacion(viajes)

    # Solo debe conservarse el viaje completamente
    # comprendido dentro del entrenamiento.
    assert set(resultado["estacion_id"]) == {101, 102}

    estacion_101 = resultado.set_index("estacion_id").loc[101]

    assert estacion_101["salidas_laboral_08"] == 1.0
    assert estacion_101["salidas_laboral_23"] == 0.0

    # No deben incorporarse estaciones de viajes excluidos.
    assert 103 not in resultado["estacion_id"].values
    assert 104 not in resultado["estacion_id"].values


def test_filtrar_por_primera_actividad_conserva_ceros_posteriores():
    import pandas as pd

    from mibici.features import filtrar_por_primera_actividad

    demanda = pd.DataFrame(
        {
            "estacion_id": [1, 1, 1, 1, 2, 2, 3],
            "fecha": pd.to_datetime(
                [
                    "2024-10-01",
                    "2024-10-02",
                    "2024-10-03",
                    "2024-10-04",
                    "2024-10-01",
                    "2024-10-02",
                    "2024-10-01",
                ]
            ),
            "demanda": [0, 0, 5, 0, 0, 3, 0],
        }
    )

    actividad = pd.DataFrame(
        {
            "estacion_id": [1, 2],
            "primera_actividad": pd.to_datetime(["2024-10-03", "2024-10-02"]),
        }
    )

    resultado = filtrar_por_primera_actividad(demanda, actividad)

    assert len(resultado) == 3
    assert resultado["estacion_id"].tolist() == [1, 1, 2]
    assert resultado["demanda"].tolist() == [5, 0, 3]
    assert resultado["fecha"].min() == pd.Timestamp("2024-10-02")


def test_consolidar_demanda_diaria_respeta_fechas(tmp_path):
    import pandas as pd

    from mibici.features import consolidar_demanda_diaria

    directorio = tmp_path / "viajes"
    directorio.mkdir()

    estaciones = pd.DataFrame(
        {
            "estacion_id": [1, 2],
        }
    )

    viajes = pd.DataFrame(
        {
            "viaje_id": [1, 2, 3],
            "inicio": pd.to_datetime(
                [
                    "2024-10-31 10:00",
                    "2024-11-01 10:00",
                    "2024-11-02 10:00",
                ]
            ),
            "fin": pd.to_datetime(
                [
                    "2024-10-31 10:15",
                    "2024-11-01 10:15",
                    "2024-11-02 10:15",
                ]
            ),
            "origen_id": [1, 1, 2],
            "destino_id": [2, 2, 1],
        }
    )

    viajes.to_parquet(directorio / "viajes_2024_11.parquet")

    resultado = consolidar_demanda_diaria(
        directorio_viajes=directorio,
        estaciones=estaciones,
        fecha_inicio="2024-11-01",
        fecha_fin="2024-11-02",
    )

    assert len(resultado) == 4
    assert resultado["fecha"].min() == pd.Timestamp("2024-11-01")
    assert resultado["fecha"].max() == pd.Timestamp("2024-11-02")
    assert resultado["demanda"].sum() == 2
    assert (resultado["demanda"] == 0).sum() == 2


def test_agregar_features_calendario_festivos_mexico():
    import pandas as pd

    from mibici.features import agregar_features_calendario

    datos = pd.DataFrame(
        {
            "estacion_id": [1, 1, 1, 1],
            "fecha": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-09-16",
                    "2026-02-02",
                    "2026-02-05",
                ]
            ),
            "demanda": [10, 20, 30, 40],
        }
    )

    resultado = agregar_features_calendario(datos)

    assert resultado["es_festivo"].tolist() == [1, 1, 1, 0]
    assert resultado["es_fin_semana"].tolist() == [0, 0, 0, 0]
    assert resultado["demanda"].tolist() == [10, 20, 30, 40]


def test_calcular_primera_actividad():
    import pandas as pd

    from mibici.features import calcular_primera_actividad

    demanda = pd.DataFrame(
        {
            "estacion_id": [1, 1, 1, 2, 2, 3, 3],
            "fecha": pd.to_datetime(
                [
                    "2024-10-01",
                    "2024-10-02",
                    "2024-10-03",
                    "2024-10-01",
                    "2024-10-02",
                    "2024-10-01",
                    "2024-10-02",
                ]
            ),
            "demanda": [0, 5, 3, 0, 2, 0, 0],
        }
    )

    resultado = calcular_primera_actividad(demanda)

    assert len(resultado) == 2
    assert resultado["estacion_id"].tolist() == [1, 2]
    assert resultado["primera_actividad"].tolist() == [
        pd.Timestamp("2024-10-02"),
        pd.Timestamp("2024-10-02"),
    ]


def test_preparar_datos_p1_integracion(tmp_path):
    import pandas as pd

    from mibici.features import preparar_datos_p1

    estaciones = pd.DataFrame({"estacion_id": [1, 2, 3]})

    viajes = pd.DataFrame(
        {
            "viaje_id": [1, 2, 3],
            "inicio": pd.to_datetime(
                [
                    "2024-10-02 10:00",
                    "2024-10-04 11:00",
                    "2024-10-05 12:00",
                ]
            ),
            "fin": pd.to_datetime(
                [
                    "2024-10-02 10:15",
                    "2024-10-04 11:15",
                    "2024-10-05 12:15",
                ]
            ),
            "origen_id": [1, 1, 2],
            "destino_id": [2, 2, 1],
        }
    )

    viajes.to_parquet(tmp_path / "viajes_2024_10.parquet")

    resultado = preparar_datos_p1(tmp_path, estaciones, "2024-10-01", "2024-10-07")

    assert len(resultado) == 9
    assert resultado["estacion_id"].nunique() == 2
    assert resultado["demanda"].sum() == 3
    assert (resultado["demanda"] == 0).sum() == 6

    columnas = {
        "es_festivo",
        "demanda_lag_1",
        "demanda_lag_7",
        "demanda_media_7",
    }

    assert columnas.issubset(resultado.columns)
