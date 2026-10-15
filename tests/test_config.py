from datetime import date

from mibici import config


def test_particion_temporal_ordenada_y_sin_traslape():
    fechas = [
        date.fromisoformat(f)
        for f in (
            config.FECHA_INICIO,
            config.INICIO_VALIDACION,
            config.INICIO_PRUEBA,
            config.FECHA_FIN,
        )
    ]
    assert fechas == sorted(fechas)
    assert len(set(fechas)) == len(fechas)


def test_urls_cubren_exactamente_los_24_meses_del_rango():
    inicio = date.fromisoformat(config.FECHA_INICIO)
    meses_esperados = []
    anio, mes = inicio.year, inicio.month
    for _ in range(24):
        meses_esperados.append(f"{anio}_{mes:02d}")
        anio, mes = (anio + 1, 1) if mes == 12 else (anio, mes + 1)

    assert list(config.URLS_VIAJES) == meses_esperados
    assert meses_esperados[-1] == config.FECHA_FIN[:7].replace("-", "_")


def test_urls_apuntan_al_archivo_de_su_mes():
    for mes, url in config.URLS_VIAJES.items():
        assert url.startswith("https://mibici.net/")
        assert url.endswith(f"/datos_abiertos_{mes}.csv")


def test_semilla_y_parametros_fijos():
    assert isinstance(config.SEMILLA, int)
    assert 0 < config.ALFA_RIESGO < 1
    assert 24 % config.HORAS_FRANJA == 0


def test_manifiesto_fuera_de_data_para_versionarse():
    assert config.DATA_DIR not in config.MANIFIESTO_DATOS.parents
