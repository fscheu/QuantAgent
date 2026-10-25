from datetime import datetime, timezone

import pandas as pd

from quantagent.trading.cartera_paper import sesiones_cerradas_nyse, ultima_sesion_cerrada_nyse


def test_horario_regular_corte_sesion():
    # 2024-05-15 es miércoles regular (EDT: cierre 20:00 UTC)
    antes = datetime(2024, 5, 15, 19, 59, tzinfo=timezone.utc)
    despues = datetime(2024, 5, 15, 20, 1, tzinfo=timezone.utc)
    assert ultima_sesion_cerrada_nyse(antes) == pd.Timestamp("2024-05-14")
    assert ultima_sesion_cerrada_nyse(despues) == pd.Timestamp("2024-05-15")


def test_cierre_temprano_black_friday_sin_corte_16h():
    # 2023-11-24 Black Friday cierra 13:00 EST (18:00 UTC); jueves 23 fue feriado
    antes = pd.Timestamp("2023-11-24 17:59:00", tz="UTC")
    en_cierre = pd.Timestamp("2023-11-24 18:00:00", tz="UTC")
    entre_13h_y_16h_est = pd.Timestamp("2023-11-24 19:30:00", tz="UTC")
    assert ultima_sesion_cerrada_nyse(antes) == pd.Timestamp("2023-11-22")
    assert ultima_sesion_cerrada_nyse(en_cierre) == pd.Timestamp("2023-11-24")
    assert ultima_sesion_cerrada_nyse(entre_13h_y_16h_est) == pd.Timestamp("2023-11-24")


def test_fin_de_semana_y_feriado_conservan_sesion():
    # Memorial Day 2023: viernes 26 mayo cierra 20:00 UTC; lunes 29 feriado
    viernes_post = pd.Timestamp("2023-05-26 21:00:00", tz="UTC")
    sabado = pd.Timestamp("2023-05-27 14:00:00", tz="UTC")
    lunes_feriado = pd.Timestamp("2023-05-29 18:00:00", tz="UTC")
    esperada = pd.Timestamp("2023-05-26")
    assert ultima_sesion_cerrada_nyse(viernes_post) == esperada
    assert ultima_sesion_cerrada_nyse(sabado) == esperada
    assert ultima_sesion_cerrada_nyse(lunes_feriado) == esperada


def test_ventana_sesiones_sin_futuro_y_reloj_inyectable():
    as_of_naive = datetime(2024, 5, 15, 20, 30)
    as_of_aware = pd.Timestamp("2024-05-15 20:30:00", tz="UTC")
    s_naive = sesiones_cerradas_nyse(as_of_naive, n_sesiones=253)
    s_aware = sesiones_cerradas_nyse(as_of_aware, n_sesiones=253)
    assert s_naive.equals(s_aware)
    assert len(s_naive) == 253
    assert s_naive.is_monotonic_increasing
    assert s_naive[-1] == pd.Timestamp("2024-05-15")
    assert (s_naive <= pd.Timestamp("2024-05-15")).all()
