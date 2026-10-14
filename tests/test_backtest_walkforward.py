from datetime import datetime

import click
import pytest

from quantagent.cli.walkforward import construir_ventanas, seleccionar_ganador


def test_cuatro_ventanas_calendario_sin_solapamiento():
    ventanas = construir_ventanas(datetime(2007, 1, 3), datetime(2018, 12, 31), 4, 2)
    assert ventanas == [
        (datetime(2007, 1, 3), datetime(2010, 12, 31), datetime(2011, 1, 1), datetime(2012, 12, 31)),
        (datetime(2009, 1, 1), datetime(2012, 12, 31), datetime(2013, 1, 1), datetime(2014, 12, 31)),
        (datetime(2011, 1, 1), datetime(2014, 12, 31), datetime(2015, 1, 1), datetime(2016, 12, 31)),
        (datetime(2013, 1, 1), datetime(2016, 12, 31), datetime(2017, 1, 1), datetime(2018, 12, 31)),
    ]
    assert all(ajuste_fin < prueba_inicio <= prueba_fin for _, ajuste_fin, prueba_inicio, prueba_fin in ventanas)
    assert all(a[3] < b[2] for a, b in zip(ventanas, ventanas[1:]))


def test_fecha_bisiesta_y_descarta_ultima_ventana_incompleta():
    ventanas = construir_ventanas(datetime(2008, 2, 29), datetime(2015, 12, 23), 4, 2)
    assert ventanas == [
        (datetime(2008, 2, 29), datetime(2011, 12, 31), datetime(2012, 1, 1), datetime(2013, 12, 31)),
    ]


@pytest.mark.parametrize("train,test", [(0, 2), (-1, 2), (4, 0), (4, -1)])
def test_rechaza_anios_no_positivos(train, test):
    with pytest.raises(click.ClickException, match="deben ser positivos"):
        construir_ventanas(datetime(2007, 1, 3), datetime(2018, 12, 31), train, test)


def test_rango_insuficiente():
    with pytest.raises(click.ClickException, match="ventana completa"):
        construir_ventanas(datetime(2007, 1, 3), datetime(2012, 12, 23), 4, 2)


@pytest.mark.parametrize("dia,cantidad", [(30, 3), (24, 3), (23, 2)])
def test_ultima_semana_del_anio_completa_la_ventana(dia, cantidad):
    assert len(construir_ventanas(datetime(2007, 1, 3), datetime(2016, 12, dia), 4, 2)) == cantidad


@pytest.mark.parametrize("sharpes,ganador", [([None, -0.5, None], 1), ([-0.5, None], 0), ([None, None], 0)])
def test_sharpe_ausente_pierde_y_empate_conserva_el_primero(sharpes, ganador):
    resultados = [{"sharpe_tasa_0": sharpe} for sharpe in sharpes]
    assert seleccionar_ganador(resultados) is resultados[ganador]


def test_maximo_y_empate_conservan_orden_de_grilla():
    resultados = [
        {"params": {"rsi_period": 7}, "sharpe_tasa_0": -0.5},
        {"params": {"rsi_period": 14}, "sharpe_tasa_0": 0.8},
        {"params": {"rsi_period": 21}, "sharpe_tasa_0": 0.8},
    ]
    assert seleccionar_ganador(resultados) is resultados[1]
    assert seleccionar_ganador(list(reversed(resultados))) is resultados[2]
    assert seleccionar_ganador(resultados[:1]) is resultados[0]
