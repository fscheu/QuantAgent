from types import SimpleNamespace

from quantagent.fabrica.chequeos_paper import comparar_pesos, reportar_dividendos, reportar_rechazos


def test_pesos_dentro_de_tolerancia_pasan():
    assert comparar_pesos({"SPY": 0.50001}, {"SPY": 0.5}, tolerancia=0.0001) == []


def test_diferencia_de_pesos_informa_activo_y_motivo():
    diferencias = comparar_pesos({"SPY": 0.6}, {"SPY": 0.5})
    assert len(diferencias) == 1
    assert "SPY" in diferencias[0]
    assert "difiere" in diferencias[0]


def test_rechazos_reportan_motivo_y_falla_si_falta():
    reportes = reportar_rechazos([
        SimpleNamespace(status="rejected", symbol="SPY", comment="sin poder de compra"),
        SimpleNamespace(status="rejected", symbol="QQQ", comment=None),
        SimpleNamespace(status="filled", symbol="IWM", comment=None),
    ])
    assert reportes == ["SPY: rechazo, sin poder de compra", "QQQ: rechazo sin explicación"]


def test_dividendos_listan_datos_y_fallan_si_falta_tratamiento():
    reportes = reportar_dividendos([
        ("SPY", 1.25, "USD", "acreditado"),
        ("QQQ", 0.5, "USD", None),
    ])
    assert reportes == [
        "SPY: dividendo 1.25 USD, tratamiento acreditado",
        "QQQ: dividendo 0.5 USD, falta tratamiento",
    ]
