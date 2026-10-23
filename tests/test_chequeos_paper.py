from quantagent.fabrica.chequeos_paper import comparar_pesos


def test_pesos_dentro_de_tolerancia_pasan():
    assert comparar_pesos({"SPY": 0.50001}, {"SPY": 0.5}, tolerancia=0.0001) == []


def test_diferencia_de_pesos_informa_activo_y_motivo():
    diferencias = comparar_pesos({"SPY": 0.6}, {"SPY": 0.5})
    assert len(diferencias) == 1
    assert "SPY" in diferencias[0]
    assert "difiere" in diferencias[0]
