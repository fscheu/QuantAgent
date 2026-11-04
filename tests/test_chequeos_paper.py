from types import SimpleNamespace

import pytest

from quantagent.fabrica.chequeos_paper import comparar_pesos, reconciliar_cartera, reportar_dividendos, reportar_rechazos


@pytest.mark.parametrize("quantity, pending, shadow, alerta", [
    (212, [("buy", 10, 0)], .5, False),  # DBC 2026-11-04: objetivo 222.5 acciones.
    (212, [("buy", 15, 5)], .5, False),
    (212, [("buy", 6, 1), ("buy", 8, 3)], .5, False),
    (232, [("sell", 15, 5)], .5, False),
    (212, [("buy", 9.5, 0)], .5, False),  # Exactamente una acción.
    (212, [("buy", 9, 0)], .5, True),
    (232, [("sell", 14, 5)], .5, False),
    (232, [("sell", 13, 5)], .5, True),
    (0, [("buy", 222, 0)], .5, False),  # Compra inicial aún pendiente.
    (0, [], .5, True),  # Sombra sin posición ni orden.
    (2, [], None, True),  # Posición fuera de sombra.
    (0, [("buy", 2, 0)], None, True),  # Pendiente fuera de sombra.
])
def test_reconciliar_cartera_reserva_y_cantidades_pendientes(quantity, pending, shadow, alerta):
    posiciones = [{"symbol": "DBC", "quantity": quantity}] if quantity else []
    pendientes = [{"symbol": "DBC", "side": side, "quantity": q, "filled_quantity": filled}
                  for side, q, filled in pending]
    diferencias = reconciliar_cartera(posiciones, pendientes, {"DBC": 19}, 8900,
                                     {} if shadow is None else {"DBC": shadow})
    assert bool(diferencias) == alerta
    if alerta:
        assert "DBC: peso paper" in diferencias[0]


def test_reconciliar_cartera_recorre_union_de_activos():
    diferencias = reconciliar_cartera(
        [{"symbol": "SPY", "quantity": 2}], [], {"SPY": 100}, 1000, {"QQQ": .5})
    assert [d.split(":")[0] for d in diferencias] == ["QQQ", "SPY"]


def test_reconciliar_cartera_pendiente_sin_precio_alerta():
    assert reconciliar_cartera([], [{"symbol": "DBC", "side": "buy", "quantity": 10,
                                    "filled_quantity": 0}], {}, 8900, {}) == [
        "DBC: sin precio para reconciliar posición y pendientes"]


def test_pesos_dentro_de_tolerancia_pasan():
    assert comparar_pesos({"SPY": 0.50001}, {"SPY": 0.5}, tolerancia=0.0001) == []


def test_diferencia_de_pesos_informa_activo_y_motivo():
    diferencias = comparar_pesos({"SPY": 0.6}, {"SPY": 0.5})
    assert len(diferencias) == 1
    assert "SPY" in diferencias[0]
    assert "difiere" in diferencias[0]


def test_tolerancias_por_activo_incluyen_limite_y_fallback():
    tolerancias = {"SPY": 600 / 99941}
    assert comparar_pesos({}, {"SPY": tolerancias["SPY"]}, tolerancias=tolerancias) == []
    assert comparar_pesos({"SPY": 11520 / 99941}, {"SPY": 12120 / 99941}, tolerancias=tolerancias) == []
    assert comparar_pesos({}, {"SPY": tolerancias["SPY"] + 1e-10}, tolerancias=tolerancias)
    diferencias = comparar_pesos({}, {"SPY": .01, "QQQ": .0002}, tolerancias=tolerancias)
    assert len(diferencias) == 2
    assert f"(tolerancia {tolerancias['SPY']})" in diferencias[1]
    assert "QQQ: peso paper 0.0 difiere de sombra 0.0002 (tolerancia 0.0001)" == diferencias[0]


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
