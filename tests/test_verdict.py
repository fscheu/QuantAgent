import pytest

from quantagent.backtesting.verdict import evaluar


def test_limites_inclusivos_y_pnl_positivo():
    assert evaluar(2, 0.01, 0.10, 0.35) == (True, [])
    assert evaluar(10, 100, 0.50, 0.20) == (True, [])


@pytest.mark.parametrize("campo,valor,motivo", [
    ("trades", 1, "trades < 2"),
    ("pnl", 0, "pnl <= 0"),
    ("pnl", -0.01, "pnl <= 0"),
    ("sharpe_tasa_0", 0.0999, "sharpe_tasa_0 < 0.10"),
    ("max_drawdown", 0.3501, "max_drawdown > 0.35"),
])
def test_cada_falla_tiene_su_motivo(campo, valor, motivo):
    fila = dict(trades=2, pnl=0.01, sharpe_tasa_0=0.10, max_drawdown=0.35)
    fila[campo] = valor
    assert evaluar(**fila) == (False, [motivo])


def test_motivos_acumulados_en_orden():
    assert evaluar(1, 0, 0.09, 0.35) == (
        False, ["trades < 2", "pnl <= 0", "sharpe_tasa_0 < 0.10"]
    )
    assert evaluar(1, 0, 0.09, 0.36) == (
        False, ["trades < 2", "pnl <= 0", "sharpe_tasa_0 < 0.10", "max_drawdown > 0.35"]
    )


@pytest.mark.parametrize("campo", ["trades", "pnl", "sharpe_tasa_0", "max_drawdown"])
@pytest.mark.parametrize("ausente", [None, ""])
def test_valor_ausente(campo, ausente):
    fila = dict(trades=2, pnl=0.01, sharpe_tasa_0=0.10, max_drawdown=0.35)
    fila[campo] = ausente
    assert evaluar(**fila) == (False, [f"sin {campo}"])


def test_todos_ausentes():
    assert evaluar(None, "", None, "") == (
        False, ["sin trades", "sin pnl", "sin sharpe_tasa_0", "sin max_drawdown"]
    )
