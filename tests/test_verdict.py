from math import sqrt
from statistics import NormalDist

import pytest

from quantagent.backtesting.verdict import evaluar, sharpe_deflactado


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


def test_sharpe_deflactado_contra_cuenta_a_mano():
    # N = 10, V = 0.0001, T = 1001, SR = 0.05, asimetría 0, curtosis 3.
    # Z⁻¹(1 − 1/10) = Z⁻¹(0.9) = 1.2815515655
    # Z⁻¹(1 − 1/(10·e)) = Z⁻¹(0.9632120559) = 1.7892417646
    # SR0 = 0.01 · (0.4227843351 · 1.2815515655 + 0.5772156649 · 1.7892417646)
    #     = 0.01 · (0.5418199265 + 1.0327783748) = 0.01 · 1.5745983013 = 0.0157459830
    # numerador = (0.05 − 0.0157459830) · sqrt(1000) = 1.0832071269
    # denominador = sqrt(1 − 0 · 0.05 + (3 − 1)/4 · 0.0025) = sqrt(1.00125) = 1.0006248048
    # z = 1.0832071269 / 1.0006248048 = 1.0825307565 → Φ(z) = 0.8604916217
    assert sharpe_deflactado(0.05, 10, 0.0001, 1001, 0, 3) == pytest.approx(0.8604916217, abs=1e-6)


def test_sharpe_deflactado_baja_con_mas_intentos():
    dsr = [sharpe_deflactado(0.05, n, 0.0001, 1001, 0, 3) for n in (1, 10, 100)]
    assert dsr[0] > dsr[1] > dsr[2]


def test_sharpe_deflactado_un_intento_tiene_sr0_cero():
    sr, t, asimetria, curtosis = 0.04, 500, -0.5, 5
    esperado = NormalDist().cdf(
        sr * sqrt(t - 1) / sqrt(1 - asimetria * sr + (curtosis - 1) / 4 * sr**2)
    )
    # Con N = 1 la varianza entre intentos no interviene.
    for var in (0.0, 0.0001, 1.0):
        assert sharpe_deflactado(sr, 1, var, t, asimetria, curtosis) == pytest.approx(esperado, abs=1e-12)
