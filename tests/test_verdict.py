import csv
import json
from math import sqrt
from statistics import NormalDist

import pytest
from click.testing import CliRunner

from quantagent.backtesting.verdict import evaluar, sharpe_deflactado
from quantagent.cli.backtest import backtest_group
from quantagent.cli.verdict import COLUMNAS


def test_verdict_robustez_desde_trades(tmp_path):
    tabla, salida = tmp_path / "tabla.csv", tmp_path / "salida.csv"
    filas = [dict.fromkeys(COLUMNAS, "") for _ in range(5)]
    for i, fila in enumerate(filas):
        fila.update(estrategia="rsi", activo=f"ETF{i}")
    with tabla.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS)
        writer.writeheader()
        writer.writerows(filas)
    for i, pnls in enumerate(([0, -12, 0], [3, 3, 3], [], [0, 0, 0])):
        with (tmp_path / f"rsi-ETF{i}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["exit_time", "pnl"])
            writer.writerows(zip(["2020-01-01", "2019-12-31T23:59:59", "2020-06-01"], pnls))
    args = ["verdict", "--tabla", str(tabla), "--out", str(salida)]
    runner = CliRunner()
    original = runner.invoke(backtest_group, args)
    assert original.exit_code == 0, original.output
    sin_trades = salida.read_text()
    assert next(csv.reader(sin_trades.splitlines())) == COLUMNAS
    resultado = runner.invoke(backtest_group, args + ["--trades-dir", str(tmp_path)])
    assert resultado.exit_code == 0, resultado.output
    with salida.open() as f:
        obtenidas = list(csv.DictReader(f))
    assert [f["concentracion_pnl"] for f in obtenidas] == ["1.0", "0.3333", "", "", ""]
    assert obtenidas[0]["pnl_por_anio"] == '{"2019": -12.00, "2020": 0.00}'
    assert json.loads(obtenidas[1]["pnl_por_anio"]) == {"2019": 3, "2020": 6}
    assert obtenidas[2]["pnl_por_anio"] == obtenidas[4]["pnl_por_anio"] == ""
    assert json.loads(obtenidas[3]["pnl_por_anio"]) == {"2019": 0, "2020": 0}
    assert [{c: f[c] for c in COLUMNAS} for f in obtenidas] == list(csv.DictReader(sin_trades.splitlines()))
    repetido = runner.invoke(backtest_group, args)
    assert repetido.output == original.output
    assert salida.read_text() == sin_trades


@pytest.mark.parametrize("sin_columna", [False, True])
def test_verdict_csv_real(tmp_path, sin_columna):
    tabla, salida = tmp_path / "tabla.csv", tmp_path / "salida.csv"
    columnas = [c for c in COLUMNAS if c not in ("veredicto", "motivos")]
    filas = []
    for i in range(4):
        fila = dict.fromkeys(columnas, "")
        fila.update(estrategia="rsi", activo=f"ETF{i}", desde="2007-01-01", hasta="2018-12-31", control="coinciden 2 de 2")
        for sufijo in ("", "_apertura_siguiente"):
            fila.update({c + sufijo: v for c, v in zip(("trades", "pnl", "sharpe_tasa_0", "max_drawdown"), (2, 10, 0.10, 0.35))})
        filas.append(fila)
    filas[1]["pnl_apertura_siguiente"] = 0
    filas[2]["trades"] = 1
    for c in columnas:
        if c.endswith("_apertura_siguiente"):
            filas[3][c] = ""
    if sin_columna:
        columnas.remove("sharpe_tasa_0_apertura_siguiente")
    with tabla.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(filas)
    resultado = CliRunner().invoke(backtest_group, ["verdict", "--tabla", str(tabla), "--out", str(salida)])
    if sin_columna:
        assert resultado.exit_code == 1
        assert "sharpe_tasa_0_apertura_siguiente" in resultado.output
        assert not salida.exists()
    else:
        assert resultado.exit_code == 0, resultado.output
        assert "PASA: 1 de 4" in resultado.output
        with salida.open() as f:
            obtenidas = list(csv.DictReader(f))
        assert [(f["veredicto"], f["motivos"]) for f in obtenidas] == [
            ("PASA", ""), ("NO PASA", "apertura: pnl <= 0"), ("NO PASA", "cierre: trades < 2"),
            ("NO PASA", "apertura: sin trades; apertura: sin pnl; apertura: sin sharpe_tasa_0; apertura: sin max_drawdown"),
        ]


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
