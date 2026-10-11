"""QuantAgent-832: comparacion motor vs port (scripts/crossval_compare.py). Se omite sin `backtesting`.

Corre en un subproceso (crossval_rsi fija TRADING_SLIPPAGE_PCT=0 al importarse; no debe filtrarse a otros
tests) que ejecuta main() dos veces, la segunda con un pnl del port alterado; el motor corre una sola vez.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.oraculo

pytest.importorskip("backtesting")

ROOT = Path(__file__).resolve().parent.parent
# Confianza de rsi (motor con cierres a 8 decimales, libreria con los del snapshot) en dos entradas reales de SPY
# 2007-2018, medidas en QuantAgent-zd9n: 2012-03-02 (la de mayor diferencia relativa) y 2011-01-18 (la de mayor
# diferencia en acciones).
CONFS = [(0.0022591031701116055, 0.002259100020987148), (0.36350611783976394, 0.36350611239534203)]
DRIVER = """
import contextlib, io, json, sys
sys.path.insert(0, "scripts")
import crossval_compare as c
def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return {"rc": c.main(argv), "out": buf.getvalue()}
def altered(col, delta, qty="10"):  # un trade inventado: el port difiere del motor solo en `col`, en `delta` absoluto
    row = {"entry_time": "t0", "exit_time": "t1", "symbol": "SPY", "side": "buy", "qty": qty,
           "entry_price": "100", "exit_price": "101", "pnl": "10"}
    lines, ok = c.compare_trades([row], [{**row, col: repr(float(row[col]) + delta)}])
    return {"ok": ok, "out": " / ".join(lines)}
def sized(pct, conf_motor, conf_lib, factor=1.0):  # una entrada con 100000 de equity a 106.36, sin movimiento
    q = lambda conf: 100000 * pct * conf / 106.36
    row = {"entry_time": "t0", "exit_time": "t1", "symbol": "SPY", "side": "buy", "qty": f"{q(conf_motor):.8f}",
           "entry_price": "106.36", "exit_price": "106.36", "pnl": "0"}
    return c.compare_trades([row], [{**row, "qty": repr(q(conf_lib) * factor)}])[1]
CONFS = __CONFS__
print(json.dumps([
    run([]),
    run(["--port-pnl-delta", "5:0.5"]),
    run(["--intrabar"]),
    run(["--intrabar", "--port-pnl-delta", "5:0.5"]),
    run(["--intrabar", "--engine-pnl-delta", "5:0.5"]),
    altered("qty", 2e-5),
    altered("qty", 1e-5),
    altered("exit_price", 1e-7),
    altered("exit_price", 1e-8),
    altered("qty", 2e-6, qty="1"),
    [[sized(pct, *conf, factor) for factor in (1.0, 1.0001, 0.9999)] for pct in (0.05, 1.0) for conf in CONFS],
]))
""".replace("__CONFS__", repr(CONFS))


@pytest.fixture(scope="module")
def runs():
    p = subprocess.run([sys.executable, "-c", DRIVER], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.splitlines()[-1])


def test_comparison_passes_on_unmodified_engines(runs):
    # Valida que motor y port coinciden de verdad: mismos 227 trades, metricas y curvas dentro de tolerancia.
    assert runs[0]["rc"] == 0, runs[0]["out"]
    assert "Trades: motor=227 port=227" in runs[0]["out"]
    assert "NO COMPARADO" not in runs[0]["out"]


def test_comparison_fails_and_names_the_altered_trade(runs):
    # Valida que la comparacion PUEDE fallar: un pnl del port alterado en 0.5 USD (tolerancia ~1e-7)
    # en el trade nº 6 debe dar exit != 0, nombrar ese trade y la columna pnl, y marcar total_pnl.
    assert runs[1]["rc"] == 1, runs[1]["out"]
    assert "PRIMER TRADE DIFERENTE: nº 6, columnas: pnl" in runs[1]["out"]
    assert re.search(r"total_pnl .*NO\n", runs[1]["out"]), runs[1]["out"]


def test_intrabar_comparison_passes_on_unmodified_engines(runs):
    # Valida que motor y port coinciden en modo intravela: 340 trades y curvas dentro de tolerancia.
    assert runs[2]["rc"] == 0, runs[2]["out"]
    assert "Trades: motor=340 port=340" in runs[2]["out"]
    assert "RESULTADO: TODO DENTRO DE TOLERANCIA" in runs[2]["out"]


def test_intrabar_comparison_fails_and_names_altered_trade_on_either_side(runs):
    # Valida que la comparacion intravela falla si se altera un trade de cualquiera de los dos lados:
    # tanto alterando el port (runs[3]) como alterando el motor propio (runs[4]).
    for idx in (3, 4):
        run_res = runs[idx]
        assert run_res["rc"] == 1, run_res["out"]
        assert "PRIMER TRADE DIFERENTE: nº 6, columnas: pnl" in run_res["out"]
        assert re.search(r"total_pnl .*NO\n", run_res["out"]), run_res["out"]


def test_row_tolerances_still_catch_an_altered_qty_and_exit_price(runs):
    # Valida sobre un trade inventado (qty 10, salida 101): con tolerancia de qty 1.5e-6 relativa
    # (QuantAgent-zd9n; 1.5e-5 acciones sobre 10), un qty alterado en 2e-5 falla y nombra la columna y uno en
    # 1e-5 pasa; con tolerancia de precio 2e-8, un exit_price alterado en 1e-7 falla y uno en 1e-8 (lo medido
    # sobre SPY real) pasa.
    assert runs[5]["ok"] is False, runs[5]["out"]
    assert "PRIMER TRADE DIFERENTE: nº 1, columnas: qty (|dif|=2e-05 > tol 1.5e-05)" in runs[5]["out"]
    assert runs[6]["ok"] is True, runs[6]["out"]
    assert runs[7]["ok"] is False, runs[7]["out"]
    assert "PRIMER TRADE DIFERENTE: nº 1, columnas: exit_price (|dif|=1e-07 > tol 2e-08)" in runs[7]["out"]
    assert runs[8]["ok"] is True, runs[8]["out"]


def test_qty_tolerance_does_not_depend_on_position_size(runs):
    # Valida QuantAgent-zd9n: la tolerancia de qty escala con la cantidad. Los 2e-6 acciones que pasan sobre
    # qty 10 fallan sobre qty 1. Y con las confianzas reales de CONFS, al 5% y al 100% del capital (la
    # cantidad cambia 20 veces), la diferencia medida entre motor y libreria pasa y la misma cantidad alterada
    # en 0,01% hacia arriba o hacia abajo falla.
    assert runs[9]["ok"] is False, runs[9]["out"]
    assert "qty (|dif|=2e-06 > tol 1.5e-06)" in runs[9]["out"]
    assert runs[10] == [[True, False, False]] * 4
