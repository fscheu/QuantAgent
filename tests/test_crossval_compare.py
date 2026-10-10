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
DRIVER = """
import contextlib, io, json, sys
sys.path.insert(0, "scripts")
import crossval_compare as c
def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return {"rc": c.main(argv), "out": buf.getvalue()}
def altered(col, delta):  # un trade inventado: el port difiere del motor solo en `col`, en `delta` absoluto
    row = {"entry_time": "t0", "exit_time": "t1", "symbol": "SPY", "side": "buy", "qty": "10",
           "entry_price": "100", "exit_price": "101", "pnl": "10"}
    lines, ok = c.compare_trades([row], [{**row, col: repr(float(row[col]) + delta)}])
    return {"ok": ok, "out": " / ".join(lines)}
print(json.dumps([
    run([]),
    run(["--port-pnl-delta", "5:0.5"]),
    run(["--intrabar"]),
    run(["--intrabar", "--port-pnl-delta", "5:0.5"]),
    run(["--intrabar", "--engine-pnl-delta", "5:0.5"]),
    altered("qty", 2e-6),
    altered("qty", 5e-7),
    altered("exit_price", 1e-7),
    altered("exit_price", 1e-8),
]))
"""


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
    # Valida QuantAgent-824 sobre un trade inventado (qty 10, salida 101): con tolerancia de qty 1e-6 acciones,
    # un qty alterado en 2e-6 falla y nombra la columna y uno en 5e-7 pasa; con tolerancia de precio 2e-8,
    # un exit_price alterado en 1e-7 falla y uno en 1e-8 (lo medido sobre SPY real) pasa.
    assert runs[5]["ok"] is False, runs[5]["out"]
    assert "PRIMER TRADE DIFERENTE: nº 1, columnas: qty (|dif|=2e-06 > tol 1e-06)" in runs[5]["out"]
    assert runs[6]["ok"] is True, runs[6]["out"]
    assert runs[7]["ok"] is False, runs[7]["out"]
    assert "PRIMER TRADE DIFERENTE: nº 1, columnas: exit_price (|dif|=1e-07 > tol 2e-08)" in runs[7]["out"]
    assert runs[8]["ok"] is True, runs[8]["out"]
