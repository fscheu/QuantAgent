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
print(json.dumps([
    run([]),
    run(["--port-pnl-delta", "5:0.5"]),
    run(["--intrabar"]),
    run(["--intrabar", "--port-pnl-delta", "5:0.5"]),
    run(["--intrabar", "--engine-pnl-delta", "5:0.5"]),
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
