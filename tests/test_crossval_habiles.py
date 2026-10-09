"""QuantAgent-1xd: fixture con huecos de fin de semana y `--fixture` en el control con backtesting.py.

`tests/fixtures/spy-90d-habiles.csv` es `spy-90d.csv` sin sabados ni domingos (1536 filas). Se genero con:
python -c "import pandas as pd; d=pd.read_csv('tests/fixtures/spy-90d.csv'); d[pd.to_datetime(d.timestamp).dt.dayofweek<5].to_csv('tests/fixtures/spy-90d-habiles.csv', index=False)"

Los scripts corren en subproceso: crossval_rsi fija TRADING_SLIPPAGE_PCT=0 al importarse y no debe filtrarse.
Se omite sin `backtesting`.
"""

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("backtesting")

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"


def _lines(name: str) -> list[str]:
    return (FIXTURES / f"{name}.csv").read_text().splitlines()


def test_habiles_is_spy_90d_without_weekends():
    # Valida el fixture: mismas filas, byte a byte, que spy-90d salvo las de sabado y domingo.
    full, habiles = _lines("spy-90d"), _lines("spy-90d-habiles")
    expected = [full[0]] + [r for r in full[1:] if datetime.fromisoformat(r.split(",")[2]).weekday() < 5]
    assert habiles == expected
    assert len(habiles) - 1 == 1536


def test_port_reads_the_fixture_it_is_given(tmp_path):
    # Valida que --fixture llega al port: con spy-90d-habiles toda entrada y salida es una vela de ese fixture
    # (si el port leyera spy-90d, operaria en fines de semana, que existen en spy-90d).
    out = tmp_path / "port.csv"
    p = subprocess.run([sys.executable, "scripts/crossval_rsi.py", "--fixture", "spy-90d-habiles", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    stamps = {r.split(",")[2] for r in _lines("spy-90d-habiles")[1:]}
    rows = list(csv.DictReader(out.open()))
    assert rows
    assert {r["entry_time"] for r in rows} <= stamps
    assert {r["exit_time"] for r in rows} <= stamps


def test_weekend_entries_counts_saturday_and_sunday_only():
    # Valida la linea "Entradas en fin de semana": cuenta solo entry_time en sabado o domingo.
    rows = [{"entry_time": t} for t in ("2026-01-02T23:00:00", "2026-01-03T00:00:00",
                                         "2026-01-04T23:00:00", "2026-01-05T00:00:00")]
    code = f"import sys; sys.path.insert(0, 'scripts'); import crossval_compare as c; print(c.weekend_entries({rows!r}))"
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    assert json.loads(p.stdout.splitlines()[-1]) == 2
