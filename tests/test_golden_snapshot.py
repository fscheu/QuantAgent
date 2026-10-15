"""QuantAgent-688: golden de rsi sobre SPY diario real 2007-2018, con costos etf y 100 % del capital.

El golden vive fuera del repo (los datos no se publican), en
$QUANTAGENT_SNAPSHOT_DIR/golden/etf-1d-2026-10/rsi-SPY-2007-2018-etf-100.csv. Se genera una vez, con
DATABASE_URL apuntando a una base SQLite inicializada con init_db():

    mkdir -p $QUANTAGENT_SNAPSHOT_DIR/golden/etf-1d-2026-10
    python -m quantagent.cli backtest run --strategy rsi --snapshot etf-1d-2026-10 --symbol SPY \
        --from 2007-01-03 --to 2018-12-31 --costos etf --tamano-posicion 1.0 --llenado cierre \
        --out $QUANTAGENT_SNAPSHOT_DIR/golden/etf-1d-2026-10/rsi-SPY-2007-2018-etf-100.csv

Los mensajes de falla dicen linea y columna pero no muestran filas: los precios no deben salir en un log.
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.snapshot, pytest.mark.oraculo]

pytest.importorskip("backtesting")

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT, GOLDEN_REL = "etf-1d-2026-10", "golden/etf-1d-2026-10/rsi-SPY-2007-2018-etf-100.csv"
RUN = ["--strategy", "rsi", "--snapshot", SNAPSHOT, "--symbol", "SPY", "--from", "2007-01-03", "--to", "2018-12-31",
       "--costos", "etf"]
SNAPSHOT_DIR = os.environ.get("QUANTAGENT_SNAPSHOT_DIR")
if not SNAPSHOT_DIR:
    pytest.skip(f"Falta QUANTAGENT_SNAPSHOT_DIR (debe apuntar a local-data/ con el snapshot {SNAPSHOT})", allow_module_level=True)
if not (Path(SNAPSHOT_DIR) / SNAPSHOT).exists():
    pytest.skip(f"Falta el snapshot {SNAPSHOT} en {SNAPSHOT_DIR}", allow_module_level=True)
GOLDEN = Path(SNAPSHOT_DIR) / GOLDEN_REL
if not GOLDEN.exists():
    pytest.skip(f"Falta el golden {GOLDEN}. Generarlo: mkdir -p {GOLDEN.parent} && python -m quantagent.cli backtest "
                f"run {' '.join(RUN)} --llenado cierre --tamano-posicion 1.0 --out {GOLDEN}", allow_module_level=True)

# Imprime solo los renglones de compare_trades sin filas, y cuantos motivos de salida coinciden.
COMPARE = ("import sys; sys.path.insert(0, 'scripts'); import crossval_compare as c; "
           "e, a = c.read_rows(sys.argv[1]), c.read_rows(sys.argv[2]); lines, ok = c.compare_trades(e, a); "
           "same = sum(x['exit_reason'] == y['exit_reason'] for x, y in zip(e, a)); "
           "print(*[l for l in lines if not l.startswith('  ')], sep='\\n'); "
           "print(f'Motivos de salida: coinciden {same} de {len(e)}'); sys.exit(not ok)")


def first_difference(actual: bytes, golden: bytes) -> str:
    """Primera linea (y columnas) en que dos CSV difieren, sin mostrar los valores."""
    a, g = actual.decode().splitlines(), golden.decode().splitlines()
    header = g[0].split(",")
    for n, (x, y) in enumerate(zip(a, g), start=1):
        if x != y:
            cols = [c for c, p, q in zip(header, x.split(","), y.split(",")) if p != q]
            return f"linea {n}, columnas: {', '.join(cols) or 'cantidad de columnas'}"
    if len(a) != len(g):
        return f"linea {min(len(a), len(g)) + 1}: el motor tiene {len(a)} lineas y el golden {len(g)}"
    return "solo en los saltos de linea o en el final del archivo"


def test_motor_igual_al_golden_y_al_adaptador(tmp_path):
    """El motor reproduce el golden byte a byte, da 187 trades y PnL -9141.21, y el adaptador coincide."""
    (tmp_path / "engine.db").touch()
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{tmp_path / 'engine.db'}", "TRADING_BASE_POSITION_PCT": "1.0"}
    motor, adaptador = str(tmp_path / "motor.csv"), str(tmp_path / "adaptador.csv")
    for cmd in (["-c", "from quantagent.database import init_db; init_db()"],
                ["-m", "quantagent.cli", "backtest", "run", *RUN, "--llenado", "cierre", "--tamano-posicion", "1.0", "--out", motor],
                ["scripts/bt_adapter.py", *RUN, "--fill", "close", "--out", adaptador],
                ["-c", COMPARE, motor, adaptador]):
        proc = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
        if cmd[1] != COMPARE:
            assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]

    actual, golden = Path(motor).read_bytes(), GOLDEN.read_bytes()
    if actual != golden:
        pytest.fail(f"El CSV del motor difiere del golden {GOLDEN} en {first_difference(actual, golden)}")

    with open(motor, newline="") as f:
        pnls = [float(row["pnl"]) for row in csv.DictReader(f)]
    assert (len(pnls), round(sum(pnls), 2)) == (187, -9141.21)

    assert proc.returncode == 0, proc.stdout + proc.stderr[-2000:]
    assert "Motivos de salida: coinciden 187 de 187" in proc.stdout, proc.stdout
