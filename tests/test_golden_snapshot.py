"""QuantAgent-688: golden de rsi sobre SPY diario real 2007-2018, con costos etf y 100 % del capital.

Los goldens congelados viven en $QUANTAGENT_SNAPSHOT_DIR/golden/etf-1d-2026-10/.
Regeneración manual (base SQLite inicializada con init_db()):

    python -m quantagent.cli backtest run --strategy rsi --snapshot etf-1d-2026-10 --symbol SPY \
        --from 2007-01-03 --to 2018-12-31 --costos etf --tamano-posicion 1.0 \
        --llenado apertura \
        --out "$QUANTAGENT_SNAPSHOT_DIR/golden/etf-1d-2026-10/rsi-SPY-2007-2018-etf-100-apertura.csv"

Para cierre, usar --llenado cierre y el nombre rsi-SPY-2007-2018-etf-100.csv.

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
SNAPSHOT = "etf-1d-2026-10"
RUN = ["--strategy", "rsi", "--snapshot", SNAPSHOT, "--symbol", "SPY", "--from", "2007-01-03", "--to", "2018-12-31",
       "--costos", "etf"]
SNAPSHOT_DIR = os.environ.get("QUANTAGENT_SNAPSHOT_DIR")
if not SNAPSHOT_DIR:
    pytest.skip(f"Falta QUANTAGENT_SNAPSHOT_DIR (debe apuntar a local-data/ con el snapshot {SNAPSHOT})", allow_module_level=True)
if not (Path(SNAPSHOT_DIR) / SNAPSHOT).exists():
    pytest.skip(f"Falta el snapshot {SNAPSHOT} en {SNAPSHOT_DIR}", allow_module_level=True)

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


@pytest.mark.parametrize("fill,trades,pnl,sharpe", [("apertura", 181, -21841.01, -1.23), ("cierre", 187, -9141.21, None)])
def test_motor_igual_al_golden_y_al_adaptador(tmp_path, fill, trades, pnl, sharpe):
    """Cada llenado reproduce su golden externo y el control; apertura es el default."""
    suffix = "-apertura" if fill == "apertura" else ""
    golden = Path(SNAPSHOT_DIR) / "golden" / SNAPSHOT / f"rsi-SPY-2007-2018-etf-100{suffix}.csv"
    if not golden.exists():
        pytest.skip(
            "Falta golden; generar con python -m quantagent.cli backtest run "
            + " ".join(RUN) + f' --llenado {fill} --tamano-posicion 1.0 --out "{golden}"'
        )
    (tmp_path / "engine.db").touch()
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{tmp_path / 'engine.db'}", "TRADING_BASE_POSITION_PCT": "1.0"}
    motor, adaptador = str(tmp_path / "motor.csv"), str(tmp_path / "adaptador.csv")
    equity = tmp_path / "equity.csv"
    outputs = []
    for cmd in (["-c", "from quantagent.database import init_db; init_db()"],
                ["-m", "quantagent.cli", "backtest", "run", *RUN, *([] if fill == "apertura" else ["--llenado", fill]),
                 "--tamano-posicion", "1.0", "--out", motor, "--equity-out", str(equity)],
                ["scripts/bt_adapter.py", *RUN, "--fill", "next-open" if fill == "apertura" else "close", "--out", adaptador],
                ["scripts/recalc_metrics.py", motor, "--equity", str(equity)],
                ["-c", COMPARE, motor, adaptador]):
        proc = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
        if cmd[1] != COMPARE:
            assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
        outputs.append(proc.stdout)

    actual, expected = Path(motor).read_bytes(), golden.read_bytes()
    if actual != expected:
        pytest.fail(f"El CSV del motor difiere del golden en {first_difference(actual, expected)}")

    with open(motor, newline="") as f:
        pnls = [float(row["pnl"]) for row in csv.DictReader(f)]
    assert (len(pnls), round(sum(pnls), 2)) == (trades, pnl)
    for line in (f"Trades: {trades}", f"Total PnL: {pnl:.2f}"):
        assert line in outputs[1] and line in outputs[3]
    cli_sharpe = next(line for line in outputs[1].splitlines() if line.startswith("Sharpe ratio:"))
    assert cli_sharpe in outputs[3]
    if sharpe is not None:
        assert cli_sharpe == f"Sharpe ratio: {sharpe:.2f}"
    assert f"PnL por trade: {trades}/{trades} filas coinciden" in outputs[3]
    print(f"{fill}: {trades} trades / {pnl:.2f} / {cli_sharpe}; recálculo coincide")

    assert proc.returncode == 0, proc.stdout + proc.stderr[-2000:]
    assert f"Motivos de salida: coinciden {trades} de {trades}" in proc.stdout, proc.stdout


def test_golden_apertura_mutado_detecta_linea_y_columna(tmp_path, monkeypatch):
    name = "rsi-SPY-2007-2018-etf-100-apertura.csv"
    source = Path(SNAPSHOT_DIR) / "golden" / SNAPSHOT / name
    if not source.exists():
        pytest.skip("Falta golden de apertura; ver comando de regeneración en el docstring")
    lines = source.read_bytes().splitlines(keepends=True)
    column = lines[0].decode().strip().split(",").index("pnl")
    cells = lines[90].split(b",")
    original = cells[column]
    digit = next(i for i, value in enumerate(original) if 48 <= value <= 57)
    cells[column] = original[:digit] + (b"1" if original[digit:digit + 1] != b"1" else b"2") + original[digit + 1:]
    lines[90] = b",".join(cells)
    golden_dir = tmp_path / "golden" / SNAPSHOT
    golden_dir.mkdir(parents=True)
    (golden_dir / name).write_bytes(b"".join(lines))
    monkeypatch.setitem(globals(), "SNAPSHOT_DIR", str(tmp_path))
    with pytest.raises(pytest.fail.Exception) as error:
        test_motor_igual_al_golden_y_al_adaptador(tmp_path, "apertura", 181, -21841.01, -1.23)
    assert str(error.value) == "El CSV del motor difiere del golden en linea 91, columnas: pnl"
