"""Tests para scripts/compare_control.py (V23b, QuantAgent-k6n9)."""

import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("backtesting")
from click.testing import CliRunner
from sqlalchemy import create_engine

import quantagent.database as database
from quantagent import settings
from quantagent.backtesting.fixtures import FIXTURES_DIR
from quantagent.cli.backtest import backtest_group
from quantagent.data.snapshot import write_snapshot
from quantagent.models import Base

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "compare_control.py"
sys.path.insert(0, str(ROOT / "scripts"))
import compare_control as control


def test_metricas_curva_calculo_manual():
    equity = pd.Series([100.0, 110.0, 99.0, 108.9], index=pd.date_range("2020-01-01", periods=4))
    # Retornos 0.1, -0.1, 0.1; media 1/30, varianza muestral 1/75.
    sharpe, drawdown = control.metricas_curva(equity)
    assert sharpe == pytest.approx((1 / 30) / (1 / 75) ** 0.5 * 365.25 ** 0.5, abs=1e-9, rel=0)
    assert drawdown == pytest.approx(0.1, abs=1e-9, rel=0)


@pytest.mark.parametrize("alias,costos,expected", [
    ("rsi", "cripto", 46), ("rsi", "etf", 46), ("fifty-two-week-high", "etf", 6),
])
def test_control_cortos_al_cien_por_ciento(snap_env, alias, costos, expected):
    runner, tmp_path = snap_env
    write_snapshot("t", {"SPY": _daily_frame()})
    tabla, trades = tmp_path / "cmp.csv", tmp_path / "trades"
    result = runner.invoke(backtest_group, [
        "compare", "--llenado", "cierre", "--strategies", alias, "--symbols", "SPY", "--snapshot", "t",
        "--costos", costos, "--tamano-posicion", "1.0", "--out", str(tabla), "--trades-dir", str(trades),
    ])
    assert result.exit_code == 0, result.output
    _run_control(tabla, trades, "t", None, "--costos", costos)
    with tabla.open(newline="") as f:
        assert next(csv.DictReader(f))["control"] == f"coinciden {expected} de {expected}"


def test_largo_mayor_que_efectivo_sigue_cancelado(monkeypatch):
    from quantagent.strategy.base import TradingSignal

    class LongOverCash:
        required_history_bars = 30
        sent = False

        def generate_signal(self, window, symbol, timeframe, price):
            if self.sent:
                return None
            self.sent = True
            return TradingSignal(decision="LONG", confidence=1.0,
                                 entry_price=price, stop_loss=price * 0.5, take_profit=price * 2)

    monkeypatch.setattr(settings, "TRADING_BASE_POSITION_PCT", 1.0001)
    df = control.port.load_fixture(FIXTURES_DIR / "spy-2y-1d.csv").iloc[:32].copy()
    with pytest.warns(UserWarning, match="insufficient margin"):
        stats = control.bt_adapter.run_adapter(LongOverCash(), df, symbol="SPY", timeframe="1d",
                                               commission_pct=0.001, slippage_pct=0.0005)
    assert stats["# Trades"] == 0
    assert not stats["_strategy"].trades
    assert stats["_strategy"]._broker._cash == control.bt_adapter.INITIAL_CASH


def _daily_frame() -> pd.DataFrame:
    with (FIXTURES_DIR / "spy-2y-1d.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    df = pd.DataFrame(
        {
            "open": [float(r["open"]) for r in rows],
            "high": [float(r["high"]) for r in rows],
            "low": [float(r["low"]) for r in rows],
            "close": [float(r["close"]) for r in rows],
            "volume": [int(r["volume"]) for r in rows],
        },
        index=pd.DatetimeIndex([r["timestamp"] for r in rows], name="timestamp"),
    )
    df.insert(4, "adj_close", df["close"])
    return df


@pytest.fixture
def snap_env(monkeypatch, tmp_path):
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path / "snaps"))
    monkeypatch.setenv("QUANTAGENT_RESERVA_DESDE", "2030-01-01")
    monkeypatch.setenv("QUANTAGENT_PRUEBA_DESDE", "2030-01-01")
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    original_database_url = settings.DATABASE_URL
    try:
        yield CliRunner(), tmp_path
    finally:
        settings.DATABASE_URL = original_database_url
        database._engine = None
        database._SessionLocal = None


def _run_control(tabla: Path, trades_dir: Path, snapshot: str, md: Path | None = None, *extra: str) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--tabla", str(tabla),
        "--trades-dir", str(trades_dir),
        "--snapshot", snapshot,
        "--llenado", "cierre",
        *extra,
    ]
    if md:
        cmd.extend(["--md", str(md)])
    subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )


def test_compare_control_coincide_y_falla_si_se_altera_un_trade(snap_env):
    runner, tmp_path = snap_env
    write_snapshot("t", {"SPY": _daily_frame()})
    out_csv = tmp_path / "cmp.csv"
    trades_dir = tmp_path / "trades"

    res = runner.invoke(
        backtest_group,
        [
            "compare", "--llenado", "cierre",
            "--strategies", "fifty-two-week-high",
            "--symbols", "SPY",
            "--snapshot", "t",
            "--out", str(out_csv),
            "--trades-dir", str(trades_dir),
        ],
    )
    assert res.exit_code == 0, res.output

    # Caso 1: corrida normal coincide N de N y genera markdown si se pide
    out_md = tmp_path / "ctrl.md"
    _run_control(out_csv, trades_dir, "t", md=out_md)
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["control"] == "coinciden 6 de 6"
    assert out_md.exists()
    ctrl_md_text = out_md.read_text(encoding="utf-8")
    assert "coinciden 6 de 6" in ctrl_md_text
    assert "- **Snapshot:** t" in ctrl_md_text

    # Caso 2: alterando un trade del motor (pnl en fila 1) da sin validar nombrando fila y columna
    motor_file = trades_dir / "fifty-two-week-high-SPY.csv"
    with open(motor_file, newline="") as f:
        trade_rows = list(csv.DictReader(f))
    trade_rows[0]["pnl"] = str(float(trade_rows[0]["pnl"]) + 100.0)
    with open(motor_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trade_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trade_rows)

    _run_control(out_csv, trades_dir, "t")
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["control"] == "sin validar: fila 1, columna pnl"
    assert all(rows[0][c] == "" for c in control.METRIC_COLUMNS)

    # Caso 3: alterando exit_reason en fila 2
    trade_rows[0]["pnl"] = str(float(trade_rows[0]["pnl"]) - 100.0)  # restaurar fila 1
    trade_rows[1]["exit_reason"] = "OTRO_MOTIVO"
    with open(motor_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trade_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trade_rows)

    _run_control(out_csv, trades_dir, "t")
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["control"] == "sin validar: fila 2, columna exit_reason"

    # Caso 4: distinta cantidad de trades
    with open(motor_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trade_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trade_rows[:-1])

    _run_control(out_csv, trades_dir, "t")
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["control"] == "sin validar: distinta cantidad de trades (motor=5, port=6)"

    # Caso 5: falta archivo de trades
    motor_file.unlink()
    _run_control(out_csv, trades_dir, "t")
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["control"] == "sin validar: falta archivo fifty-two-week-high-SPY.csv"


def test_compare_control_sensibilidad_apertura_siguiente_gap(snap_env, monkeypatch):
    runner, tmp_path = snap_env
    df = _daily_frame().iloc[:330].copy()
    close_signal = df.loc[df.index[320], "close"]
    gap = 10.0
    open_next = close_signal + gap
    df.loc[df.index[321], "open"] = open_next
    df.loc[df.index[321], "high"] = max(df.loc[df.index[321], "high"], open_next)

    write_snapshot("t_gap", {"SPY": df})
    out_csv = tmp_path / "cmp_gap.csv"
    trades_dir = tmp_path / "trades_gap"

    res = runner.invoke(
        backtest_group,
        [
            "compare", "--llenado", "cierre",
            "--strategies", "fifty-two-week-high",
            "--symbols", "SPY",
            "--snapshot", "t_gap",
            "--out", str(out_csv),
            "--trades-dir", str(trades_dir),
        ],
    )
    assert res.exit_code == 0, res.output

    _run_control(out_csv, trades_dir, "t_gap")
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["control"] == "coinciden 1 de 1"
    assert rows[0]["trades"] == "1"
    assert rows[0]["trades_apertura_siguiente"] == "1"

    pnl_cierre = float(rows[0]["pnl"])
    pnl_apertura = float(rows[0]["pnl_apertura_siguiente"])
    assert pnl_apertura < pnl_cierre

    with open(trades_dir / "fifty-two-week-high-SPY.csv", newline="") as f:
        trade = next(csv.DictReader(f))
    slippage = float(settings.TRADING_SLIPPAGE_PCT)
    high_52w = df["high"].iloc[320 - 252:320].max()
    confidence = max(0.1, min(1.0, (close_signal - high_52w) / (high_52w + 1e-10) * 10))
    manual_pnl = []
    for reference in (close_signal, open_next):
        qty = round(100000 * 1.0 * confidence / reference * 2**27) / 2**27
        entry = reference * (1 + slippage)
        exit_price = float(trade["exit_price"])
        manual_pnl.append(qty * (exit_price - entry) - settings.TRADING_COMMISSION_PCT * qty * (entry + exit_price))
    assert float(trade["qty"]) == pytest.approx(100000 * confidence / close_signal, abs=1e-8)
    assert pnl_cierre == pytest.approx(round(manual_pnl[0], 2), abs=0.01)
    assert pnl_apertura == pytest.approx(round(manual_pnl[1], 2), abs=0.01)

    metrics = []
    adapter_df = control.port.load_fixture(snapshot="t_gap", symbol="SPY")
    monkeypatch.setattr(settings, "TRADING_BASE_POSITION_PCT", 1.0)
    for fill in ("close", "next-open"):
        stats = control.bt_adapter.run_adapter(control.build_strategy("FiftyTwoWeekHighStrategy"), adapter_df.copy(), symbol="SPY", timeframe="1d", fill=fill, commission_pct=settings.TRADING_COMMISSION_PCT, slippage_pct=settings.TRADING_SLIPPAGE_PCT)
        equity = control.bt_adapter.equity_curve(stats)
        returns = equity.pct_change().dropna()
        periods = (len(equity) - 1) * 365.25 * 86400 / (equity.index[-1] - equity.index[0]).total_seconds()
        metrics.append((float(returns.mean() / returns.std() * periods ** 0.5), float((1 - equity / equity.cummax()).max())))
    ds, dd = (metrics[1][i] - metrics[0][i] for i in (0, 1))
    row = rows[0]
    assert list(row)[-4:] == list(control.METRIC_COLUMNS)
    assert float(row["ajuste_sharpe_apertura"]) == round(ds, 2)
    assert float(row["ajuste_drawdown_apertura"]) == round(dd, 6)
    assert float(row["sharpe_tasa_0_apertura_siguiente"]) == round(float(row["sharpe_tasa_0"]) + ds, 2)
    assert float(row["max_drawdown_apertura_siguiente"]) == round(max(0, float(row["max_drawdown"]) + dd), 6)
    row["sharpe_tasa_0"] = ""
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    _run_control(out_csv, trades_dir, "t_gap")
    with out_csv.open(newline="") as f:
        assert all(next_row[c] == "" for next_row in csv.DictReader(f) for c in control.METRIC_COLUMNS)
    row["sharpe_tasa_0"] = "0.5"
    run_adapter = control.bt_adapter.run_adapter
    for failed_fill in ("close", "next-open"):
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        def fail_run(*args, **kwargs):
            if kwargs["fill"] == failed_fill:
                raise RuntimeError("corrida fallida")
            return run_adapter(*args, **kwargs)
        monkeypatch.setattr(control.bt_adapter, "run_adapter", fail_run)
        assert control.main(["--tabla", str(out_csv), "--trades-dir", str(trades_dir), "--snapshot", "t_gap", "--llenado", "cierre"]) == 0
        with out_csv.open(newline="") as f:
            assert all(next_row[c] == "" for next_row in csv.DictReader(f) for c in control.METRIC_COLUMNS)


def test_compare_control_con_params_valida_contra_las_mismas_estrategias(snap_env):
    runner, tmp_path = snap_env
    write_snapshot("t", {"SPY": _daily_frame()})
    params = tmp_path / "p.json"
    params.write_text('{"fifty-two-week-high": {"stop_loss_pct": 0.01, "take_profit_pct": 0.02, "trailing_stop_pct": 0.02}}')
    out_csv, trades_dir, out_md = tmp_path / "cmp.csv", tmp_path / "trades", tmp_path / "ctrl.md"
    cmd = ["compare", "--llenado", "cierre", "--strategies", "fifty-two-week-high", "--symbols", "SPY", "--snapshot", "t", "--out", str(out_csv)]
    res = runner.invoke(backtest_group, [*cmd, "--trades-dir", str(trades_dir), "--params", str(params)])
    assert res.exit_code == 0, res.output

    def control() -> str:
        with open(out_csv, newline="") as f:
            return next(csv.DictReader(f))["control"]

    _run_control(out_csv, trades_dir, "t")
    assert control().startswith("sin validar")
    _run_control(out_csv, trades_dir, "t", out_md, "--params", str(params))
    assert control().startswith("coinciden")
    assert "take_profit_pct=0.02," in out_md.read_text(encoding="utf-8")


def test_llenado_invalido():
    result = subprocess.run([sys.executable, str(SCRIPT), "--tabla", "x", "--trades-dir", "x",
                             "--snapshot", "x", "--llenado", "otro"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


@pytest.mark.parametrize("llenado", [None, "cierre"])
def test_control_mismo_llenado_motor(snap_env, llenado):
    runner, tmp_path = snap_env
    write_snapshot("t", {"SPY": _daily_frame()})
    tabla, trades = tmp_path / "cmp.csv", tmp_path / "trades"
    extra = ["--llenado", llenado] if llenado else []
    result = runner.invoke(backtest_group, ["compare", "--strategies", "fifty-two-week-high",
        "--symbols", "SPY", "--snapshot", "t", "--costos", "etf",
        "--out", str(tabla), "--trades-dir", str(trades), *extra])
    assert result.exit_code == 0, result.output
    assert control.main(["--tabla", str(tabla), "--trades-dir", str(trades),
                         "--snapshot", "t", "--costos", "etf", *extra]) == 0
    with tabla.open(newline="") as f:
        row = next(csv.DictReader(f))
    assert row["control"] == "coinciden 6 de 6"
    if llenado is None:
        assert row["sharpe_tasa_0_apertura_siguiente"] == row["sharpe_tasa_0"]
        assert float(row["max_drawdown_apertura_siguiente"]) == float(row["max_drawdown"])


def test_compare_control_con_orden_minima(snap_env):
    runner, tmp_path = snap_env
    write_snapshot("t", {"SPY": _daily_frame()})
    out_csv, trades_dir = tmp_path / "cmp.csv", tmp_path / "trades"
    cmd = [
        "compare", "--llenado", "cierre",
        "--strategies", "fifty-two-week-high",
        "--symbols", "SPY",
        "--snapshot", "t",
        "--orden-minima", "50",
        "--out", str(out_csv),
        "--trades-dir", str(trades_dir),
    ]
    res = runner.invoke(backtest_group, cmd)
    assert res.exit_code == 0, res.output

    _run_control(out_csv, trades_dir, "t", None, "--orden-minima", "50")
    with open(out_csv, newline="") as f:
        row = next(csv.DictReader(f))
    assert row["control"] == "coinciden 6 de 6"


@pytest.mark.parametrize("alias, costos", [
    ("fifty-two-week-high", None),  # solo entradas al cierre; las salidas son STOP_LOSS (QuantAgent-jzb6)
    ("sma-cross", "cripto"),  # salida por regla (SMA_CROSS_DOWN) y backtest_end, con comision
])
def test_equity_curve_del_adaptador_iguala_la_del_motor(snap_env, monkeypatch, alias, costos):
    """QuantAgent-mxy7: con llenado al cierre, equity_curve descuenta el costo en la vela de la orden, como el motor."""
    from quantagent.backtesting import backtest as motor

    runner, tmp_path = snap_env
    write_snapshot("t", {"SPY": _daily_frame()})
    motores = []
    run = motor.Backtest.run
    monkeypatch.setattr(motor.Backtest, "run", lambda self, *a, **k: (motores.append(self), run(self, *a, **k))[1])
    extra = ["--costos", costos] if costos else []
    res = runner.invoke(backtest_group, ["compare", "--llenado", "cierre", "--strategies", alias, "--symbols", "SPY", "--snapshot", "t",
                                         "--out", str(tmp_path / "cmp.csv"), "--trades-dir", str(tmp_path / "tr"), *extra])
    assert res.exit_code == 0, res.output
    with open(tmp_path / "cmp.csv", newline="") as f:
        sharpe_motor = float(next(csv.DictReader(f))["sharpe_tasa_0"])
    curva_motor = motores[-1].get_equity_curve()["equity"].to_numpy()

    monkeypatch.setattr(settings, "TRADING_BASE_POSITION_PCT", 1.0)
    tasas = settings.COSTOS_PERFILES[costos] if costos else {
        "commission_pct": settings.TRADING_COMMISSION_PCT, "slippage_pct": settings.TRADING_SLIPPAGE_PCT}
    df = control.port.load_fixture(snapshot="t", symbol="SPY")
    stats = {fill: control.bt_adapter.run_adapter(control.build_strategy(control.cli_aliases()[alias]), df.copy(),
                                                  symbol="SPY", timeframe="1d", fill=fill, **tasas)
             for fill in ("close", "next-open")}

    curva = control.bt_adapter.equity_curve(stats["close"])
    sin_corregir = stats["close"]["_equity_curve"]["Equity"].iloc[:len(curva)].to_numpy()
    assert len(curva) == len(curva_motor) == 520
    assert abs(sin_corregir - curva_motor).max() > 1.0  # el corrimiento existe en esta celda
    assert abs(curva.to_numpy() - curva_motor).max() < 1e-4
    assert round(control.metricas_curva(curva)[0], 2) == sharpe_motor

    abierta = stats["next-open"]
    curva_open = control.bt_adapter.equity_curve(abierta)
    pd.testing.assert_series_equal(curva_open, abierta["_equity_curve"]["Equity"].iloc[:abierta["_strategy"].n_real_bars])
