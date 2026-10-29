"""Cross-validation: motor, vbt_filter y backtesting.py (QuantAgent-fnpr.3.4).

Verifica coincidencia trade por trade (fecha y precio entrada/salida) en fixture con
gap, intravela y vencimiento. Subproceso aislado por crossval_rsi.
Guía PM (#554): salidas en barras posteriores para no depender de la semántica de
órdenes contingentes evaluadas en la misma barra del llenado stop.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.oraculo

CANDLES = [
    (1, 90.0, 92.0, 89.0, 91.0), (2, 91.0, 93.0, 90.0, 92.0),
    (3, 91.0, 93.0, 90.0, 92.0), (4, 92.0, 94.0, 91.0, 93.0),
    (5, 93.0, 105.0, 92.0, 94.0), (6, 94.0, 95.0, 93.0, 94.0),
    (7, 95.0, 103.0, 94.0, 98.0), (8, 98.0, 102.0, 97.0, 101.0),
    (9, 102.0, 104.0, 101.0, 103.0), (10, 95.0, 96.0, 94.0, 95.0),
    (11, 105.0, 110.0, 104.0, 108.0), (12, 108.0, 112.0, 107.0, 111.0),
    (13, 110.0, 112.0, 109.0, 111.0), (14, 110.0, 111.0, 109.0, 110.0),
]


def _make_df():
    import pandas as pd
    start = datetime(2026, 3, 1)
    rows = [{"timestamp": start + timedelta(days=d - 1), "open": o, "high": h, "low": l, "close": c, "volume": 1000}
            for d, o, h, l, c in CANDLES]
    df = pd.DataFrame(rows)
    df.index = df["timestamp"]
    return df


def _run_subproc(workdir: Path) -> dict:
    import numpy as np
    import pandas as pd
    sys.path.extend([str(ROOT), str(ROOT / "scripts")])
    import bt_adapter
    from quantagent.backtesting.backtest import Backtest
    from quantagent.backtesting.signals import bar_signals
    from quantagent.backtesting.vbt_filter import run_filter
    from quantagent.database import SessionLocal, init_db
    from quantagent.models import MarketData, Trade
    from quantagent.strategy.base import TradingSignal, TradingStrategy

    class CrossvalStopStrategy(TradingStrategy):
        required_history_bars = 1

        def generate_signal(self, kline_data, symbol, timeframe, current_price):
            day = (kline_data[-1]["timestamp"]).day
            if day in (2, 6, 10):
                return TradingSignal(decision="LONG", confidence=1.0, entry_order="stop", entry_price=100.0,
                                     entry_expiry_bars=2, stop_loss=80.0, take_profit=200.0)
            return TradingSignal(decision="HOLD", confidence=0.0)

        def should_exit(self, position, current_price, ohlc_data):
            day = (ohlc_data["timestamp"].iloc[-1] if "timestamp" in ohlc_data else ohlc_data.index[-1]).day
            return (True, "REGLA_PROPIA") if day in (8, 12) else (False, "")

        def should_reevaluate(self, position, current_price):
            return False

    df, sym = _make_df(), "CROSSVAL"
    # 1. Motor propio
    os.environ["DATABASE_URL"] = f"sqlite:///{workdir / 'engine.db'}"
    init_db()
    session = SessionLocal()
    session.query(MarketData).filter(MarketData.symbol == sym).delete()
    for ts, r in df.iterrows():
        session.add(MarketData(symbol=sym, timeframe="1d", timestamp=ts.to_pydatetime(),
                               open=r["open"], high=r["high"], low=r["low"], close=r["close"], volume=1000))
    session.commit()
    bt = Backtest(start_date=df.index[0].to_pydatetime(), end_date=df.index[-1].to_pydatetime(),
                  assets=[sym], timeframe="1d", fill="apertura",
                  config={"offline_data": True, "market_hours_filter": False, "slippage_pct": 0.0, "commission_pct": 0.0},
                  db_session=session, strategy=CrossvalStopStrategy())
    bt.run()
    m_trades = session.query(Trade).filter(Trade.backtest_run_id == bt.backtest_run_id).order_by(Trade.opened_at).all()
    motor = [{"entry_time": t.opened_at.strftime("%Y-%m-%d"), "entry_price": float(t.entry_price),
              "exit_time": t.closed_at.strftime("%Y-%m-%d"), "exit_price": float(t.exit_price)} for t in m_trades]
    session.close()

    # 2. vbt_filter
    sig = bar_signals(CrossvalStopStrategy(), df, sym, "1d")
    sig["entry_order"], sig["entry_price"], sig["entry_expiry_bars"] = "market", np.nan, np.nan
    for i, (d, *_) in enumerate(CANDLES):
        if d in (2, 6, 10):
            sig.iloc[i, sig.columns.get_loc("entry_order")] = "stop"
            sig.iloc[i, sig.columns.get_loc("entry_price")] = 100.0
            sig.iloc[i, sig.columns.get_loc("entry_expiry_bars")] = 2
    r_vbt = run_filter(sig, df, commission_pct=0.0, slippage_pct=0.0, capital=100_000.0)
    vbt = [{"entry_time": pd.Timestamp(r["Entry Timestamp"]).strftime("%Y-%m-%d"),
            "entry_price": float(r["Avg Entry Price"]),
            "exit_time": pd.Timestamp(r["Exit Timestamp"]).strftime("%Y-%m-%d"),
            "exit_price": float(r["Avg Exit Price"])} for _, r in r_vbt["portfolio"].trades.records_readable.iterrows()]

    # 3. backtesting.py
    df_bt = pd.DataFrame({c.capitalize(): df[c] / bt_adapter.SCALE for c in ["open", "high", "low", "close"]}, index=df.index)
    df_bt["Volume"] = df["volume"]
    sent = df_bt.iloc[-1:].copy()
    sent.index = sent.index + pd.Timedelta(days=1)
    stats = bt_adapter.run_adapter(CrossvalStopStrategy(), pd.concat([df_bt, sent]), symbol=sym, timeframe="1d", fill="next-open")
    bt_res = [{"entry_time": pd.Timestamp(r.entry_time).strftime("%Y-%m-%d"), "entry_price": float(r.entry_price),
               "exit_time": pd.Timestamp(r.exit_time).strftime("%Y-%m-%d"), "exit_price": float(r.exit_price)}
              for r in bt_adapter.trade_rows(stats, sym)]

    return {"motor": motor, "vbt": vbt, "bt": bt_res}


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    pytest.importorskip("backtesting")
    pytest.importorskip("vectorbt")
    proc = subprocess.run([sys.executable, __file__, str(tmp_path_factory.mktemp("crossval_stop"))],
                          cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.splitlines()[-1])


def test_motor_vbt_y_backtesting_coinciden_trade_por_trade(results):
    motor, vbt, bt = results["motor"], results["vbt"], results["bt"]
    assert len(motor) == 2 and motor == vbt == bt
    # Trade 1: intravela
    assert (motor[0]["entry_time"], motor[0]["entry_price"]) == ("2026-03-07", pytest.approx(100.0))
    assert (motor[0]["exit_time"], motor[0]["exit_price"]) == ("2026-03-09", pytest.approx(102.0))
    # Trade 2: gap
    assert (motor[1]["entry_time"], motor[1]["entry_price"]) == ("2026-03-11", pytest.approx(105.0))
    assert (motor[1]["exit_time"], motor[1]["exit_price"]) == ("2026-03-13", pytest.approx(110.0))


if __name__ == "__main__":
    workdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp")
    print(json.dumps(_run_subproc(workdir)))
