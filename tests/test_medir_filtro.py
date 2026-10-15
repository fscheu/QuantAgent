import numpy as np
import pandas as pd

from quantagent.data.snapshot import write_snapshot
from scripts import medir_filtro as script


def test_48_cells_metrics_and_protected_dates(tmp_path, monkeypatch, capsys):
    index = pd.date_range("2018-11-01", "2019-01-05")
    assert set(script.SYMBOLS) == {"SPY", "IWM", "EFA", "EEM", "TLT", "SHY", "GLD", "DBC", "XLE", "XLF", "XLK", "XLU"}
    close = 100 + 10 * np.sin(np.arange(len(index)) / 3)
    frame = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                          "close": close, "adj_close": close, "volume": 1000}, index=index)
    write_snapshot(script.SNAPSHOT, {s: frame for s in script.SYMBOLS}, base_dir=tmp_path)
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    calls = {"signals": 0, "simulation": 0}
    real_signals, real_filter = script.bar_signals, script.run_filter

    def signals(strategy, candles, symbol, timeframe):
        assert candles.index.min() == pd.Timestamp("2018-11-01")
        assert candles.index.max() == pd.Timestamp("2018-12-31")
        calls["signals"] += 1
        return real_signals(strategy, candles, symbol, timeframe)

    def simulate(signals, candles, **kwargs):
        assert candles.index.max() == pd.Timestamp("2018-12-31")
        assert kwargs == {"capital": 100_000.0, "commission_pct": 0.0, "slippage_pct": 0.0005}
        calls["simulation"] += 1
        return real_filter(signals, candles, **kwargs)

    monkeypatch.setattr(script, "bar_signals", signals)
    monkeypatch.setattr(script, "run_filter", simulate)
    out = tmp_path / "filter.csv"
    script.medir(out)
    rows = pd.read_csv(out)
    assert set(zip(rows.strategy, rows.symbol)) == {(s, a) for s in script.STRATEGIES for a in script.SYMBOLS}
    assert len(rows) == 48 and calls == {"signals": 48, "simulation": 48}
    assert (rows[["signals_s", "simulation_s"]] > 0).all().all()
    assert np.isfinite(rows[["trades", "pnl", "sharpe_tasa_0"]]).all().all()
    assert rows.loc[rows.strategy == "rsi", "trades"].min() > 0
    assert "Celdas: 48" in capsys.readouterr().out
