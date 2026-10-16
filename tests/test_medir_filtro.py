import numpy as np
import pandas as pd
import pytest

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


@pytest.fixture
def comparison_csvs(tmp_path):
    keys = [(s, a) for s in script.STRATEGIES for a in script.SYMBOLS]
    table = pd.DataFrame(keys, columns=["strategy", "symbol"])
    table = table.assign(desde="2007-01-01", hasta="2018-12-31", trades=np.arange(48), sharpe_tasa_0=np.arange(48))
    paths = [tmp_path / f"{name}.csv" for name in ("filter", "open", "close")]
    table.to_csv(paths[0], index=False)
    motor = table.rename(columns={"strategy": "estrategia", "symbol": "activo"})
    motor.iloc[::-1].to_csv(paths[1], index=False)
    motor.assign(sharpe_tasa_0=-np.arange(48), trades=0).to_csv(paths[2], index=False)
    return paths


def test_comparison_ranks_keys_and_top_five(comparison_csvs, capsys):
    script.comparar(*comparison_csvs)
    output = capsys.readouterr().out
    assert "Spearman apertura: 1.000000" in output
    assert "Spearman cierre: -1.000000" in output
    assert "Top cinco apertura" in output and "Top cinco cierre" in output
    top = output.split("Top cinco cierre por diferencia absoluta:\n")[1]
    assert [int(line.split()[-1]) for line in top.splitlines()[1:]] == [47, 46, 45, 44, 43]
    for path, values in zip(comparison_csvs[:2], ([0] * 24 + [1] * 24, range(48))):
        table = pd.read_csv(path).sort_values(list(pd.read_csv(path).columns[:2]))
        table["sharpe_tasa_0"] = list(values)
        table.to_csv(path, index=False)
    script.comparar(*comparison_csvs)
    assert "Spearman apertura: 0.866213" in capsys.readouterr().out


def test_comparison_accepts_motor_first_candle(comparison_csvs, capsys):
    for path in comparison_csvs[1:]:
        table = pd.read_csv(path)
        table.assign(desde="2007-01-03").to_csv(path, index=False)
    script.comparar(*comparison_csvs)
    output = capsys.readouterr().out
    assert "Spearman apertura: 1.000000" in output
    assert "Spearman cierre: -1.000000" in output


@pytest.mark.parametrize("position", range(3))
@pytest.mark.parametrize("damage", ["missing", "duplicate", "early", "late"])
def test_reject_invalid_comparison(comparison_csvs, position, damage):
    path = comparison_csvs[position]
    table = pd.read_csv(path)
    if damage == "missing":
        table = table.iloc[:-1]
    elif damage == "duplicate":
        table.iloc[0] = table.iloc[1]
    else:
        table.loc[0, "desde" if damage == "early" else "hasta"] = "2006-12-31" if damage == "early" else "2019-01-01"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="48 claves únicas|período requerido"):
        script.comparar(*comparison_csvs)
