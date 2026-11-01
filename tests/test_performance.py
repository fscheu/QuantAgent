import numpy as np
import pandas as pd
import pytest

from quantagent.performance import build_report


SUBJECT = {
    "type": "backtest", "id": "bt-1", "name": "Test", "env": None, "run_label": None,
    "account_id": None, "hypothesis_id": None, "backtest_id": "bt-1", "out_of_sample_start": None,
    "costs_bps": 0, "strategies": None,
}


def report(equity, spy=None, **kwargs):
    if spy is None:
        spy = equity
    return build_report(equity, spy, subject=SUBJECT, range="all", **kwargs)


def test_compounds_returns_and_marks_first_period_loss_as_drawdown():
    dates = pd.bdate_range("2024-01-01", periods=22)
    values = pd.Series([100, 90] + [99] * 20, index=dates)
    result = report(values)
    assert result["metrics"]["total_return"]["value"] == pytest.approx((99 / 100 - 1) * 100)
    assert result["metrics"]["max_drawdown"]["value"] == pytest.approx(-10)
    assert result["series"]["drawdown"][1]["dd"] == pytest.approx(-0.1)


def test_drawdown_includes_recovery_and_unrecovered_episode():
    dates = pd.bdate_range("2024-01-01", periods=5)
    result = report(pd.Series([100, 80, 100, 90, 70], index=dates))
    episodes = result["worst_drawdowns"]
    assert episodes[0]["depth"] == pytest.approx(-0.3)
    assert episodes[0]["recovery_date"] is None
    assert result["headline"]["max_dd_recovery_days"] is None


def test_excludes_incomplete_calendar_months():
    dates = pd.bdate_range("2024-01-15", "2024-03-15")
    values = pd.Series(np.arange(100, 100 + len(dates)), index=dates)
    result = report(values)
    assert [(r["year"], r["month"]) for r in result["monthly_returns"]] == [(2024, 2)]


def test_missing_spy_dates_are_not_filled_from_future():
    dates = pd.bdate_range("2024-01-01", periods=24)
    equity = pd.Series(np.arange(100, 124), index=dates)
    spy = pd.Series(np.arange(200, 223), index=dates.delete(5))
    result = report(equity, spy)
    assert all(point["date"] != dates[5].date().isoformat() for point in result["series"]["equity"])
    assert result["period"]["trading_days"] == len(spy) - 1


def test_each_threshold_controls_metric_computability():
    dates = pd.bdate_range("2024-01-01", periods=22)
    values = pd.Series(np.linspace(100, 110, len(dates)), index=dates)
    result = report(values, thresholds={"descriptive_days": 23, "risk_days": 24, "annualized_days": 25})
    assert result["metrics"]["total_return"]["status"] == "not_computable"
    assert result["metrics"]["max_drawdown"]["status"] == "not_computable"
    assert result["metrics"]["relative_return"]["status"] == "not_computable"
    assert result["metrics"]["volatility"]["status"] == "not_computable"
    assert result["metrics"]["beta"]["status"] == "not_computable"
    assert result["metrics"]["cagr"]["status"] == "not_computable"
    assert result["metrics"]["alpha"]["status"] == "not_computable"


def test_align_with_uses_common_period():
    dates = pd.bdate_range("2020-01-01", periods=40)
    equity = pd.Series(np.arange(100, 140), index=dates)
    spy = equity * 2
    other = pd.Series([1, 2, 3], index=dates[10:13])
    result = report(equity, spy, align_with=other)
    assert result["period"]["start"] == dates[10].date().isoformat()
    assert result["period"]["end"] == dates[12].date().isoformat()
    assert result["period"]["trading_days"] == 2


def test_output_top_level_fields_match_performance_report_contract():
    dates = pd.bdate_range("2024-01-01", periods=3)
    result = report(pd.Series([100, 101, 99], index=dates))
    assert set(result) == {
        "subject", "benchmark", "status", "not_computable_reason", "range", "available_ranges", "period", "as_of",
        "stale", "aligned_with", "sample", "flows", "insufficient", "no_runs", "headline", "metrics",
        "benchmark_metrics", "series", "rolling", "monthly_returns", "annual_returns", "distributions",
        "worst_drawdowns", "cash_flows", "conventions",
    }
    assert set(result["metrics"]) == set(result["benchmark_metrics"])
