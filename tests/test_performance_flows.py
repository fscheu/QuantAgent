import pandas as pd
import pytest

from quantagent.performance import build_report


def test_deposit_is_excluded_and_subperiod_returns_are_linked():
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    report = build_report(pd.Series([100, 200, 150, 165], index=dates),
        pd.Series([100, 100, 100, 100], index=dates), subject={"type": "portfolio"}, range="all",
        flows=[{"date": dates[1], "amount": 100, "reconciled": True},
               {"date": dates[2], "amount": -50, "reconciled": True}],
        thresholds={"descriptive_days": 1})
    assert report["metrics"]["total_return"]["value"] == pytest.approx(10)


def test_unreconciled_flow_blocks_portfolio_returns():
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    report = build_report(pd.Series([100, 200, 220], index=dates),
        pd.Series([100, 100, 100], index=dates), subject={"type": "portfolio"}, range="all",
        flows=[{"date": dates[1], "amount": 100, "reconciled": False}],
        thresholds={"descriptive_days": 1})
    assert report["status"] == "returns_not_computable"
    assert report["not_computable_reason"] == "unregistered_flows"
    assert report["flows"]["unreconciled"] == [{"date": str(dates[1]), "amount": 100.0}]
