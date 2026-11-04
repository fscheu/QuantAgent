from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from apps.api.market_regime import regimen_spy
from quantagent.fabrica.regimen import clasificar_regimen
from quantagent.models import MarketBar


def _seed(db_session, closes):
    dates = pd.bdate_range("2023-01-02", periods=len(closes))
    db_session.add_all([
        MarketBar(symbol="SPY", session_date=day.date(), open=Decimal(str(price)),
                  high=Decimal(str(price)), low=Decimal(str(price)), close=Decimal(str(price)),
                  volume=Decimal("1000"), source="test")
        for day, price in zip(dates, closes)
    ])
    db_session.flush()
    return dates


def _closes(n, drift, amp_from, amp_to):
    """Retornos con deriva fija y amplitud alternada que cambia linealmente: fija si la vol de hoy queda sobre o bajo su mediana."""
    amp = np.linspace(amp_from, amp_to, n)
    returns = drift + amp * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    return 100 * np.cumprod(1 + returns)


def test_regime_metrics_and_quiet_uptrend_match_independent_calculation(db_session):
    closes = _closes(300, 0.001, 0.01, 0.0005)  # sube y la vol cae: hoy está bajo su mediana
    _seed(db_session, closes)
    result = regimen_spy(db_session)
    series = pd.Series(closes)
    vol = series.pct_change().rolling(20).std()
    assert result["regime"] == "up_calm"
    assert result["ma200"] == pytest.approx(np.mean(closes[-200:]), rel=1e-6)
    assert result["vol20"] == pytest.approx(vol.iloc[-1], rel=1e-5)
    assert result["vol_median_1y"] == pytest.approx(vol.rolling(252).median().iloc[-1], rel=1e-5)
    assert result["vol20"] < result["vol_median_1y"] and result["price"] > result["ma200"]


def test_agitated_downtrend_is_down_vol(db_session):
    closes = _closes(300, -0.002, 0.0005, 0.02)  # baja y la vol sube: hoy está sobre su mediana
    _seed(db_session, closes)
    result = regimen_spy(db_session)
    assert result["regime"] == "down_vol"
    assert result["vol20"] > result["vol_median_1y"] and result["price"] < result["ma200"]


def test_regime_change_sets_since_sessions_and_strip(db_session):
    calm = _closes(275, 0.001, 0.01, 0.0005)
    agitated = calm[-1] * np.cumprod(1 + (-0.01 + 0.03 * np.where(np.arange(25) % 2 == 0, 1.0, -1.0)))
    closes = np.concatenate([calm, agitated])
    dates = _seed(db_session, closes)
    result = regimen_spy(db_session)
    expected = clasificar_regimen(pd.Series(closes, index=dates))
    assert expected.iloc[-1] != expected.iloc[-2] or expected.nunique() > 1  # hubo un cambio real
    start = len(expected) - 1
    while start > 0 and expected.iloc[start - 1] == expected.iloc[-1]:
        start -= 1
    assert result["regime"] == {"alcista_baja_vol": "up_calm", "alcista_alta_vol": "up_vol",
                                "bajista_baja_vol": "down_calm", "bajista_alta_vol": "down_vol"}[expected.iloc[-1]]
    assert result["since"] == dates[start].date().isoformat()
    assert result["sessions"] == len(expected) - start
    assert 0 < start and result["sessions"] < len(expected)
    assert result["strip"][-1]["regime"] == result["regime"] and result["strip"][-1]["to"] == dates[-1].date().isoformat()


def test_missing_or_insufficient_spy_history_returns_none(db_session):
    assert regimen_spy(db_session) is None
    _seed(db_session, np.linspace(100, 110, 271))
    assert regimen_spy(db_session) is None
