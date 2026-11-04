"""SPY daily regime summary for the market page."""

import pandas as pd
from sqlalchemy import select

from quantagent.fabrica.regimen import clasificar_regimen
from quantagent.models import MarketBar


def regimen_spy(session):
    bars = session.scalars(
        select(MarketBar).where(MarketBar.symbol == "SPY").order_by(MarketBar.session_date)
    ).all()
    if not bars:
        return None

    closes = pd.Series(
        [float(bar.close) for bar in bars],
        index=pd.DatetimeIndex([bar.session_date for bar in bars]),
    )
    returns = closes.pct_change()
    vol = returns.rolling(20).std()
    median_vol = vol.rolling(252).median()
    ma = closes.rolling(200).mean()
    labels = clasificar_regimen(closes)
    if pd.isna(ma.iloc[-1]) or pd.isna(median_vol.iloc[-1]) or pd.isna(labels.iloc[-1]):
        return None

    current = labels.iloc[-1]
    since_index = len(labels) - 1
    while since_index > 0 and labels.iloc[since_index - 1] == current:
        since_index -= 1
    cutoff = closes.index[-1] - pd.DateOffset(months=12)
    selected = labels.loc[labels.index >= cutoff]
    segments = []
    keys = selected.fillna("__missing__")
    for _, group in selected.groupby(keys.ne(keys.shift()).cumsum()):
        regime = group.iloc[0]
        segments.append({
            "regime": _regime_id(regime) if pd.notna(regime) else None,
            "from": group.index[0].date().isoformat(),
            "to": group.index[-1].date().isoformat(),
            "sessions": len(group),
        })

    return {
        "symbol": "SPY",
        "regime": _regime_id(current),
        "since": closes.index[since_index].date().isoformat(),
        "sessions": len(labels) - since_index,
        "price": float(closes.iloc[-1]),
        "ma200": float(ma.iloc[-1]),
        "vol20": float(vol.iloc[-1]),
        "vol_median_1y": float(median_vol.iloc[-1]),
        "strip": segments,
    }


def _regime_id(label):
    return {
        "alcista_baja_vol": "up_calm",
        "alcista_alta_vol": "up_vol",
        "bajista_baja_vol": "down_calm",
        "bajista_alta_vol": "down_vol",
    }[label]
