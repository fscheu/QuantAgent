"""Motor del filtro sobre vectorbt (QuantAgent-d4cq.5). Llena siempre en el open de la vela siguiente a la señal."""

import numpy as np
import pandas as pd
import vectorbt as vbt

PERIODS_PER_YEAR = 252  # diario, igual que BacktestEngine._get_periods_per_year
RISK_FREE_RATE = 0.02  # default de BacktestEngine._calculate_sharpe_ratio


def run_filter(signals: pd.DataFrame, candles: pd.DataFrame, *, commission_pct: float, slippage_pct: float, capital: float) -> dict:
    """Señal en la vela i -> llena al open de la i+1 (la de la última no abre nada). `exit` cierra larga o corta.
    Devuelve trades, pnl, sharpe (252, tasa libre 2%) y max_drawdown_pct (0-100)."""
    entry = signals["entry"].shift(1, fill_value=0)
    exit_ = signals["exit"].astype(bool).shift(1, fill_value=False)
    pf = vbt.Portfolio.from_signals(
        close=candles["close"], price=candles["open"],
        entries=entry == 1, exits=exit_, short_entries=entry == -1, short_exits=exit_,
        direction="both", fees=commission_pct, slippage=slippage_pct,
        init_cash=capital, freq="1D",
    )
    equity = pf.value()
    returns = equity.pct_change().dropna()
    std = returns.std()
    sharpe = 0.0
    if len(returns) > 1 and std > 0:
        sharpe = float((returns.mean() - RISK_FREE_RATE / PERIODS_PER_YEAR) / std * np.sqrt(PERIODS_PER_YEAR))
    return {
        "trades": int(pf.trades.count()),
        "pnl": float(equity.iloc[-1] - capital),
        "sharpe": sharpe,
        "max_drawdown_pct": float(((equity.cummax() - equity) / equity.cummax()).max() * 100),
    }
