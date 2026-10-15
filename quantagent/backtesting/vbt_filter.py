"""Motor del filtro sobre vectorbt (QuantAgent-d4cq.5). Llena siempre en el open de la vela siguiente a la señal."""

import numpy as np
import pandas as pd
import vectorbt as vbt

STOP_COLUMNS = ("stop_loss_pct", "take_profit_pct")  # fracciones del precio llenado; NaN = sin stop
RISK_FREE_RATE = 0.02  # default de BacktestEngine._calculate_sharpe_ratio


def equity_metrics(equity: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> dict:
    """Sharpe y drawdown como el motor: periodos/año = (n-1)/años entre primera y última fecha del índice."""
    returns = equity.pct_change().dropna()
    drawdown = float(((equity.cummax() - equity) / equity.cummax()).max())
    years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 86400.0)
    if len(returns) < 2 or returns.std() == 0 or years <= 0:
        return {"sharpe": 0.0, "max_drawdown": drawdown}
    ppy = (len(equity) - 1) / years
    sharpe = (returns.mean() - risk_free_rate / ppy) / returns.std() * np.sqrt(ppy)
    return {"sharpe": float(sharpe), "max_drawdown": drawdown}


def run_filter(signals: pd.DataFrame, candles: pd.DataFrame, *, commission_pct: float, slippage_pct: float, capital: float) -> dict:
    """Señal en la vela i -> llena al open de la i+1 (la de la última no abre nada). `exit` cierra larga o corta.
    Stops por vela (`stop_loss_pct`, `take_profit_pct`, opcionales; el trailing va en d4cq.8) de la señal, medidos
    desde el precio de entrada llenado y evaluados con high/low; si una vela toca stop y take profit sale por
    el stop; un gap más allá del stop sale al open. Devuelve trades, pnl, sharpe (tasa 2%), sharpe_tasa_0 y max_drawdown (fracción 0-1)."""
    entry = signals["entry"].shift(1, fill_value=0)
    exit_ = signals["exit"].astype(bool).shift(1, fill_value=False)
    stops = {k: signals[k].shift(1) if k in signals else pd.Series(np.nan, index=signals.index) for k in STOP_COLUMNS}
    pf = vbt.Portfolio.from_signals(
        close=candles["close"], price=candles["open"],
        entries=entry == 1, exits=exit_, short_entries=entry == -1, short_exits=exit_,
        open=candles["open"], high=candles["high"], low=candles["low"],
        sl_stop=stops["stop_loss_pct"], tp_stop=stops["take_profit_pct"],
        direction="both", fees=commission_pct, slippage=slippage_pct, init_cash=capital, freq="1D",
    )
    equity = pf.value()
    m = equity_metrics(equity)
    return {
        "trades": int(pf.trades.count()),
        "pnl": float(equity.iloc[-1] - capital),
        "sharpe": m["sharpe"],
        "sharpe_tasa_0": equity_metrics(equity, 0.0)["sharpe"],
        "max_drawdown": m["max_drawdown"],
    }
