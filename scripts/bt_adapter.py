#!/usr/bin/env python
"""QuantAgent-i6n: adaptador generico que corre cualquier TradingStrategy del proyecto dentro de backtesting.py.

Uso:  python scripts/bt_adapter.py --strategy <alias> (--fixture <f> | --snapshot <s> --symbol <S> --from --to)
          [--fill close|next-open] [--commission-pct C] [--out trades.csv]

La DECISION es de la estrategia; la EJECUCION es de la libreria (sl=/tp= contra maximo y minimo,
commission=, su efectivo y su equity). El adaptador solo traduce:
- ventana -> generate_signal: las ultimas `required_history_bars` velas DEL DATO (no dias de calendario,
  QuantAgent-8to), como lista de dicts timestamp/open/high/low/close/volume, mas simbolo, timeframe y el
  close de la ultima vela, igual que quantagent/backtesting/backtest.py::_analyze_and_trade.
- senal -> orden: tamano = equity * TRADING_BASE_POSITION_PCT * confianza / precio; stop y take profit de
  la senal van como sl=/tp= (si faltan, los defaults del motor: 2 % y 3 %), redondeados a 8 decimales.
- posicion -> ActivePosition sin persistir (mismos campos que crea PositionMonitor.open_position) para
  should_exit(posicion, close, ventana). entry_price y quantity son los del fill de la libreria. Si no
  sale, candles_since_entry += 1 (lo que hace el motor en update_candle_tracking).
- cierre por regla -> cierra la posicion; despues se evalua la senal de esa misma vela, como el motor.

Llenado: `close` = al cierre de la vela de la senal (trade_on_close=True, vela centinela y orden de
mercado opuesta, como scripts/crossval_rsi.py); `next-open` = apertura de la vela siguiente (default de la
libreria, Position.close() y finalize_trades=True; sin centinela). Precios y unidades usan la escala 2**27
del port. --commission-pct es fraccion del nocional por lado (0.001 = 0.1 %), el `commission=` nativo.

exit_reason del CSV: el motivo que devolvio should_exit; STOP_LOSS/TAKE_PROFIT cuando cerro la libreria
(segun de que lado del stop quedo el precio de salida); backtest_end para lo que seguia abierto al final.
Nada bajo quantagent/ importa backtesting (ADR vigente): este archivo vive en scripts/.
"""

from __future__ import annotations

import argparse
import inspect
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import crossval_rsi as port  # noqa: E402  (fija TRADING_SLIPPAGE_PCT=0, agrega la raiz al path, sale si falta backtesting)
import pandas as pd  # noqa: E402
from backtesting import Backtest, Strategy  # noqa: E402

from quantagent import settings  # noqa: E402
from quantagent.backtesting.backtest import Backtest as ProjectBacktest  # noqa: E402
from quantagent.backtesting.export import TradeRow, trades_to_csv  # noqa: E402
from quantagent.backtesting.fixtures import fixture_metadata  # noqa: E402
from quantagent.cli.backtest import STRATEGY_ALIASES  # noqa: E402
from quantagent.data.snapshot import load_manifest  # noqa: E402
from quantagent.models import ActivePosition, OrderSide  # noqa: E402
from quantagent.strategy.registry import build_strategy  # noqa: E402

SCALE = port.SCALE
INITIAL_CASH = inspect.signature(ProjectBacktest.__init__).parameters["initial_capital"].default


class StrategyAdapter(Strategy):
    strategy = None  # TradingStrategy del proyecto
    symbol = "SPY"
    timeframe = "1d"
    n_real_bars = 0  # sin la vela centinela
    fill_close = True

    def init(self):
        df = self.data.df
        cols = {c.lower(): df[c].values * (1.0 if c == "Volume" else SCALE) for c in df.columns}
        self.bars = pd.DataFrame({"timestamp": df.index, **cols})  # precios reales, columnas del motor
        self.pos = None  # ActivePosition de la posicion abierta (o con orden de entrada pendiente)
        self.info = {}  # vela de entrada -> {"stop", "reason"} para el CSV

    def next(self):
        n = len(self.data)
        if n > self.n_real_bars:  # vela centinela: solo procesa las ordenes de la ultima vela real
            return
        if self.pos is not None and not self.position:
            last = self.closed_trades[-1] if self.closed_trades else None
            if last is not None and last.entry_bar == self.key:  # la libreria cerro por sl=/tp=
                stop = self.info[self.key]["stop"] / SCALE
                hit_stop = last.exit_price <= stop if last.is_long else last.exit_price >= stop
                self.info[self.key]["reason"] = "STOP_LOSS" if hit_stop else "TAKE_PROFIT"
            self.pos = None  # (si no hay trade, la libreria cancelo la orden por falta de efectivo)
        history = self.strategy.required_history_bars
        history = history if history > 0 else 30
        if n >= history:
            window = self.bars.iloc[n - history:n]
            price = float(window["close"].iloc[-1])
            if self.pos is not None:
                trade = self.trades[0]
                self.pos.entry_price = trade.entry_price * SCALE
                self.pos.quantity = abs(trade.size) / SCALE
                should_exit, reason = self.strategy.should_exit(self.pos, price, window)
                if should_exit:
                    self._close(reason)
                else:
                    self.pos.candles_since_entry += 1
            if self.pos is None:
                self._maybe_enter(window, price, n)
        if self.fill_close and n == self.n_real_bars and self.pos is not None:
            self._close("backtest_end")  # next-open: lo hace la libreria (finalize_trades=True)

    def _close(self, reason):
        self.info[self.key]["reason"] = reason
        if self.fill_close:
            # La orden opuesta llena a este cierre. Sin cancelar sl/tp, un stop tocado en la vela
            # siguiente se procesaria antes y la orden opuesta abriria una posicion contraria.
            for trade in self.trades:
                trade.sl = trade.tp = None
            (self.sell if self.units > 0 else self.buy)(size=abs(self.units))
        else:
            self.position.close()
        self.pos = None

    def _maybe_enter(self, window, price, n):
        signal = self.strategy.generate_signal(window.to_dict(orient="records"), self.symbol, self.timeframe, price)
        if signal is None or signal.decision == "HOLD":
            return
        long = signal.decision == "LONG"
        qty = self.equity * settings.TRADING_BASE_POSITION_PCT * signal.confidence / price
        units = int(round(qty * SCALE))
        if units < 1:
            return
        stop = round(signal.stop_loss or price * (0.98 if long else 1.02), 8)
        take = round(signal.take_profit or price * (1.03 if long else 0.97), 8)
        (self.buy if long else self.sell)(size=units, sl=stop / SCALE, tp=take / SCALE)
        self.units = units if long else -units
        self.key = n - 1 if self.fill_close else n  # indice de la vela en la que la libreria llena la orden
        self.info[self.key] = {"stop": stop, "reason": "backtest_end"}
        self.pos = ActivePosition(
            symbol=self.symbol, side=OrderSide.BUY if long else OrderSide.SELL,
            entry_price=signal.entry_price or price, stop_loss=stop, take_profit=take, quantity=units / SCALE,
            decision_timestamp=window["timestamp"].iloc[-1].to_pydatetime(), candles_since_entry=0,
            exit_policy=signal.exit_policy, max_hold_candles=signal.max_hold_candles,
            trailing_stop_pct=signal.trailing_stop_pct, prediction_horizon=3, candles_direction=[],
        )


def run_adapter(strategy, df: pd.DataFrame, *, symbol: str, timeframe: str, fill: str = "close",
                commission_pct: float = 0.0, cash: float = INITIAL_CASH):
    """Corre `strategy` sobre `df` (el de crossval_rsi.load_fixture: escalado y con vela centinela)."""
    close = fill == "close"
    bt = Backtest(df if close else df.iloc[:-1], StrategyAdapter, cash=cash, commission=commission_pct, margin=1.0,
                  trade_on_close=close, hedging=False, exclusive_orders=False, finalize_trades=not close)
    return bt.run(strategy=strategy, symbol=symbol, timeframe=timeframe, n_real_bars=len(df) - 1, fill_close=close)


def trade_rows(stats, symbol: str) -> list[TradeRow]:
    info = stats["_strategy"].info
    return [
        TradeRow(t.EntryTime, t.ExitTime, symbol, "buy" if t.Size > 0 else "sell", abs(t.Size) / SCALE,
                 t.EntryPrice * SCALE, t.ExitPrice * SCALE, info[t.EntryBar]["stop"], t.PnL, info[t.EntryBar]["reason"])
        for t in stats["_trades"].sort_values("EntryTime").itertuples()
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strategy", required=True, choices=sorted(STRATEGY_ALIASES))
    ap.add_argument("--fixture", help="fixture bajo tests/fixtures/ (sin .csv)")
    ap.add_argument("--snapshot", help="Snapshot en $QUANTAGENT_SNAPSHOT_DIR (requiere --symbol)")
    ap.add_argument("--symbol", help="Simbolo del snapshot")
    ap.add_argument("--from", dest="from_date", help="Primera sesion, YYYY-MM-DD")
    ap.add_argument("--to", dest="to_date", help="Ultima sesion, YYYY-MM-DD")
    ap.add_argument("--fill", choices=["close", "next-open"], default="close")
    ap.add_argument("--commission-pct", type=float, default=0.0, help="fraccion del nocional por lado (0.001 = 0.1 %%)")
    ap.add_argument("--out", help="CSV de trades")
    args = ap.parse_args(argv)
    if bool(args.fixture) == bool(args.snapshot) or (args.snapshot and not args.symbol):
        ap.error("usar --fixture o bien --snapshot con --symbol")

    if args.snapshot:
        symbol, timeframe = args.symbol, load_manifest(args.snapshot)["timeframe"]
        df = port.load_fixture(snapshot=args.snapshot, symbol=symbol, from_date=args.from_date, to_date=args.to_date)
    else:
        meta = fixture_metadata(args.fixture)
        symbol, timeframe = meta.symbol, meta.timeframe
        df = port.load_fixture(port.FIXTURES_DIR / f"{args.fixture}.csv")
    stats = run_adapter(build_strategy(STRATEGY_ALIASES[args.strategy]), df, symbol=symbol, timeframe=timeframe,
                        fill=args.fill, commission_pct=args.commission_pct)
    rows = trade_rows(stats, symbol)
    if args.out:
        Path(args.out).write_text(trades_to_csv(rows))

    pf = stats["Profit Factor"]
    print(f"Trades: {int(stats['# Trades'])}")
    print(f"Win rate: {stats['Win Rate [%]'] / 100:.2%}")
    print("Profit factor: n/a" if math.isnan(pf) or math.isinf(pf) else f"Profit factor: {pf:.2f}")
    print(f"Sharpe ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"Total PnL: {sum(r.pnl for r in rows):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
