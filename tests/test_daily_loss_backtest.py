"""QuantAgent-cub (V17): ¿el límite de pérdida diaria actúa en un backtest? Diagnóstico, sin corregir.

Escenario: velas de 1h de dos días de 2018 (pasado respecto del reloj). Capital 100000, posición del 10%
(100 acciones a 100) y límite diario de 0.5% (unos 500 USD). La estrategia entra LONG en toda vela sin
posición, con stop en 92. La segunda vela del día 1 toca 90: sale por stop a 92 y pierde 800 USD, más que
el límite. Tres respuestas:
1. El circuit breaker SÍ actúa por día de la vela: el motor lo reinicia cuando cambia la fecha de la vela.
2. `get_daily_pnl` NO ve esa pérdida: filtra `closed_at >= date.today()` (reloj) y la vela es de 2018.
3. Lo único que le queda al chequeo "pérdida diaria" es la pérdida abierta, y con ella rechaza también
   la orden que cierra la posición: el trade queda cerrado en la base y las acciones siguen en cartera.
"""

from datetime import datetime
from decimal import Decimal

import pytest

from quantagent.backtesting.backtest import Backtest, CloseRejectedError
from quantagent.models import MarketData, OrderSide, Trade
from quantagent.strategy.base import TradingSignal, TradingStrategy
from quantagent.trading.paper_broker import PaperBroker

DAY1 = [datetime(2018, 3, 5, h) for h in range(10, 16)]
DAY2 = [datetime(2018, 3, 6, h) for h in range(10, 16)]
CRASH = DAY1[1]


class AlwaysLong(TradingStrategy):
    """Pide LONG en cada vela sin posición y anota lo que el risk manager real dice que se perdió hoy."""

    def __init__(self):
        self.bt = None
        self.daily_pnl_seen = {}

    @property
    def required_history_bars(self) -> int:
        return 1

    def generate_signal(self, kline_data, symbol, timeframe, current_price, **kwargs):
        risk = self.bt.order_manager.risk_manager
        self.daily_pnl_seen[kline_data[-1]["timestamp"]] = risk.get_daily_pnl()
        return TradingSignal(decision="LONG", confidence=1.0, entry_price=current_price,
                             stop_loss=92.0, take_profit=300.0, reasoning="siempre long")

    def should_reevaluate(self, position, current_price):
        return False


def _run(db_session, crash_close: int = 100):
    for ts in DAY1 + DAY2:
        crash = ts == CRASH
        db_session.add(MarketData(symbol="SPY", timeframe="1h", timestamp=ts, open=Decimal(100),
                                  high=Decimal(101), low=Decimal(90 if crash else 99),
                                  close=Decimal(crash_close if crash else 100), volume=Decimal(1000)))
    db_session.commit()
    strategy = AlwaysLong()
    bt = Backtest(start_date=DAY1[0], end_date=DAY2[-1], assets=["SPY"], timeframe="1h",
                  db_session=db_session, strategy=strategy, intrabar_stops=True,
                  config={"market_hours_filter": False, "offline_data": True, "slippage_pct": 0.0,
                          "base_position_pct": 0.10, "max_position_pct": 0.20, "max_daily_loss_pct": 0.005})
    strategy.bt = bt
    bt.run(name="limite-diario")
    return bt, strategy, db_session.query(Trade).order_by(Trade.id).all()


def test_circuit_breaker_stops_trading_for_the_rest_of_the_candle_day(db_session):
    # Valida que, tras perder 800 USD (> límite) en la vela CRASH, no se abre ningún trade en el resto
    # del día 1 (la estrategia lo pide en cada vela) y que se vuelve a operar en la primera vela del día 2.
    _, strategy, trades = _run(db_session)
    assert [(t.opened_at, t.closed_at, float(t.pnl)) for t in trades] == [
        (DAY1[0], CRASH, -800.0),
        (DAY2[0], DAY2[-1], 0.0),
    ]
    assert [ts for ts in strategy.daily_pnl_seen if ts.date() == DAY1[0].date()] == DAY1


@pytest.mark.xfail(strict=True, reason="QuantAgent-cub: get_daily_pnl filtra por date.today() (reloj), no "
                   "por el día de la vela: en backtest la pérdida realizada del día siempre da 0")
def test_get_daily_pnl_counts_the_loss_realized_on_the_candle_day(db_session):
    # Comportamiento correcto: en la vela del stop, la pérdida del día según el risk manager es -800.
    _, strategy, _ = _run(db_session)
    assert strategy.daily_pnl_seen[CRASH] == pytest.approx(-800.0)


def test_stop_exit_is_not_rejected_by_the_daily_loss_limit(db_session):
    # Igual que el caso base, pero la vela CRASH cierra en 91: la pérdida abierta (-900) supera el límite
    # antes del stop. Comportamiento correcto: el stop se ejecuta a 92 y la cartera queda sin acciones.
    bt, _, trades = _run(db_session, crash_close=91)
    assert float(trades[0].pnl) == -800.0
    assert bt.portfolio.positions["SPY"]["qty"] == 0.0
    assert bt.portfolio.cash == pytest.approx(100000.0 - 800.0)


def test_close_rejected_by_the_broker_aborts_the_run_naming_the_trade(db_session, monkeypatch):
    # QuantAgent-oat: si el cierre no se ejecuta por otro motivo (acá el broker rechaza toda venta), bt.run
    # corta con un error que nombra el trade y las acciones que quedan, y el trade no figura como cerrado.
    real_place_order = PaperBroker.place_order

    def reject_sells(self, order):
        if order.side == OrderSide.SELL:
            raise RuntimeError("venta rechazada por el broker")
        return real_place_order(self, order)

    monkeypatch.setattr(PaperBroker, "place_order", reject_sells)
    with pytest.raises(CloseRejectedError) as exc:
        _run(db_session)
    trade = db_session.query(Trade).one()
    assert f"Trade {trade.id} (SPY)" in str(exc.value) and "conserva 100.0 acciones" in str(exc.value)
    assert (trade.opened_at, trade.closed_at, trade.pnl) == (DAY1[0], None, None)
