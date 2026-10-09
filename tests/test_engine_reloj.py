"""QuantAgent-48n: con offline_data el engine avanza por las velas cargadas, no por la grilla de calendario.

Escenario: velas de 1h del viernes 2026-01-02 y del lunes 2026-01-05, sin sabado ni domingo. La estrategia
entra LONG en la ultima vela del viernes (close 100, SL 95), cuyo minimo es 90. Esa vela es la de entrada:
los stops intravela no se evaluan sobre ella. Con la grilla, el sabado 00:00 la ventana termina en esa misma
vela y el engine la vuelve a evaluar como si fuera nueva: sale por STOP_LOSS a 95 el sabado, sin vela, y
vuelve a entrar al cierre del viernes (con el motor anterior: 24 trades, 23 salidas en sabado).
Con el reloj del dato la siguiente vela es la del lunes (minimo 99) y la posicion llega al cierre final.
"""

from datetime import datetime
from decimal import Decimal

from quantagent.backtesting.backtest import Backtest
from quantagent.models import MarketData, Trade
from quantagent.strategy.base import TradingSignal, TradingStrategy

FRIDAY_LAST = datetime(2026, 1, 2, 23)
STAMPS = [datetime(2026, 1, 2, h) for h in range(24)] + [datetime(2026, 1, 5, h) for h in range(6)]


class EnterOnFridayLastCandle(TradingStrategy):
    @property
    def required_history_bars(self) -> int:
        return 2

    def generate_signal(self, kline_data, symbol, timeframe, current_price, **kwargs):
        if kline_data[-1]["timestamp"] != FRIDAY_LAST:
            return None
        return TradingSignal(decision="LONG", confidence=1.0, entry_price=current_price,
                             stop_loss=95.0, take_profit=200.0, reasoning="ultima vela del viernes")

    def should_reevaluate(self, position, current_price):
        return False


def _run(db_session) -> Backtest:
    for ts in STAMPS:
        low = 90 if ts == FRIDAY_LAST else 99
        db_session.add(MarketData(symbol="SPY", timeframe="1h", timestamp=ts, open=Decimal(100),
                                  high=Decimal(101), low=Decimal(low), close=Decimal(100), volume=Decimal(1000)))
    db_session.commit()
    bt = Backtest(start_date=STAMPS[0], end_date=STAMPS[-1], assets=["SPY"], timeframe="1h",
                  db_session=db_session, strategy=EnterOnFridayLastCandle(), intrabar_stops=True,
                  config={"market_hours_filter": False, "offline_data": True, "slippage_pct": 0.0,
                          "max_daily_loss_pct": 1.0, "max_position_pct": 1.0})
    bt.run(name="reloj-del-dato")
    return bt


def test_weekend_gap_does_not_reevaluate_the_entry_candle(db_session):
    # Valida que el sabado no se vuelve a evaluar la vela de entrada del viernes: la posicion no sale por
    # el minimo de esa vela (95 el sabado) sino al cierre final del lunes, a 100.
    _run(db_session)
    trade = db_session.query(Trade).one()
    assert trade.opened_at == FRIDAY_LAST
    assert trade.closed_at == STAMPS[-1]
    assert float(trade.exit_price) == 100.0


def test_clock_is_the_loaded_candles(db_session):
    # Valida el reloj: la curva de equity tiene un punto por vela cargada y ninguno en el hueco.
    bt = _run(db_session)
    assert [e["date"] for e in bt.equity_curve] == STAMPS
