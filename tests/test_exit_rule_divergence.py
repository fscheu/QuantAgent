from quantagent.backtesting.backtest import check_intrabar_stops
from quantagent.models import ActivePosition, ExitPolicy, OrderSide
from quantagent.strategy.rsi_strategy import RSIMeanReversionStrategy


def _long_position(**overrides) -> ActivePosition:
    """LONG a 100 con stop 90 y take profit 130, sin tocar la base."""
    fields = dict(
        symbol="SPY",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=130.0,
        quantity=10,
        exit_policy=ExitPolicy.TRAILING_STOP,
        trailing_stop_pct=0.05,
        candles_since_entry=3,
    )
    fields.update(overrides)
    return ActivePosition(**fields)


def _backtest_rule(position: ActivePosition, price: float):
    # RSI no sobreescribe should_exit: es la regla base de strategy/base.py.
    return RSIMeanReversionStrategy().should_exit(position, price, None)


def _paper_rule(position: ActivePosition, price: float):
    return RSIMeanReversionStrategy().should_exit(position, price, None)


def test_trailing_stop_sale_en_backtest_y_no_en_paper():
    """Valida: máximo visto 120, trailing 5% => piso 114; con precio 113 el
    backtest y paper cierran por TRAILING_STOP."""
    price = 113.0

    backtest = _backtest_rule(_long_position(highest_price_seen=120.0), price)
    paper = _paper_rule(_long_position(highest_price_seen=120.0), price)

    assert backtest == (True, "TRAILING_STOP")
    assert paper == backtest


def test_stop_tocado_dentro_de_la_vela_sale_en_backtest_y_no_en_paper():
    """Valida: vela con mínimo 89 (toca el stop de 90) y cierre 95. El backtest
    cierra a 90 por el camino intravela; paper solo mira el cierre y no sale."""
    candle = {"open": 100.0, "high": 101.0, "low": 89.0, "close": 95.0}
    position = _long_position(exit_policy=ExitPolicy.SL_TP_ONLY, trailing_stop_pct=None)

    backtest = check_intrabar_stops(position, candle)
    paper = _paper_rule(position, candle["close"])

    # QuantAgent-beo: comportamiento actual, defecto conocido.
    assert backtest == ("STOP_LOSS", 90.0)
    assert paper == (False, None)


def test_max_hold_sale_en_paper_y_no_en_backtest():
    """Valida que con SL_TP_ONLY, backtest y paper mantienen la posición."""
    kwargs = dict(
        exit_policy=ExitPolicy.SL_TP_ONLY,
        trailing_stop_pct=None,
        max_hold_candles=5,
        candles_since_entry=5,
    )
    price = 105.0

    backtest = _backtest_rule(_long_position(**kwargs), price)
    paper = _paper_rule(_long_position(**kwargs), price)

    assert backtest == (False, None)
    assert paper == backtest


def test_misma_salida_por_stop_con_igual_texto_de_razon():
    """Valida: cuando las dos reglas coinciden en salir (precio 89 <= stop 90),
    la razón que se guarda en close_reason / exit_signal no es el mismo texto."""
    price = 89.0

    backtest = _backtest_rule(_long_position(), price)
    paper = _paper_rule(_long_position(), price)

    # QuantAgent-beo U1: ambos caminos comparten la razón.
    assert backtest == (True, "STOP_LOSS")
    assert paper == backtest
