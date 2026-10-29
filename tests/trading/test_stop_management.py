import random
from decimal import Decimal

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quantagent.models import ActivePosition, Environment, OrderSide, Trade
from quantagent.trading.position_monitor import PositionMonitor
from quantagent.trading.stop_management import (
    nuevo_stop,
    propuesta_breakeven,
    propuesta_chandelier,
    propuesta_invalidacion,
)


def test_stops_never_move_against_position_for_random_sequences():
    rng = random.Random(43)
    for lado in (OrderSide.BUY, OrderSide.SELL):
        for _ in range(200):
            actual = 100
            acumuladas = []
            for _ in range(50):
                propuesto = rng.choice([actual - 5, actual, actual + 5, rng.randint(80, 120)])
                decision = nuevo_stop(actual, propuesto, lado)
                assert decision.stop >= actual if lado == OrderSide.BUY else decision.stop <= actual
                actual = decision.stop
                acumuladas.append(propuesto)
            esperado = max([100, *acumuladas]) if lado == OrderSide.BUY else min([100, *acumuladas])
            assert actual == esperado


def test_breakeven_requires_threshold_of_one_and_a_half_r():
    assert propuesta_breakeven(100, 98, 102.999, OrderSide.BUY) is None
    assert propuesta_breakeven(100, 98, 103, OrderSide.BUY) == 100
    assert propuesta_breakeven(100, 102, 97, OrderSide.SELL) == 100


def test_chandelier_and_thesis_invalidation_proposals():
    assert propuesta_chandelier(120, 2, OrderSide.BUY) == 114
    assert propuesta_chandelier(80, 2, OrderSide.SELL) == 86
    assert propuesta_invalidacion([95, 92], OrderSide.BUY) == 92
    assert propuesta_invalidacion([105, 108], OrderSide.SELL) == 108


def test_weaker_stop_proposal_is_marked_rejected():
    decision = nuevo_stop(98, 95, OrderSide.BUY)
    assert decision.stop == 98
    assert decision.rechazado is True


def test_position_monitor_persists_tighter_stop_and_rejects_looser_one():
    engine = create_engine("sqlite:///:memory:")
    Trade.__table__.create(bind=engine)
    ActivePosition.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with Session() as session:
        monitor = PositionMonitor(session, environment=Environment.PAPER)
        position = monitor.open_position(
            symbol="BTC", side=OrderSide.BUY, entry_price=100, stop_loss=98,
            take_profit=110, quantity=Decimal("1"), exit_policy="sl_tp_only",
        )
        sube = monitor.actualizar_stop(position, 99)
        assert sube.stop == Decimal("99")
        session.refresh(position)
        assert position.stop_loss == Decimal("99")
        baja = monitor.actualizar_stop(position, 97)
        assert baja.rechazado is True
        session.refresh(position)
        assert position.stop_loss == Decimal("99")
