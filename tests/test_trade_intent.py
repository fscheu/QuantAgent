import pytest
from pydantic import ValidationError

from quantagent.models import TradeIntentRecord
from quantagent.strategy.trade_intent import TradeIntent
from quantagent.trading.intent_persistence import save_trade_intent


def _intent(**kwargs):
    fields = dict(
        decision="LONG",
        confidence=0.8,
        stop_loss=95,
        bucket_id="m5-paper",
        strategy_id="arquetipo-a-pullback",
        symbol="SPY",
        atr_at_signal=2.5,
        reasoning_metadata={"sector_score": 1.2, "rsi": 28},
    )
    return TradeIntent(**(fields | kwargs))


def test_trade_intent_requires_stop_loss():
    with pytest.raises(ValidationError):
        _intent(stop_loss=None)


def test_risk_rejected_intent_persists_reason_and_metadata(db_session):
    saved = save_trade_intent(
        db_session, _intent(), status="rechazado_riesgo", reject_reason="max_heat"
    )

    loaded = db_session.get(TradeIntentRecord, saved.id)
    assert loaded.status == "rechazado_riesgo"
    assert loaded.reject_reason == "max_heat"
    assert loaded.reasoning_metadata == {"sector_score": 1.2, "rsi": 28}
