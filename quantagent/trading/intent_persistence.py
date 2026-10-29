"""Persistence helper for trade intents."""

from sqlalchemy.orm import Session

from quantagent.models import TradeIntentRecord
from quantagent.strategy.trade_intent import TradeIntent


def save_trade_intent(
    db: Session,
    intent: TradeIntent,
    status: str,
    reject_reason: str | None = None,
    order_id: int | None = None,
) -> TradeIntentRecord:
    record = TradeIntentRecord(
        bucket_id=intent.bucket_id,
        strategy_id=intent.strategy_id,
        symbol=intent.symbol,
        intent=intent.dict(),
        status=status,
        reject_reason=reject_reason,
        order_id=order_id,
        reasoning_metadata=intent.reasoning_metadata,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
