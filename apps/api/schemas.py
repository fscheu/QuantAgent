from typing import Literal

from pydantic import BaseModel


class AccountRef(BaseModel):
    id: str
    name: str
    env: Literal["paper", "live"]


class AccountSummaryNoData(AccountRef):
    status: Literal["no_data"]
    no_data_reason: Literal["not_opened", "not_funded", "unknown"] = "unknown"
    checked_at: str
    next_check_at: None = None


class AccountSummaryActive(AccountRef):
    status: Literal["active"]
    as_of: str
    stale: bool
    equity: float | None
    long_value: float
    short_value: float
    cash: float | None
    day_change: float | None
    day_change_pct: float | None
    week_change: float | None
    week_change_pct: float | None
    positions_count: int
    orders_today: int
    orders_rejected_today: int
    alarms: list = []
    spark_30d: None = None


class PositionResponse(BaseModel):
    symbol: str
    description: None = None
    side: Literal["long", "short"]
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_pl_pct: float
    stop_price: float | None
    stop_distance: float | None
    opened_at: str
