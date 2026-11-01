from typing import Literal

from pydantic import BaseModel, Field


class Alarm(BaseModel):
    id: str
    account_id: str
    severity: Literal["critical", "warning", "info", "neutral"]
    kind: str
    title: str
    message: str
    since: str


class HomeStatus(BaseModel):
    level: Literal["ok", "critical", "warning", "info", "neutral"]
    headline: str
    message: str
    link: str | None


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
    alarms: list[Alarm] = Field(default_factory=list)
    spark_30d: list[float] | None = None


class EquityPoint(BaseModel):
    date: str
    account: float
    benchmark: None = None
    shadow: None = None


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


class OrderResponse(BaseModel):
    id: int
    symbol: str
    side: Literal["buy", "sell"]
    type: str
    status: Literal["pending", "partially_filled", "filled", "rejected", "cancelled", "expired"]
    qty: float
    filled_qty: float
    price: float | None
    stop_price: float | None
    avg_fill_price: float | None
    created_at: str
    filled_at: str | None
    reject_reason: str | None
    reject_detail: None = None
    reject_reason_raw: None = None
