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
    cash: float | None = None
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
    benchmark: float | None = None
    shadow: float | None = None


class DividendResponse(BaseModel):
    date: str
    symbols: list[str]
    net: float
    withheld: float


class SectorExposure(BaseModel):
    sector: str
    long: float
    short: float
    pct: float


class PositionResponse(BaseModel):
    symbol: str
    description: None = None
    side: Literal["long", "short"]
    status: Literal["executed", "pending"]
    qty: float
    avg_entry_price: float | None
    current_price: float | None
    market_value: float | None
    unrealized_pl: float | None
    unrealized_pl_pct: float | None
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


class BacktestRef(BaseModel):
    id: int
    label: str
    hypothesis_name: str | None
    start: str
    end: str
    total_return: float


class MarketPage(BaseModel):
    as_of: dict | None
    clock: dict | None
    regime: dict | None
    motor: dict | None
    sectors: list[dict]
    macro: list[dict]
    recessions: list[dict]
    etfs: list[dict]
    events: list[dict]
    errors: dict[str, str] = Field(default_factory=dict)


class MarketSeriesPoint(BaseModel):
    date: str
    value: float


class MarketSeries(BaseModel):
    symbol: str
    points: list[MarketSeriesPoint]


class DailyCandle(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float


class FunnelStage(BaseModel):
    stage: str
    reached: int
    here: int
    passed: int
    cut: int
    error: int
    alarm: int


class FunnelLater(BaseModel):
    phase: str
    reached: int
    here: int


class FunnelDropReason(BaseModel):
    stage: str
    verdict: Literal["CORTA", "ERROR", "ALARMA"]
    cause_code: str | None
    cause_label: str | None
    count: int
    example: dict | None


class Funnel(BaseModel):
    cycle: str
    period: Literal["30d", "90d", "all"]
    total: int
    stages: list[FunnelStage]
    later: list[FunnelLater]
    promoted: int
    drop_reasons: list[FunnelDropReason]
    sample_too_small: bool


class CycleMilestone(BaseModel):
    key: str
    day: int | None
    date: str
    done: bool


class CycleSummary(BaseModel):
    cycle: str
    status: Literal["open", "closed"]
    started_at: str
    closed_at: str | None
    entered: int
    active: int
    cut: int
    held: int
    promoted: int
    reached_by_stage: dict[str, int | None]
    milestones: list[CycleMilestone]
    headline: str | None
    receipt_id: str | None


class ReceiptStage(BaseModel):
    stage: Literal["e0", "e1", "e2", "e3"]
    reached: int
    here: int
    passed: int
    cut: int
    error: int
    alarm: int
    pending: bool


class ReceiptSnapshot(BaseModel):
    name: str
    as_of: str
    hash: str


class CycleReceipt(BaseModel):
    id: str | None
    hash: str | None
    cycle: str
    partial: bool
    generated_at: str
    counts: dict[str, int]
    stages: list[ReceiptStage]
    drops: list[dict]
    attempts: list[dict]
    snapshots: list[ReceiptSnapshot]


class ReviewCycle(BaseModel):
    cycle: str
    promoted: int


class ReviewRule(BaseModel):
    limit: int
    streak: int
    reached: bool
    last_cycles: list[ReviewCycle]


class HistoryStage(BaseModel):
    stage: Literal["e0", "e1", "e2", "e3"]
    total: int
    pass_: int = Field(alias="pass")
    cut: int
    error: int
    alarm: int

    class Config:
        populate_by_name = True


class HistoryAttemptLabel(BaseModel):
    label: str | None
    count: int


class HistorySummary(BaseModel):
    stage_events: int
    attempts: int
    events_by_stage: list[HistoryStage]
    attempts_by_label: list[HistoryAttemptLabel]


class Hypothesis(BaseModel):
    id: str
    name: str
    idea: str | None
    cycle: str | None
    stage: str
    verdict: Literal["PASA", "CORTA", "ERROR", "ALARMA"] | None
    status: Literal["active", "promoted", "discarded", "held"]
    promoted: bool
    used_reserve: bool
    attempts_valid: int
    attempts_invalid: int
    last_metric: dict | None
    discard_reason: str | None
    discard_measure: dict | None
    updated_at: str


class HypothesisStageResult(BaseModel):
    stage: Literal["e0", "e1", "e2", "e3"]
    verdict: Literal["PASA", "CORTA", "ERROR", "ALARMA"]
    decided_at: str
    reason: str
    measure: dict | None
    imported: bool


class HypothesisAttempt(BaseModel):
    id: str
    hypothesis_id: str
    stage: str
    ran_at: str
    subject: str
    snapshot_hash: str
    sharpe: float | None
    trades: int | None
    pnl_pct: float | None
    invalid: bool
    invalid_reason: str | None


class HypothesisDetail(Hypothesis):
    timeline: list[HypothesisStageResult]
    later_phases: list[dict]
    attempts: list[HypothesisAttempt]
    backtests: list[dict]
    paper_account: dict | None
