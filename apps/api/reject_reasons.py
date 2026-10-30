"""English display text for persisted order rejection codes."""

REJECT_REASONS = {
    "insufficient_buying_power": "Insufficient buying power",
    "risk_limit": "Risk limit exceeded",
    "market_closed": "Market is closed",
    "symbol_not_tradable": "Symbol is not tradable",
    "duplicate_order": "Duplicate order",
    "position_limit": "Position limit exceeded",
}


def rejection_text(code: str) -> str:
    return REJECT_REASONS.get(code, "Rejected (reason not recognised)")
