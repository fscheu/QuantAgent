"""Resolve and validate strategy stop prices for new positions."""

from typing import Tuple

UNREACHABLE_STOP_LOSS = 0.01


def resolve_stop_loss(
    stop_loss: float | None, side: str, entry_price: float
) -> Tuple[float, bool]:
    """Return (stop price, uses_default) with a 2% default for missing stops."""
    uses_default = (
        stop_loss is None or stop_loss == 0 or stop_loss == UNREACHABLE_STOP_LOSS
    )
    resolved = stop_loss
    if uses_default:
        resolved = entry_price * (0.98 if side.upper() == "LONG" else 1.02)
    if (side.upper() == "LONG" and resolved >= entry_price) or (
        side.upper() == "SHORT" and resolved <= entry_price
    ):
        raise ValueError(f"{side.upper()} stop_loss must be beyond entry price")
    return float(resolved), uses_default
