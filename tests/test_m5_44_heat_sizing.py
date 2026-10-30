import pytest

from quantagent.models import OrderSide
from quantagent.trading.position_sizer import PositionSizer
from quantagent.trading.risk_manager import RiskManager


class Portfolio:
    def __init__(self, positions, equity=10_000):
        self.positions = positions
        self.cash = equity
        self.equity = equity

    def get_total_value(self):
        return self.equity


@pytest.mark.parametrize(
    ("positions", "qty", "price", "stop", "accepted"),
    [
        ({}, 10_001, 0.5, 0.3, False),
        ({}, 10_000, 0.5, 0.3, True),  # exactly at the limit is allowed
        ({"OLD": {"qty": 10, "avg_cost": 0.5, "stop_price": 0.4}}, 9_990, 0.5, 0.3, True),
    ],
)
def test_heat_gate_uses_open_risk_and_allows_boundary(positions, qty, price, stop, accepted):
    risk = RiskManager(Portfolio(positions, equity=100_000), max_heat_pct=0.02)
    valid, reason = risk.validate_trade("NEW", OrderSide.BUY, qty, price, stop_price=stop)
    assert valid is accepted
    if not accepted:
        assert "max_heat" in reason


def test_reducing_position_is_not_rejected_by_heat():
    risk = RiskManager(Portfolio({"OLD": {"qty": 10, "avg_cost": 100, "stop_price": 90}}), max_heat_pct=0.001)
    assert risk.validate_trade("OLD", OrderSide.SELL, 1, 100)[0]


@pytest.mark.parametrize(
    ("capital", "expected"),
    [(50, 0.0), (350, 3.0)],
)
def test_non_fractionable_sizing_floors_to_whole_shares(capital, expected):
    sizer = PositionSizer(base_position_pct=0.1, regla="fijo")
    qty = sizer.calculate_size("XYZ", 1, 10, capital, fraccionable=False)
    assert qty == expected
