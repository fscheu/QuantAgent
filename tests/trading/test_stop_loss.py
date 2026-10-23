import pytest

from quantagent.stop_loss import resolve_stop_loss


@pytest.mark.parametrize("missing", [None, 0, 0.01])
@pytest.mark.parametrize(("side", "expected"), [("LONG", 98.0), ("SHORT", 102.0)])
def test_missing_stop_resolves_to_two_percent_and_marks_default(missing, side, expected):
    assert resolve_stop_loss(missing, side, 100.0) == (expected, True)


@pytest.mark.parametrize(("stop", "side"), [(98.0, "LONG"), (105.0, "SHORT")])
def test_real_strategy_stop_is_preserved(stop, side):
    assert resolve_stop_loss(stop, side, 100.0) == (stop, False)


@pytest.mark.parametrize(("stop", "side"), [(100.0, "LONG"), (101.0, "LONG"), (100.0, "SHORT"), (99.0, "SHORT")])
def test_stop_on_wrong_side_raises(stop, side):
    with pytest.raises(ValueError, match="stop_loss must be beyond entry"):
        resolve_stop_loss(stop, side, 100.0)
