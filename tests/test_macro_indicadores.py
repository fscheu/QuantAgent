"""Tests for the date and change logic of scripts/macro/indicadores.py (QuantAgent-577).

No network: the fetchers are not exercised. Expected values are worked out by hand in the comments.
A regression here means the weekly macro note reports a change against the wrong day.
"""

import importlib.util
from datetime import date
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "macro" / "indicadores.py"
spec = importlib.util.spec_from_file_location("macro_indicadores", SCRIPT)
ind = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ind)

# Two trading weeks with a holiday on Fri 2026-09-25 and nothing on weekends.
SERIES = [
    (date(2026, 9, 21), 100.0),
    (date(2026, 9, 22), 101.0),
    (date(2026, 9, 23), 102.0),
    (date(2026, 9, 24), 104.0),  # last close of the first week: the Friday is missing
    (date(2026, 9, 28), 105.0),
    (date(2026, 9, 29), 106.0),
    (date(2026, 9, 30), 107.0),
    (date(2026, 10, 1), 108.0),
    (date(2026, 10, 2), 110.5),  # Friday close of the second week
    (date(2026, 10, 5), 90.0),  # Monday after: must not leak into the weekly numbers
]


@pytest.mark.parametrize(
    "as_of, expected",
    [
        (date(2026, 10, 2), date(2026, 10, 2)),  # a Friday is its own week close
        (date(2026, 10, 4), date(2026, 10, 2)),  # the Sunday run reports the Friday two days back
        (date(2026, 10, 7), date(2026, 10, 2)),  # a midweek test run still reports the previous Friday
        (date(2026, 10, 8), date(2026, 10, 2)),  # Thursday: the current week has not closed yet
    ],
)
def test_week_close_is_the_last_friday_on_or_before(as_of, expected):
    assert ind.week_close(as_of) == expected


def test_weekly_compares_friday_close_with_last_close_of_previous_week():
    w = ind.weekly(SERIES, date(2026, 10, 4), "pct")
    assert (w["date"], w["value"]) == ("2026-10-02", 110.5)
    # Previous Friday (09-25) has no observation, so the Thursday close stands in for it.
    assert (w["prev_date"], w["prev_value"]) == ("2026-09-24", 104.0)
    assert w["change"] == pytest.approx(6.25)  # 110.5 / 104 - 1 = 6.25 %


def test_weekly_ignores_data_after_the_week_close_but_reports_it_as_latest():
    w = ind.weekly(SERIES, date(2026, 10, 7), "pct")
    assert w["value"] == 110.5  # the Monday drop to 90 is not this week's close
    assert (w["latest_date"], w["latest_value"]) == ("2026-10-05", 90.0)


def test_weekly_absolute_change_for_rates():
    rates = [(date(2026, 9, 25), 5.11), (date(2026, 10, 2), 5.28)]
    assert ind.weekly(rates, date(2026, 10, 4), "abs")["change"] == pytest.approx(0.17)  # 5.28 - 5.11


def test_weekly_reports_no_change_when_the_source_has_nothing_new():
    stale = [(date(2026, 9, 23), 77.0)]  # a single old observation serves as both closes
    w = ind.weekly(stale, date(2026, 10, 4), "pct")
    assert w["value"] == 77.0
    assert w["change"] is None and w["prev_value"] is None


def test_weekly_fails_without_observations_up_to_the_close():
    with pytest.raises(ValueError):
        ind.weekly([(date(2026, 10, 5), 1.0)], date(2026, 10, 4), "pct")


def test_yoy_uses_the_same_month_one_year_earlier():
    monthly = [(date(2025, 8, 1), 320.0), (date(2025, 9, 1), 321.0), (date(2026, 8, 1), 330.88), (date(2026, 9, 1), 330.63)]
    readings = ind.yoy(monthly)
    assert [d for d, _ in readings] == [date(2026, 8, 1), date(2026, 9, 1)]
    assert readings[0][1] == pytest.approx(3.4)  # 330.88 / 320 - 1 = 3.4 %
    assert readings[1][1] == pytest.approx(3.0, abs=0.01)  # 330.63 / 321 - 1 = 3.0 %
