from datetime import date, datetime

from quantagent.data.market_bars import market_bar_status


def test_market_bar_status_uses_last_closed_nyse_session_and_holiday_calendar():
    # July 4 is closed; before the July 5 close, July 3 remains the latest session.
    session, stale, missing = market_bar_status({date(2024, 7, 3)}, datetime(2024, 7, 5, 14))
    assert (session, stale, missing) == (date(2024, 7, 3), False, [])


def test_market_bar_status_skips_weekend_and_marks_after_close_stale():
    session, stale, missing = market_bar_status({date(2024, 7, 3)}, datetime(2024, 7, 8, 22))
    assert session == date(2024, 7, 8)
    assert stale
    assert missing == [date(2024, 7, 5), date(2024, 7, 8)]


def test_market_bar_status_reports_intermediate_gaps():
    session, stale, missing = market_bar_status(
        {date(2024, 1, 2), date(2024, 1, 4)}, datetime(2024, 1, 5, 22)
    )
    assert session == date(2024, 1, 5)
    assert stale
    assert missing == [date(2024, 1, 3), date(2024, 1, 5)]
