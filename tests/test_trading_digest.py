import sqlite3
from datetime import date, datetime, timedelta

from quantagent.digest import build_digest


def test_digest_renders_seeded_paper_data(tmp_path):
    db = sqlite3.connect(tmp_path / "paper.db")
    db.executescript("CREATE TABLE positions (symbol, quantity, price, pnl); CREATE TABLE trades (symbol, side, quantity, pnl);")
    db.execute("INSERT INTO positions VALUES ('SPY', 2, 500, 10)")
    db.execute("INSERT INTO trades VALUES ('SPY', 'sell', 1, 12)")
    positions = [{"symbol": row[0], "quantity": row[1], "price": row[2], "unrealized_pnl": row[3]} for row in db.execute("SELECT * FROM positions")]
    trades = [{"symbol": row[0], "side": row[1], "quantity": row[2], "pnl": row[3]} for row in db.execute("SELECT * FROM trades")]
    text = build_digest(date(2026, 10, 25), [{"name": "alpha", "positions": positions}], trades,
                        [{"timestamp": datetime(2026, 10, 25), "status": "completed"}], ["API warning"], ["positions match"], 1200)
    assert "alpha positions" in text and "SPY: 2 @ 500" in text
    assert "Activity and P&L (1 trades)" in text and "Daily realized P&L: 12.00" in text
    assert "Cumulative realized P&L: 1200.00" in text and "API warning" in text
    assert "completed" in text and "Reconciliation" in text and "positions match" in text


def test_digest_warns_after_stale_unsuccessful_heartbeat():
    report_day = date.today()
    old = datetime.combine(report_day - timedelta(days=4), datetime.min.time())
    text = build_digest(report_day, [], [], [{"timestamp": old, "status": "error"}])
    assert "no successful cycle in more than one business day" in text


def test_digest_reports_no_activity_and_truncates_by_section():
    text = build_digest(date.today(), [], [], [])
    assert "No trading activity" in text and "No heartbeat" in text
    huge = build_digest(date.today(), [], [], [], ["x" * 5000])
    assert len(huge) < 4000 and huge.endswith("[Digest truncated]")
    assert "## Warnings and errors" not in huge
