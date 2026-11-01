from datetime import date, datetime

from click.testing import CliRunner
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from quantagent.cli.__main__ import cli
from quantagent.database import Base
from quantagent.digest_m5 import build_digest_m5, consultar_datos_m5
from quantagent.models import Order, OrderSide, OrderStatus, OrderType, Position, TradeIntentRecord


def _database(path):
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    return engine


def _intent(symbol, status, when, *, reason=None, order_id=None, entry=100, stop=90):
    return TradeIntentRecord(created_at=when, updated_at=when, bucket_id="m5", strategy_id="test",
                             symbol=symbol, intent={"entry_price": entry, "stop_loss": stop},
                             status=status, reject_reason=reason, order_id=order_id, reasoning_metadata={})


def test_digest_m5_reports_rejection_symbol_and_reason(tmp_path):
    engine = _database(tmp_path / "m5.db")
    with Session(engine) as session:
        session.add(_intent("QQQ", "rechazado_riesgo", datetime(2026, 10, 29, 12), reason="max_heat"))
        session.commit()
        text = build_digest_m5(date(2026, 10, 29), consultar_datos_m5(session, date(2026, 10, 29)))
    assert "- QQQ: max_heat" in text


def test_digest_m5_uses_open_stops_and_calculates_heat(tmp_path):
    engine = _database(tmp_path / "m5.db")
    with Session(engine) as session:
        session.add_all([
            Position(symbol="SPY", quantity=2, average_entry_price=100, current_price=105,
                     unrealized_pnl=30, unrealized_pnl_pct=0.15, side=OrderSide.BUY),
            Position(symbol="QQQ", quantity=1, average_entry_price=50, current_price=52,
                     unrealized_pnl=20, unrealized_pnl_pct=0.04, side=OrderSide.BUY),
        ])
        session.add_all([
            Order(symbol="SPY", side=OrderSide.SELL, order_type=OrderType.STOP, quantity=2,
                  stop_price=90, status=OrderStatus.PENDING),
            Order(symbol="SPY", side=OrderSide.SELL, order_type=OrderType.STOP, quantity=2,
                  stop_price=80, status=OrderStatus.CANCELLED),
        ])
        session.add(_intent("QQQ", "aprobado", datetime(2026, 10, 28), entry=50, stop=45))
        session.commit()
        text = build_digest_m5(date(2026, 10, 29), consultar_datos_m5(session, date(2026, 10, 29)))
    assert "- SPY: 90.00" in text and "- QQQ: 45.00" in text
    assert "Heat: 0.82 % (límite 2.00 %)" in text


def test_digest_m5_explicitly_reports_no_intents(tmp_path):
    engine = _database(tmp_path / "m5.db")
    with Session(engine) as session:
        text = build_digest_m5(date(2026, 10, 29), consultar_datos_m5(session, date(2026, 10, 29)))
    assert "Sin actividad de intents hoy." in text
    assert "- Sin posiciones abiertas" in text


def test_digest_m5_cli_uses_selected_database(tmp_path):
    db_path = tmp_path / "m5.db"
    engine = _database(db_path)
    with Session(engine) as session:
        session.add(_intent("QQQ", "rechazado_riesgo", datetime(2026, 10, 29, 12), reason="max_heat"))
        session.commit()
    result = CliRunner().invoke(cli, ["trading", "digest", "--bucket", "m5", "--date", "2026-10-29",
                                      "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert "Resumen M5 — 2026-10-29" in result.output and "- QQQ: max_heat" in result.output
