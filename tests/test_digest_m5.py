from datetime import date, datetime

from click.testing import CliRunner
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from quantagent.cli.__main__ import cli
from quantagent.cli import reconcile
from quantagent.database import Base
from quantagent.digest_m5 import build_digest_m5, build_digest_m5_mensajes, consultar_datos_m5
from quantagent.models import Environment, Fill, Order, OrderSide, OrderStatus, OrderType, Position, SchedulerHeartbeat, TradeIntentRecord


def _database(path):
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    return engine


def _intent(symbol, status, when, *, reason=None, order_id=None, entry=100, stop=90):
    return TradeIntentRecord(created_at=when, updated_at=when, bucket_id="m5", strategy_id="test",
                             symbol=symbol, intent={"entry_price": entry, "stop_loss": stop},
                             status=status, reject_reason=reason, order_id=order_id, reasoning_metadata={})


def _completed(day):
    return SchedulerHeartbeat(timestamp=datetime.combine(day, datetime.min.time()), status="completed",
                              completed_at=datetime.combine(day, datetime.min.time()),
                              environment=Environment.PAPER, stats={"modo": "diario"})


class _Broker:
    def get_account(self):
        return {"equity": "3050", "cash": "1000"}


def test_digest_m5_reports_rejection_symbol_and_reason(tmp_path):
    engine = _database(tmp_path / "m5.db")
    with Session(engine) as session:
        session.add(_intent("QQQ", "rechazado_riesgo", datetime(2026, 10, 29, 12), reason="max_heat"))
        session.add(_completed(date(2026, 10, 29)))
        session.commit()
        text = build_digest_m5(date(2026, 10, 29), consultar_datos_m5(session, date(2026, 10, 29), _Broker()))
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
        session.add(_completed(date(2026, 10, 29)))
        session.add_all([
            Order(symbol="SPY", side=OrderSide.SELL, order_type=OrderType.STOP, quantity=2,
                  stop_price=90, status=OrderStatus.PENDING),
            Order(symbol="SPY", side=OrderSide.SELL, order_type=OrderType.STOP, quantity=2,
                  stop_price=80, status=OrderStatus.CANCELLED),
        ])
        session.add(_intent("QQQ", "aprobado", datetime(2026, 10, 28), entry=50, stop=45))
        session.commit()
        text = build_digest_m5(date(2026, 10, 29), consultar_datos_m5(session, date(2026, 10, 29), _Broker()))
    assert "- SPY: 90.00" in text and "- QQQ: 45.00" in text
    assert "Heat: 0.82 % (límite 2.00 %)" in text


def test_digest_m5_explicitly_reports_no_intents(tmp_path):
    engine = _database(tmp_path / "m5.db")
    with Session(engine) as session:
        text = build_digest_m5(date(2026, 10, 29), consultar_datos_m5(session, date(2026, 10, 29), _Broker()))
    assert "Sin actividad de intents hoy." in text
    assert text.startswith("ALERTA: el ciclo diario M5 no terminó OK")
    assert "- Sin posiciones abiertas" in text


def test_digest_m5_cli_uses_selected_database(tmp_path, monkeypatch):
    db_path = tmp_path / "m5.db"
    engine = _database(db_path)
    with Session(engine) as session:
        session.add(_intent("QQQ", "rechazado_riesgo", datetime(2026, 10, 29, 12), reason="max_heat"))
        session.add(_completed(date(2026, 10, 29)))
        session.commit()
    original_consultar = reconcile.consultar_datos_m5
    monkeypatch.setattr(reconcile, "consultar_datos_m5",
                        lambda session, day: original_consultar(session, day, _Broker()))
    result = CliRunner().invoke(cli, ["trading", "digest", "--bucket", "m5", "--date", "2026-10-29",
                                      "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert "Resumen M5 — 2026-10-29" in result.output and "- QQQ: max_heat" in result.output


def test_digest_m5_alerta_error_y_completed_conserva_texto(tmp_path):
    engine = _database(tmp_path / "m5.db")
    day = date(2026, 10, 29)
    with Session(engine) as session:
        session.add(SchedulerHeartbeat(timestamp=datetime.combine(day, datetime.min.time()), status="error",
                                       environment=Environment.PAPER, stats={"modo": "diario"},
                                       error_message="RuntimeError: falló"))
        session.commit()
        data = consultar_datos_m5(session, day, _Broker())
        assert "ALERTA: el ciclo diario M5 falló: RuntimeError: falló" in build_digest_m5(day, data)
        data["ciclo"].append({"modo": "diario", "status": "completed", "error_message": None})
        assert not build_digest_m5(day, data).startswith("ALERTA:")


def test_digest_m5_secciones_completas_y_resumen_corto():
    datos = {
        "aprobados": [{"symbol": f"APP{i:02}", "qty": i + 1, "entry": 100, "stop": 90} for i in range(16)],
        "rechazados": [{"symbol": f"REJ{i:02}", "reject_reason": "riesgo " + "x" * 200} for i in range(40)],
        "posiciones": [{"symbol": f"POS{i:02}", "quantity": i + 1, "avg_entry": 100,
                        "unrealized_pnl": i, "stop": 90} for i in range(20)],
        "semanal": ["Candidatos de la semana", *[f"CAND{i:02}: " + "y" * 200 for i in range(40)]],
    }
    datos["cuenta"] = {"equity": 3190, "cash": 1200, "consulted_at": datetime(2026, 10, 29)}
    messages = build_digest_m5_mensajes(date(2026, 10, 29), datos)
    assert len(messages[0].splitlines()) <= 11
    assert all(len(message) <= 3900 for message in messages)
    assert "Equity: 3190.00 USD" in messages[0] and "Intents aprobados: 16; rechazados: 40" in messages[0]
    text = build_digest_m5(date(2026, 10, 29), datos)
    assert "[Resumen truncado]" not in text and "... y " not in text
    for section in ("aprobados", "rechazados", "posiciones"):
        for item in datos[section]:
            assert any(item["symbol"] in message for message in messages[1:])
    for line in datos["semanal"]:
        assert line in text
    assert len([message for message in messages if "REJ" in message]) > 1
    assert len([message for message in messages if "CAND" in message]) > 1


def test_digest_m5_account_and_realized_pnl_from_fills():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    class Broker:
        def get_account(self):
            return {"equity": "3100", "cash": "900"}
    day = date(2026, 10, 29)
    with Session(engine) as session:
        buy = Order(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=2,
                    status=OrderStatus.FILLED, environment=Environment.PAPER)
        sell = Order(symbol="SPY", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=1,
                     status=OrderStatus.FILLED, environment=Environment.PAPER)
        session.add_all([buy, sell]); session.flush()
        session.add_all([Fill(order_id=buy.id, quantity=2, price=100, commission=0, filled_at=datetime(2026,10,28)),
                         Fill(order_id=sell.id, quantity=1, price=110, commission=0, filled_at=datetime(2026,10,29))])
        session.commit()
        data = consultar_datos_m5(session, day, Broker())
    summary = build_digest_m5(day, data)
    assert "Equity: 3100.00 USD" in summary and "Caja: 900.00 USD" in summary
    assert "P&L del día realizado: 10.00 USD" in summary
    assert "Caja: no disponible" not in summary and "P&L del día realizado: sin datos" not in summary


def test_digest_m5_account_failure_is_reported_without_failing_digest():
    class Broker:
        def get_account(self):
            raise RuntimeError("secret detail")
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        data = consultar_datos_m5(session, date(2026, 10, 29), Broker())
    summary = build_digest_m5(date(2026, 10, 29), data)
    assert "Equity: no disponible: RuntimeError" in summary
    assert "Caja: no disponible: RuntimeError" in summary
    assert "secret detail" not in summary


def test_digest_m5_account_without_cash_reports_unavailable():
    class Broker:
        def get_account(self):
            return {"equity": "3000"}
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        data = consultar_datos_m5(session, date(2026, 10, 29), Broker())
    summary = build_digest_m5(date(2026, 10, 29), data)
    assert "Equity: 3000.00 USD" in summary
    assert "Caja: no disponible" in summary
    assert "Caja: no disponible: RuntimeError" not in summary
