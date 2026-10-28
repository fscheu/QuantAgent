import sqlite3
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from quantagent.digest import build_digest
from quantagent.database import Base
from quantagent.digest_datos import consultar_datos_cartera, consultar_datos_paper, consultar_valuacion_actual
from quantagent import settings
from quantagent.models import ActivePosition, Environment, ExitPolicy, Fill, Order, OrderSide, OrderStatus, OrderType, Position, SchedulerHeartbeat, Trade
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from click.testing import CliRunner

from quantagent.cli.__main__ import cli
from quantagent.trading.alpaca_broker import AlpacaBroker


@pytest.mark.parametrize("count", [0, 1, 200])
def test_digest_cartera_valuacion_ejecuciones_y_pendientes(count):
    snapshot = {"equity": 1000, "cash": 50, "consulted_at": datetime(2026, 10, 28),
                "notice": "Foto actual de Alpaca; no es saldo histórico de --date.",
                "portfolios": [{"name": "alpha", "positions": [
                    {"symbol": "SPY", "quantity": 2, "price": 475, "value": 950, "unrealized_pnl": -30}]}]}
    executions = [{"symbol": "SPY", "side": "buy", "quantity": 2, "price": 475,
                   "commission": 1, "filled_at": datetime(2026, 10, 25, 12)}] * count
    pending = [{"symbol": "QQQ", "side": "sell", "status": "partially_filled",
                "quantity": 5, "filled_quantity": 2}] * count
    text = build_digest(date(2026, 10, 25), [], [], [], valuation=snapshot,
                        executions=executions, pending=pending)
    assert "Equity: 1000.00 USD" in text and "Caja: 50.00 USD (5.00 %)" in text
    assert "reserva objetivo 5 %" in text and snapshot["notice"] in text
    assert "Consultada: 2026-10-28" in text and "2026-10-25" in text
    assert "valor de mercado 950.00 USD" in text and "P&L no realizado -30" in text
    assert "Sin datos de P&L realizado" in text and "P&L realizado diario: 0" not in text
    assert f"Ejecuciones del día ({count})" in text and f"Órdenes pendientes ({count})" in text
    if count == 1:
        assert "SPY buy 2 @ 475; comisión 1" in text
        assert "QQQ sell: partially_filled, 2/5 ejecutadas" in text
    assert len(text) < 4000
    assert ("[Resumen truncado]" in text) == (count == 200)


def test_digest_sin_datos_nuevos_conserva_formato():
    text = build_digest(date(2026, 10, 25), [], [], [])
    assert text == ("Resumen de paper trading — 2026-10-25\n\n## Alertas y avisos\n- Ninguno\n\n"
                    "## Señal del scheduler\n- ALERT: sin ciclo completado en más de un día hábil\n"
                    "- Sin heartbeat\n\n## Actividad y P&L (0 operaciones)\nP&L realizado diario: 0.00\n"
                    "P&L realizado acumulado: 0.00\nSin actividad de trading")


@pytest.mark.parametrize("failure", [None, "account", "positions", "missing"])
def test_valuacion_actual_alpaca_sin_api_real(failure):
    class Client:
        def get_account(self):
            if failure == "account":
                raise RuntimeError("unavailable")
            return SimpleNamespace(equity="1000", cash="50")

        def get_all_positions(self):
            if failure == "positions":
                raise RuntimeError("unavailable")
            return [SimpleNamespace(symbol=symbol, current_price="475", market_value="950",
                                    unrealized_pl="-30") for symbol in ["QQQ", "SPY"]
                    if failure != "missing" or symbol != "SPY"]

    local = [{"name": "alpha", "positions": [
        {"symbol": "SPY", "quantity": 2, "price": 500, "value": 1000, "unrealized_pnl": 20}]}]
    broker = AlpacaBroker(Client())
    if failure in ("account", "positions"):
        with pytest.raises(RuntimeError, match="Valuación actual: error al consultar Alpaca"):
            consultar_valuacion_actual(broker, local)
    elif failure == "missing":
        with pytest.raises(ValueError, match="Alpaca sin valuación para SPY"):
            consultar_valuacion_actual(broker, local)
    else:
        before = datetime.now(timezone.utc)
        snapshot = consultar_valuacion_actual(broker, local)
        assert snapshot["equity"] == 1000 and snapshot["cash"] == 50
        assert snapshot["portfolios"] == [{"name": "alpha", "positions": [
            {"symbol": "SPY", "quantity": 2, "price": 475, "value": 950, "unrealized_pnl": -30}]}]
        assert snapshot["source"] == "Alpaca"
        assert before <= snapshot["consulted_at"] <= datetime.now(timezone.utc)
        assert snapshot["notice"] == "Foto actual de Alpaca; no es saldo histórico de --date."
    assert local[0]["positions"][0] == {
        "symbol": "SPY", "quantity": 2, "price": 500, "value": 1000, "unrealized_pnl": 20}


def test_consultar_cartera_sin_active_position_ni_trade(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'cartera.db'}")
    Base.metadata.create_all(engine, tables=[Position.__table__, Order.__table__, Fill.__table__])
    start = datetime(2026, 10, 25)
    with Session(engine) as session:
        session.add(Position(symbol="SPY", quantity=2, average_entry_price=490,
                             current_price=500, unrealized_pnl=20,
                             unrealized_pnl_pct=0.02, side=OrderSide.BUY))
        for symbol, status, environment, offsets in [
            ("SPY", OrderStatus.FILLED, Environment.PAPER, [-1, 0, 86400]),
            ("QQQ", OrderStatus.PARTIALLY_FILLED, Environment.PAPER, [3600, 7200]),
            ("IWM", OrderStatus.PENDING, Environment.PAPER, []),
            ("DIA", OrderStatus.FILLED, Environment.PAPER, []),
            ("LIVE", OrderStatus.PARTIALLY_FILLED, Environment.PROD, [0]),
            ("BT", OrderStatus.PENDING, Environment.BACKTEST, [0]),
            ("CANCEL", OrderStatus.CANCELLED, Environment.PAPER, []),
            ("REJECT", OrderStatus.REJECTED, Environment.PAPER, []),
            ("EXPIRE", OrderStatus.EXPIRED, Environment.PAPER, []),
        ]:
            session.add(Order(symbol=symbol, side=OrderSide.BUY, order_type=OrderType.MARKET,
                              quantity=10, filled_quantity=len(offsets), status=status,
                              environment=environment, created_at=start - timedelta(days=2),
                              fills=[Fill(quantity=1, price=500, commission=1,
                                          filled_at=start + timedelta(seconds=offset)) for offset in offsets]))
        session.commit()
        portfolios, fills, pending = consultar_datos_cartera(session, start.date(), "alpha")
        assert portfolios == [{"name": "alpha", "positions": [
            {"symbol": "SPY", "quantity": 2, "price": 500, "value": 1000, "unrealized_pnl": 20}]}]
        assert fills == [
            {"symbol": symbol, "side": "buy", "quantity": 1, "price": 500,
             "commission": 1, "filled_at": start + timedelta(seconds=offset)}
            for symbol, offset in [("SPY", 0), ("QQQ", 3600), ("QQQ", 7200)]]
        assert pending == [
            {"symbol": symbol, "side": "buy", "status": status, "quantity": 10,
             "filled_quantity": quantity, "comment": None}
            for symbol, status, quantity in [("QQQ", "partially_filled", 2), ("IWM", "pending", 0)]]


def test_digest_renders_seeded_paper_data(tmp_path):
    db = sqlite3.connect(tmp_path / "paper.db")
    db.executescript("CREATE TABLE positions (symbol, quantity, price, pnl); CREATE TABLE trades (symbol, side, quantity, pnl);")
    db.execute("INSERT INTO positions VALUES ('SPY', 2, 500, 10)")
    db.execute("INSERT INTO trades VALUES ('SPY', 'sell', 1, 12)")
    positions = [{"symbol": row[0], "quantity": row[1], "price": row[2], "unrealized_pnl": row[3]} for row in db.execute("SELECT * FROM positions")]
    trades = [{"symbol": row[0], "side": row[1], "quantity": row[2], "pnl": row[3]} for row in db.execute("SELECT * FROM trades")]
    text = build_digest(date(2026, 10, 25), [{"name": "alpha", "positions": positions}], trades,
                        [{"timestamp": datetime(2026, 10, 25), "status": "completed"}], ["API warning"], ["positions match"], 1200)
    assert "Posiciones de alpha" in text and "SPY: 2 @ 500" in text
    assert "Actividad y P&L (1 operaciones)" in text and "P&L realizado diario: 12.00" in text
    assert "P&L realizado acumulado: 1200.00" in text and "API warning" in text
    assert "completed" in text and "Reconciliación" in text and "positions match" in text


def test_digest_warns_after_stale_unsuccessful_heartbeat():
    report_day = date.today()
    old = datetime.combine(report_day - timedelta(days=4), datetime.min.time())
    text = build_digest(report_day, [], [], [{"timestamp": old, "status": "error"}])
    assert "ALERT" in text


def test_digest_reports_no_activity_and_truncates_by_section():
    text = build_digest(date.today(), [], [], [])
    assert "Sin actividad de trading" in text and "Sin heartbeat" in text and "ALERT" in text


def test_digest_keeps_warning_and_heartbeat_with_many_trades():
    trades = [{"symbol": f"SYM{i}", "side": "sell", "quantity": 1, "pnl": 1} for i in range(200)]
    text = build_digest(date.today(), [], trades,
                        [{"timestamp": datetime.now(), "status": "completed"}], ["API warning"])
    assert "API warning" in text and "Señal del scheduler" in text
    assert "ALERT" not in text and "[Resumen truncado]" in text and len(text) < 4000


def test_digest_limits_oversized_warnings_and_reports_omitted_count():
    text = build_digest(date.today(), [], [], [], ["x" * 5000, "second warning"])
    assert len(text) < 4000
    assert "... y 1 más" in text


def test_digest_alerts_when_heartbeat_is_missing():
    text = build_digest(date.today(), [], [], [])
    assert "ALERT" in text


def test_consultar_datos_paper_usa_modelos_y_filtra_dia_y_ambiente(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'paper.db'}")
    Base.metadata.create_all(engine)
    day = date(2026, 10, 25)
    with Session(engine) as session:
        session.add(Position(symbol="SPY", quantity=2, average_entry_price=490,
                             current_price=500, unrealized_pnl=20,
                             unrealized_pnl_pct=0.02, side=OrderSide.BUY))
        session.add_all([
            ActivePosition(symbol="SPY", side=OrderSide.BUY, entry_price=490,
                           stop_loss=450, take_profit=550, quantity=2,
                           decision_timestamp=datetime(2026, 10, 24),
                           exit_policy=ExitPolicy.SL_TP_ONLY, environment=Environment.PAPER),
            ActivePosition(symbol="QQQ", side=OrderSide.BUY, entry_price=400,
                           stop_loss=350, take_profit=450, quantity=1,
                           decision_timestamp=datetime(2026, 10, 25, 9),
                           exit_policy=ExitPolicy.SL_TP_ONLY, environment=Environment.BACKTEST),
            ActivePosition(symbol="IWM", side=OrderSide.BUY, entry_price=200,
                           stop_loss=180, take_profit=220, quantity=1,
                           decision_timestamp=datetime(2026, 10, 26),
                           exit_policy=ExitPolicy.SL_TP_ONLY, environment=Environment.PAPER),
        ])
        session.add_all([
            Order(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=2,
                  filled_quantity=1, status=OrderStatus.PARTIALLY_FILLED,
                  created_at=datetime(2026, 10, 25, 11), comment="rebalance", environment=Environment.PAPER),
            Order(symbol="QQQ", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1,
                  created_at=datetime(2026, 10, 25, 11), environment=Environment.BACKTEST),
            Order(symbol="IWM", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=1,
                  created_at=datetime(2026, 10, 24, 11), environment=Environment.PAPER),
            Trade(symbol="SPY", entry_price=490, exit_price=500, quantity=1,
                  side=OrderSide.SELL, pnl=10, opened_at=datetime(2026, 10, 25, 9),
                  closed_at=datetime(2026, 10, 25, 10), environment=Environment.PAPER),
            Trade(symbol="QQQ", entry_price=400, exit_price=410, quantity=1,
                  side=OrderSide.SELL, pnl=10, opened_at=datetime(2026, 10, 25, 9),
                  closed_at=datetime(2026, 10, 25, 10), environment=Environment.BACKTEST),
            Trade(symbol="IWM", entry_price=200, exit_price=210, quantity=1,
                  side=OrderSide.SELL, pnl=10, opened_at=datetime(2026, 10, 24, 9),
                  closed_at=datetime(2026, 10, 24, 10), environment=Environment.PAPER),
            Trade(symbol="DIA", entry_price=300, exit_price=305, quantity=1,
                  side=OrderSide.SELL, pnl=5, opened_at=datetime(2026, 10, 20, 9),
                  closed_at=datetime(2026, 10, 20, 10), environment=Environment.PAPER),
            Trade(symbol="DIA", entry_price=300, exit_price=305, quantity=1,
                  side=OrderSide.SELL, pnl=100, opened_at=datetime(2026, 10, 26, 9),
                  closed_at=datetime(2026, 10, 26, 10), environment=Environment.PAPER),
            SchedulerHeartbeat(timestamp=datetime(2026, 10, 25, 10), status="completed",
                               environment=Environment.PAPER),
            SchedulerHeartbeat(timestamp=datetime(2026, 10, 22, 10), status="completed",
                               environment=Environment.PAPER),
            SchedulerHeartbeat(timestamp=datetime(2026, 10, 26, 10), status="error",
                               environment=Environment.PAPER),
            SchedulerHeartbeat(timestamp=datetime(2026, 10, 25, 10), status="error",
                               environment=Environment.BACKTEST),
            SchedulerHeartbeat(timestamp=datetime(2026, 10, 24, 10), status="error",
                               environment=Environment.PAPER),
        ])
        session.commit()
        portfolios, trades, heartbeats, realized_pnl, orders = consultar_datos_paper(session, day, "alpha")

    assert portfolios == [{"name": "alpha", "positions": [
        {"symbol": "SPY", "quantity": 2, "price": 500, "value": 1000, "unrealized_pnl": 20}
    ]}]
    assert trades == [{"symbol": "SPY", "side": "sell", "quantity": 1, "pnl": 10}]
    assert orders == [{"symbol": "SPY", "side": "buy", "status": "partially_filled",
                       "quantity": 2, "filled_quantity": 1, "comment": "rebalance"}]
    assert heartbeats == [{"timestamp": datetime(2026, 10, 22, 10),
                           "status": "completed", "error_message": None},
                          {"timestamp": datetime(2026, 10, 24, 10),
                           "status": "error", "error_message": None},
                          {"timestamp": datetime(2026, 10, 25, 10),
                           "status": "completed", "error_message": None}]
    assert realized_pnl == 25


def test_trading_digest_cli_reads_selected_sqlite_and_date(tmp_path):
    previous_database_url = settings.DATABASE_URL
    db_path = tmp_path / "paper.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Trade(symbol="SPY", entry_price=490, exit_price=500, quantity=1,
                          side=OrderSide.SELL, pnl=12, opened_at=datetime(2026, 10, 24, 9),
                          closed_at=datetime(2026, 10, 25, 10), environment=Environment.PAPER))
        session.add(SchedulerHeartbeat(timestamp=datetime(2026, 10, 25, 10), status="completed",
                                       environment=Environment.PAPER))
        session.commit()

    result = CliRunner().invoke(cli, ["trading", "digest", "--date", "2026-10-25",
                                      "--db", str(db_path), "--portfolio", "alpha"])
    assert result.exit_code == 0, result.output
    assert "2026-10-25" in result.output and "Posiciones de alpha" in result.output and "P&L realizado acumulado: 12.00" in result.output
    assert result.output.strip() and len(result.output) < 4000 and settings.DATABASE_URL == previous_database_url


def test_trading_digest_reconciles_shadow_weights_and_missing_date(tmp_path):
    db_path, shadow_path = tmp_path / "paper.db", tmp_path / "shadow.json"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Position(symbol="SPY", quantity=2, average_entry_price=400, current_price=500,
                             unrealized_pnl=200, unrealized_pnl_pct=.25, side=OrderSide.BUY))
        session.add(ActivePosition(symbol="SPY", side=OrderSide.BUY, entry_price=400, stop_loss=300,
                                   take_profit=600, quantity=2, decision_timestamp=datetime(2026, 10, 25),
                                   exit_policy=ExitPolicy.SL_TP_ONLY, environment=Environment.PAPER))
        session.commit()

    shadow_path.write_text('{"2026-10-23":{"alpha":{"pesos":{"SPY":1.0}}}}')
    args = ["trading", "digest", "--date", "2026-10-25", "--db", str(db_path),
            "--shadow", str(shadow_path), "--portfolio", "alpha"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0 and "difiere" not in result.output
    shadow_path.write_text('{"2026-10-23":{"alpha":{"pesos":{"SPY":0.4}}}}')
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1 and "SPY:" in result.output
    shadow_path.write_text('{}')
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0 and "Sombra sin datos" in result.output


def test_trading_digest_reports_rejections_and_dividend_source_notice(tmp_path):
    db_path = tmp_path / "paper.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all([
            Order(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1,
                  status=OrderStatus.REJECTED, created_at=datetime(2026, 10, 25, 11),
                  environment=Environment.PAPER),
            SchedulerHeartbeat(timestamp=datetime(2026, 10, 25, 10), status="completed",
                               environment=Environment.PAPER),
        ])
        session.commit()

    args = ["trading", "digest", "--date", "2026-10-25", "--db", str(db_path)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1
    assert "SPY: rechazo sin explicación" in result.output
    assert "Dividendos: sin fuente de dividendos." in result.output

    with Session(engine) as session:
        session.query(Order).filter_by(symbol="SPY").update({"comment": "saldo insuficiente"})
        session.commit()
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0
    assert "SPY: rechazo, saldo insuficiente" in result.output
    assert "Dividendos: sin fuente de dividendos." in result.output


def test_trading_digest_reads_alpaca_dividends_and_reports_api_error(tmp_path):
    from quantagent.strategy.assembler import StrategyAssembler

    class FakeClient:
        def get(self, path, data):
            assert path == "/account/activities/DIV" and data == {"date": "2026-10-25"}
            return [{"id": "activity-1", "activity_type": "DIV", "date": "2026-10-25",
                     "net_amount": "1.25", "symbol": "SPY", "per_share_amount": "1.25", "qty": "1"}]

    class FakeBroker:
        def __init__(self, client):
            self.client = client

    class FailingClient:
        def get(self, path, data):
            raise RuntimeError("API unavailable")

    previous = StrategyAssembler.BROKER_CONSTRUCTORS["alpaca_paper"]
    StrategyAssembler.BROKER_CONSTRUCTORS["alpaca_paper"] = lambda config: FakeBroker(FakeClient())
    try:
        args = ["trading", "digest", "--date", "2026-10-25", "--db", str(tmp_path / "paper.db"),
                "--dividendos-alpaca"]
        result = CliRunner().invoke(cli, args)
        assert result.exit_code == 0, result.output
        assert "SPY: dividendo 1.25 USD, tratamiento efectivo en cuenta" in result.output
        assert "sin fuente de dividendos" not in result.output

        StrategyAssembler.BROKER_CONSTRUCTORS["alpaca_paper"] = lambda config: FakeBroker(FailingClient())
        result = CliRunner().invoke(cli, args)
        assert result.exit_code == 1 and "Dividendos: error al consultar Alpaca (RuntimeError)." in result.output
    finally:
        StrategyAssembler.BROKER_CONSTRUCTORS["alpaca_paper"] = previous
