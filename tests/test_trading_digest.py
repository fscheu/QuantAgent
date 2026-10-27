import sqlite3
from datetime import date, datetime, timedelta

from quantagent.digest import build_digest
from quantagent.database import Base
from quantagent.digest_datos import consultar_datos_paper
from quantagent import settings
from quantagent.models import ActivePosition, Environment, ExitPolicy, Order, OrderSide, OrderStatus, OrderType, Position, SchedulerHeartbeat, Trade
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from click.testing import CliRunner

from quantagent.cli.__main__ import cli


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
    shadow_path.write_text('{"2026-10-23":{"alpha":{"pesos":{"SPY":0.5}}}}')
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
    from types import SimpleNamespace
    from quantagent.strategy.assembler import StrategyAssembler

    class FakeClient:
        def get(self, path, data):
            assert path == "/account/activities/DIV" and data == {"date": "2026-10-25"}
            return [SimpleNamespace(symbol="SPY", net_amount="1.25")]

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
        assert result.exit_code == 1 and "Dividendos: error al consultar Alpaca." in result.output
    finally:
        StrategyAssembler.BROKER_CONSTRUCTORS["alpaca_paper"] = previous
