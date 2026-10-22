from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace

from click.testing import CliRunner
import pytest

from quantagent.cli.__main__ import cli
from quantagent.models import ActivePosition, Environment, ExitPolicy, OrderSide
from quantagent.strategy.assembler import StrategyAssembler
from quantagent.trading.alpaca_broker import AlpacaBroker


@pytest.mark.parametrize(
    "remote, expected",
    [
        ([SimpleNamespace(symbol="ETH", qty="1", side="long")], "BTC: local-only"),
        ([SimpleNamespace(symbol="BTC", qty="2", side="long")], "Sin divergencias"),
    ],
)
def test_reconcile_cli_reports_result_without_mutating_positions(db_session, monkeypatch, remote, expected):
    db_session.add(ActivePosition(
        symbol="BTC", side=OrderSide.BUY, entry_price=100, stop_loss=90,
        take_profit=110, quantity=2, exit_policy=ExitPolicy.SL_TP_ONLY,
        decision_timestamp=datetime.utcnow(), prediction_horizon=3,
        candles_since_entry=0, candles_direction=[],
        is_active=True, environment=Environment.PAPER,
    ))
    db_session.commit()
    before = db_session.query(ActivePosition).count()

    @contextmanager
    def session_scope():
        yield db_session

    monkeypatch.setattr("quantagent.cli.reconcile.utils.session_scope", session_scope)
    client = SimpleNamespace(get_all_positions=lambda: remote)
    monkeypatch.setitem(StrategyAssembler.BROKER_CONSTRUCTORS, "alpaca_paper", lambda config: AlpacaBroker(client))
    result = CliRunner().invoke(cli, ["trading", "reconciliar"])

    assert result.exit_code == 0, result.output
    assert expected in result.output
    assert db_session.query(ActivePosition).count() == before
    position = db_session.query(ActivePosition).one()
    assert position.is_active and position.quantity == 2
