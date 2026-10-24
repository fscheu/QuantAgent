import json
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from quantagent.cli.__main__ import cli
from quantagent.models import ActivePosition, Environment, ExitPolicy, OrderSide
from quantagent.strategy.assembler import StrategyAssembler
from quantagent.trading.alpaca_broker import AlpacaBroker


@pytest.mark.parametrize(
    "weight, status, comment, treatment, expected_exit, expected_text",
    [
        (
            0.50001, "rejected", "Saldo insuficiente; sin explicación adicional del broker",
            "acreditado", 0, "sin explicación adicional del broker",
        ),
        (0.6, "rejected", "motivo", "acreditado", 1, "SPY: peso paper 0.6 difiere"),
        (0.5, "rejected", None, "acreditado", 1, "QQQ: rechazo sin explicación"),
        (0.5, "REJECTED", "", "acreditado", 1, "QQQ: rechazo sin explicación"),
        (0.5, "rejected", "motivo", None, 1, "IWM: dividendo 0.5 USD, falta tratamiento"),
        (0.50001, "accepted", None, "acreditado", 0, "Sin observaciones"),
        (
            0.50001, "accepted", None,
            "acreditado; se corrigió la falta tratamiento reportada",
            0, "se corrigió la falta tratamiento reportada",
        ),
    ],
)
def test_paper_digest_falla_solo_por_incumplimientos_estructurados(
    tmp_path, weight, status, comment, treatment, expected_exit, expected_text
):
    entrada = tmp_path / "digest.json"
    entrada.write_text(json.dumps({
        "paper_weights": {"SPY": weight}, "shadow_weights": {"SPY": 0.5},
        "orders": [{"status": status, "symbol": "QQQ", "comment": comment}],
        "dividends": [["IWM", 0.5, "USD", treatment]],
    }))
    result = CliRunner().invoke(cli, ["trading", "digest", str(entrada)])

    assert result.exit_code == expected_exit
    assert all(section in result.output for section in ("Pesos:", "Rechazos:", "Dividendos:"))
    assert expected_text in result.output


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
