import json
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from quantagent.cli.__main__ import cli
from quantagent.models import ActivePosition, Environment, ExitPolicy, OrderSide, Position
from quantagent.strategy.assembler import StrategyAssembler
from quantagent.trading.alpaca_broker import AlpacaBroker
from quantagent.fabrica.cartera import guardar_sombra


@pytest.mark.parametrize("spy, extra, weight, code", [
    (.1171, "IWM", .001, 0), (.13, "IWM", .001, 1),
    (.1171, "IWM", .003, 1), (.1171, "UNKNOWN", .0002, 1),
    (.1171, "UNKNOWN", .00005, 0),
    (12120 / 99941, "IWM", .001, 0),
])
def test_digest_tolera_una_accion_con_db_y_sombra(db_session, tmp_path, monkeypatch, spy, extra, weight, code):
    for symbol, quantity, price in [("SPY", 19.2, 600), ("QQQ", 88.421, 1000), ("IWM", 0, 200)]:
        db_session.add(Position(symbol=symbol, quantity=quantity, current_price=price,
                                average_entry_price=price, unrealized_pnl=0,
                                unrealized_pnl_pct=0, side=OrderSide.BUY))
        if quantity:
            db_session.add(ActivePosition(
                symbol=symbol, side=OrderSide.BUY, entry_price=price, quantity=quantity,
                stop_loss=price / 2, take_profit=price * 2, is_active=True,
                decision_timestamp=datetime(2026, 10, 26),
                exit_policy=ExitPolicy.SL_TP_ONLY, environment=Environment.PAPER))
    db_session.commit()

    @contextmanager
    def session_scope():
        yield db_session

    monkeypatch.setattr("quantagent.cli.reconcile.utils.session_scope", session_scope)
    shadow = tmp_path / "shadow.json"
    guardar_sombra(shadow, "alpha", "2026-10-27", {"SPY": spy, "QQQ": 1 - spy - weight, extra: weight}, 99941)
    result = CliRunner().invoke(cli, ["trading", "digest", "--date", "2026-10-27",
                                      "--shadow", str(shadow), "--portfolio", "alpha"])
    assert result.exit_code == code, result.output
    if code:
        symbol = "SPY" if spy == .13 else extra
        tolerance = {"SPY": 600 / 99941, "IWM": 200 / 99941}.get(symbol, 1e-4)
        assert f"{symbol}: peso paper" in result.output
        assert f"(tolerancia {tolerance})" in result.output
    else:
        assert "difiere" not in result.output


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
