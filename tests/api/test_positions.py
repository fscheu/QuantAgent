"""Regression coverage with SQLite fills and an offline Alpaca-shaped oracle."""

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from apps.api.accounts import add_account
from apps.api.alarms import evaluate_account_alarms
from apps.api.app import app, get_db
from quantagent.database import Base
from quantagent.models import (
    Account,
    Environment,
    Fill,
    MarketData,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

DAY = datetime(2026, 11, 2)


def fill(db, symbol, side, quantity, price, *, environment=Environment.PAPER):
    order = Order(symbol=symbol, side=side, quantity=quantity, order_type=OrderType.MARKET,
                  status=OrderStatus.FILLED, filled_quantity=quantity, environment=environment,
                  created_at=DAY)
    db.add(order)
    db.flush()
    db.add(Fill(order_id=order.id, quantity=quantity, price=price, filled_at=DAY))


def quote(db, symbol, price, timestamp=DAY):
    db.add(MarketData(symbol=symbol, timeframe="1d", timestamp=timestamp,
                      open=price, high=price, low=price, close=price, volume=100))


@pytest.fixture
def portfolio(tmp_path):
    source_path = tmp_path / 'nucleo.db'
    source = create_engine(f'sqlite:///{source_path}')
    master = create_engine(f'sqlite:///{tmp_path / "master.db"}')
    Base.metadata.create_all(source)
    Base.metadata.create_all(master, tables=[Account.__table__])
    with Session(source) as ledger, Session(master) as db:
        account = add_account(db, id='nucleo', name='Núcleo', env='paper', kind='alpaca',
                              strategy='nucleo-curva', database_url=f'sqlite:///{source_path}')
        app.dependency_overrides[get_db] = lambda: db
        with TestClient(app) as client:
            yield ledger, db, account, client, source_path
        app.dependency_overrides.clear()
    source.dispose()
    master.dispose()


def test_nineteen_longs_match_offline_alpaca_despite_overwritten_positions(portfolio):
    ledger, db, account, client, path = portfolio
    oracle = json.loads((Path(__file__).parents[1] / 'fixtures/bixh-alpaca-positions.json').read_text())
    for row in oracle:
        qty = Decimal(row['qty'])
        fill(ledger, row['symbol'], OrderSide.BUY, qty + 2, 100)
        fill(ledger, row['symbol'], OrderSide.SELL, 2, 105)
        # The buggy engine has overwritten the holding with the last sell fill.
        ledger.add(Position(symbol=row['symbol'], quantity=2, average_entry_price=105,
                            current_price=105, unrealized_pnl=0, unrealized_pnl_pct=0, side=OrderSide.SELL))
        quote(ledger, row['symbol'], 90, datetime(2026, 10, 30))
        quote(ledger, row['symbol'], 110)
    # Other environments must not contaminate the paper holdings.
    fill(ledger, 'DBC', OrderSide.SELL, 100, 50, environment=Environment.BACKTEST)
    ledger.commit()
    response = client.get('/api/accounts/nucleo/positions')
    assert response.status_code == 200
    positions = response.json()
    assert len(positions) == 19
    for actual, expected in zip(positions, oracle):
        assert actual['symbol'] == expected['symbol']
        assert actual['side'] == expected['side']
        for key in ('qty', 'avg_entry_price', 'current_price', 'market_value', 'unrealized_pl'):
            assert actual[key] == float(expected[key])
        assert actual['unrealized_pl_pct'] == pytest.approx(.1)
        assert actual['status'] == 'executed'
        assert actual['stop_price'] is None and actual['stop_distance'] is None
    total = sum(float(row['market_value']) for row in oracle)
    weights = {row['symbol']: float(row['market_value']) / total for row in oracle}
    path.with_suffix('.sombra.json').write_text(json.dumps({'2026-11-02': {'nucleo': {'pesos': weights}}}))
    now = datetime(2026, 11, 2, 20, tzinfo=timezone.utc)
    assert 'shadow_divergence' not in {a.kind for a in evaluate_account_alarms(account, db, now)}
    # Concentrate the true ledger while keeping the corrupt positions table unchanged.
    fill(ledger, 'DBC', OrderSide.BUY, 1000, 100)
    ledger.commit()
    assert 'shadow_divergence' in {a.kind for a in evaluate_account_alarms(account, db, now)}


def test_pending_orders_and_unpriced_positions(portfolio):
    ledger, _, _, client, _ = portfolio
    fill(ledger, 'HELD', OrderSide.BUY, 4, 20)
    for symbol, status, filled, qty in [('HELD', OrderStatus.PENDING, 0, 2),
                                       ('NEW', OrderStatus.PENDING, 0, 3),
                                       ('PARTIAL', OrderStatus.PARTIALLY_FILLED, 1, 3),
                                       ('CANCELLED', OrderStatus.CANCELLED, 0, 3)]:
        ledger.add(Order(symbol=symbol, side=OrderSide.BUY, quantity=qty, filled_quantity=filled,
                         status=status, environment=Environment.PAPER, order_type=OrderType.MARKET))
    fill(ledger, 'PARTIAL', OrderSide.BUY, 1, 25)
    ledger.commit()
    positions = {p['symbol']: p for p in client.get('/api/accounts/nucleo/positions').json()}
    assert set(positions) == {'HELD', 'NEW', 'PARTIAL'}
    assert positions['HELD']['qty'] == 4
    assert positions['PARTIAL']['qty'] == 1
    assert positions['NEW']['qty'] == 3 and positions['NEW']['avg_entry_price'] is None
    for position in positions.values():
        assert position['status'] == 'pending'
        assert position['current_price'] is None
        assert position['market_value'] is None and position['unrealized_pl'] is None
        assert position['stop_price'] is None


def test_reductions_close_and_reversal_preserve_cost_and_side(portfolio):
    ledger, _, _, client, _ = portfolio
    for symbol, operations in {
        'REDUCED': [('buy', 10, 100), ('buy', 10, 120), ('sell', 5, 150)],
        'CLOSED': [('buy', 10, 100), ('sell', 10, 120)],
        'SHORT': [('sell', 10, 120), ('buy', 4, 100)],
        'FLIPPED': [('buy', 10, 100), ('sell', 14, 120)],
        'REOPENED': [('buy', 10, 100), ('sell', 10, 120), ('buy', 3, 130)],
    }.items():
        for side, qty, price in operations:
            fill(ledger, symbol, OrderSide(side), qty, price)
        quote(ledger, symbol, 100)
    ledger.commit()
    positions = {p['symbol']: p for p in client.get('/api/accounts/nucleo/positions').json()}
    assert set(positions) == {'REDUCED', 'SHORT', 'FLIPPED', 'REOPENED'}
    for symbol, side, qty, entry, pnl in [('REDUCED', 'long', 15, 110, -150),
                                        ('SHORT', 'short', 6, 120, 120),
                                        ('FLIPPED', 'short', 4, 120, 80),
                                        ('REOPENED', 'long', 3, 130, -90)]:
        actual = positions[symbol]
        assert (actual['side'], actual['qty'], actual['avg_entry_price'], actual['unrealized_pl']) == (side, qty, entry, pnl)


def test_shadow_uses_equity_and_excludes_priced_pending_only_orders(portfolio):
    from datetime import date

    from quantagent.models import EquitySnapshot

    ledger, db, account, client, path = portfolio
    fill(ledger, 'HELD', OrderSide.BUY, 4, 20)
    quote(ledger, 'HELD', 25)
    quote(ledger, 'NEW', 100)
    ledger.add(Order(symbol='NEW', side=OrderSide.SELL, quantity=50, price=90, filled_quantity=0,
                     status=OrderStatus.PENDING, environment=Environment.PAPER, order_type=OrderType.LIMIT))
    ledger.add(EquitySnapshot(snapshot_date=date(2026, 11, 2), taken_at=DAY,
                             equity=200, cash=100, invested=100))
    ledger.commit()
    positions = {p['symbol']: p for p in client.get('/api/accounts/nucleo/positions').json()}
    assert positions['NEW']['current_price'] == 100
    assert positions['NEW']['market_value'] is None and positions['NEW']['unrealized_pl'] is None
    path.with_suffix('.sombra.json').write_text(json.dumps({'2026-11-02': {'nucleo': {'pesos': {'HELD': .5}}}}))
    alarms = evaluate_account_alarms(account, db, datetime(2026, 11, 2, 20, tzinfo=timezone.utc))
    assert 'shadow_divergence' not in {alarm.kind for alarm in alarms}


def test_shadow_does_not_compare_incomplete_valuations(portfolio):
    ledger, db, account, _, path = portfolio
    fill(ledger, 'PRICED', OrderSide.BUY, 4, 20)
    fill(ledger, 'UNPRICED', OrderSide.BUY, 4, 20)
    quote(ledger, 'PRICED', 25)
    ledger.commit()
    path.with_suffix('.sombra.json').write_text(json.dumps({'2026-11-02': {'nucleo': {'pesos': {'PRICED': .5, 'UNPRICED': .5}}}}))
    alarms = evaluate_account_alarms(account, db, datetime(2026, 11, 2, 20, tzinfo=timezone.utc))
    assert 'shadow_divergence' not in {alarm.kind for alarm in alarms}
