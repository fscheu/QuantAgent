from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from quantagent.fabrica.transversal import PRIORS_M5
from quantagent.models import OrderSide, OrderType, TradeIntentRecord
from quantagent.portfolio.manager import PortfolioManager
from quantagent.strategy.base import TradingSignal, TradingStrategy
from quantagent.trading.cartera_paper import sesiones_cerradas_nyse
from quantagent.trading.pipeline_m5 import correr_diario, correr_semanal


class DatosSinteticos:
    def __init__(self, panel):
        self.panel = panel

    def get_ohlc(self, symbol, timeframe, start_date, end_date):
        df = self.panel[symbol].copy()
        return df[(df.timestamp >= pd.Timestamp(start_date)) & (df.timestamp <= pd.Timestamp(end_date))]


class BrokerMemoria:
    def __init__(self):
        self.orders = []

    def submit_order(self, order):
        self.orders.append(order)
        return order


class SiempreSenala(TradingStrategy):
    required_history_bars = 1

    def __init__(self, stop=95):
        self.stop = stop

    def generate_signal(self, kline_data, symbol, timeframe, current_price):
        return TradingSignal(decision="LONG", confidence=1, entry_price=current_price,
                             entry_order="stop", entry_expiry_bars=3, stop_loss=self.stop)

    def should_exit(self, position, current_price, ohlc_data):
        return False, None

    def should_reevaluate(self, position, current_price):
        return False


class SinStop(TradingStrategy):
    required_history_bars = 1

    def generate_signal(self, kline_data, symbol, timeframe, current_price):
        return TradingSignal(decision="LONG", confidence=1, entry_price=current_price, stop_loss=None)

    def should_exit(self, position, current_price, ohlc_data):
        return False, None

    def should_reevaluate(self, position, current_price):
        return False


@pytest.fixture
def mercado():
    as_of = datetime(2026, 10, 30, 22, tzinfo=timezone.utc)
    sessions = sesiones_cerradas_nyse(as_of, 504)
    n = 300
    idx = sessions[-n:]
    closes = 50 + np.arange(n) * 0.25 + 2 * np.sin(np.arange(n) / 8)
    closes[-1] = closes[-2] - 1.5
    panel = {}
    for i, symbol in enumerate(PRIORS_M5["universo"]):
        c = closes * (1 + i * 0.015)
        panel[symbol] = pd.DataFrame({
            "timestamp": idx, "open": c - 0.3, "high": c + 1, "low": c - 1,
            "close": c, "volume": 1000.0,
        })
    return as_of, DatosSinteticos(panel)


def test_pipeline_semanal_y_diario_etfs_y_stops(db_session, mercado, tmp_path):
    as_of, data = mercado
    path = tmp_path / "regimen.csv"
    regimen = correr_semanal(as_of, data_provider=data, db=db_session, regimen_path=path)
    assert regimen is not None and path.exists()
    broker = BrokerMemoria()
    correr_diario(as_of, data_provider=data, db=db_session, broker=broker,
                  regimen_path=path, estrategias=(("arquetipo-a-pullback", SiempreSenala()),))
    assert not any(r.__tablename__ == "candidatos_semanales" for r in db_session.query(TradeIntentRecord).all())
    entradas = [o for o in broker.orders if o.side == OrderSide.BUY]
    assert entradas
    assert all(o.symbol in PRIORS_M5["universo"] for o in entradas)
    assert all(o.order_type == OrderType.STOP and o.stop_price is not None for o in entradas)
    for entrada in entradas:
        assert any(o.symbol == entrada.symbol and o.side == OrderSide.SELL
                   and o.order_type == OrderType.STOP and o.stop_price is not None for o in broker.orders)


def test_pipeline_rechaza_senal_sin_stop_antes_del_broker(db_session, mercado, tmp_path):
    as_of, data = mercado
    path = tmp_path / "regimen.csv"
    correr_semanal(as_of, data_provider=data, regimen_path=path)
    broker = BrokerMemoria()
    with pytest.raises(ValidationError):
        correr_diario(as_of, data_provider=data, db=db_session, broker=broker,
                      regimen_path=path, estrategias=(("sin-stop", SinStop()),))
    assert broker.orders == []


def test_pipeline_diario_idempotente(db_session, mercado, tmp_path):
    as_of, data = mercado
    path = tmp_path / "regimen.csv"
    correr_semanal(as_of, data_provider=data, regimen_path=path)
    broker = BrokerMemoria()
    args = dict(data_provider=data, db=db_session, broker=broker, regimen_path=path,
                estrategias=(("test", SiempreSenala()),))
    correr_diario(as_of, **args)
    first = (db_session.query(TradeIntentRecord).count(), len(broker.orders))
    correr_diario(as_of, **args)
    assert (db_session.query(TradeIntentRecord).count(), len(broker.orders)) == first


def test_pipeline_persiste_rechazo_heat(db_session, mercado, tmp_path):
    as_of, data = mercado
    path = tmp_path / "regimen.csv"
    correr_semanal(as_of, data_provider=data, regimen_path=path)
    broker = BrokerMemoria()
    portfolio = PortfolioManager(3000, db=db_session)
    portfolio.positions["SPY"] = {"qty": 100, "avg_cost": 20, "current_price": 20,
                                   "stop_price": 19, "pnl": 0}
    correr_diario(as_of, data_provider=data, db=db_session, broker=broker,
                  portfolio_manager=portfolio, regimen_path=path,
                  estrategias=(("heat", SiempreSenala(stop=1)),))
    rechazados = db_session.query(TradeIntentRecord).filter(
        TradeIntentRecord.reject_reason == "max_heat").all()
    assert rechazados
    assert not broker.orders
