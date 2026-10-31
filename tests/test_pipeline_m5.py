import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from quantagent.data.snapshot import write_snapshot
from quantagent.fabrica.candidatos_vivo import candidatos_en_vivo
from quantagent.fabrica.cartera import pesos, pesos_h3
from quantagent.fabrica.regimen import regimen_semanal
from quantagent.fabrica.transversal import MASCARAS, METRICAS, PRIORS_M5, REGLAS, señales_transversales
from quantagent.models import OrderSide, OrderType, TradeIntentRecord
from quantagent.portfolio.manager import PortfolioManager
from quantagent.strategy.base import TradingSignal, TradingStrategy
from quantagent.trading.cartera_paper import sesiones_cerradas_nyse
from quantagent.trading.pipeline_m5 import correr_diario, correr_semanal, fundamentales_vigente
from tests.test_candidatos_vivo import FUND
from tests.test_candidatos_vivo import datos as datos_screener  # noqa: F401


@pytest.fixture(autouse=True)
def sin_llm_externo(monkeypatch):
    monkeypatch.setattr("quantagent.trading.pipeline_m5.llm_por_defecto", lambda: None)


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


@pytest.fixture
def semana(mercado, datos_screener):  # noqa: F811 - fixture compartida con el screener
    as_of, data = mercado
    base, precios = datos_screener
    for simbolo, serie in precios.items():
        if simbolo not in data.panel:
            data.panel[simbolo] = pd.DataFrame({"timestamp": serie.index, "close": serie.to_numpy()})
    # XLK lidera sin empates y los demás ETFs tienen retornos distintos.
    for i, simbolo in enumerate(PRIORS_M5["universo"]):
        df = data.panel[simbolo]
        t = np.arange(len(df))
        df["close"] = 100 * np.exp(t * (0.004 if simbolo == "XLK" else 0.0001 * (i + 1))) + np.sin(t / 8)
    shutil.copytree(base / FUND, base / "fundamentales-sp500-2026-10-29")
    return as_of, data, base


def test_semanal_candidatos_equivalencia_y_upsert(semana):
    as_of, data, base = semana
    fecha, siguiente = pd.Timestamp("2026-10-30"), pd.Timestamp("2026-11-02")
    cierres = pd.DataFrame({s: data.panel[s].set_index("timestamp")["close"] for s in PRIORS_M5["universo"]})
    esperado = regimen_semanal(cierres, PRIORS_M5, peso_macro=0.0)
    esperado = esperado[esperado.fecha == fecha]
    assert "XLK" in esperado.iloc[0].habilitados.split(",")
    cand, _ = candidatos_en_vivo(as_of, esperado.iloc[0].habilitados.split(","), data,
                               fundamentales_vigente(base), base_dir=base)
    assert set(cand.simbolo) & {"AAA", "BBB", "CCC", "DDD"}
    # Mismas piezas y fecha de rebalanceo que scripts/m5/h3.py, con precios arbitrarios
    # de la sesión siguiente: sus valores no deben intervenir en los pesos.
    extendido = cierres.reindex(cierres.index.append(pd.DatetimeIndex([siguiente]))).fillna(10000)
    seleccion = señales_transversales(extendido, METRICAS["sector-score-h1"],
                                     mascara=MASCARAS["sin-caida-m5"], **REGLAS["sector-score-h1"])
    h1 = pesos(extendido, {0: list(PRIORS_M5["universo"])}, seleccion={0: seleccion}, semanales={0})
    h1.columns = h1.columns.droplevel(0)
    h3 = pesos_h3(h1, cand, {siguiente: fecha})
    args = dict(data_provider=data, regimen_path=base / "regimen.csv", base_dir=base, narrador=None)
    correr_semanal(as_of, **args)
    assert (base / "regimen.csv").read_text() == esperado.to_csv(index=False)
    pd.testing.assert_frame_equal(pd.read_csv(base / "m5-candidatos-semanales.csv", parse_dates=["fecha"]),
                                  cand.reset_index(drop=True))
    sombra = pd.read_csv(base / "m5-sombra-semanal.csv").set_index("activo")
    for columna, pesos_esperados in (("peso_h1", h1), ("peso_h3", h3)):
        objetivo = pesos_esperados.loc[siguiente].reindex(sombra.index, fill_value=0.0)
        np.testing.assert_allclose(sombra[columna], objetivo, rtol=1e-12, atol=1e-14)
    originales = {p: p.read_bytes() for p in base.glob("*.csv")}
    correr_semanal(as_of, **args)
    assert {p: p.read_bytes() for p in originales} == originales
    correr_semanal(as_of - pd.Timedelta(days=7), **args)
    archivos = list(base.glob("*.csv"))
    originales = {p: p.read_bytes() for p in archivos}
    correr_semanal(as_of, **args)
    assert {p: p.read_bytes() for p in archivos} == originales
    for nombre, claves in (("regimen.csv", ["fecha"]),
                           ("m5-candidatos-semanales.csv", ["fecha", "etf", "simbolo"]),
                           ("m5-sombra-semanal.csv", ["fecha", "activo"])):
        assert not pd.read_csv(base / nombre).duplicated(claves).any()
    # El upsert vacío debe quitar las acciones actuales y conservar la otra semana.
    previos = pd.read_csv(base / "m5-candidatos-semanales.csv", parse_dates=["fecha"])
    previos = previos[previos.fecha != fecha].reset_index(drop=True)
    shutil.rmtree(base / "fundamentales-sp500-2026-10-29")
    correr_semanal(as_of, **args)
    actuales = pd.read_csv(base / "m5-candidatos-semanales.csv", parse_dates=["fecha"])
    pd.testing.assert_frame_equal(actuales, previos)
    sombra = pd.read_csv(base / "m5-sombra-semanal.csv")
    sombra = sombra[sombra.fecha == str(fecha.date())]
    np.testing.assert_array_equal(sombra.peso_h1, sombra.peso_h3)


@pytest.mark.parametrize("caso", ["sin_fundamentales", "vacio", "falla", "risk-off"])
def test_semanal_degradacion_y_efectivo(semana, monkeypatch, caso):
    as_of, data, base = semana
    if caso == "sin_fundamentales":
        shutil.rmtree(base / "fundamentales-sp500-2026-10-29")
    elif caso == "vacio":
        for t in ("AAA", "BBB", "CCC", "DDD"):
            data.panel[t] = data.panel[t].iloc[:0]
        # Sólo XLK habilitado: el screener no puede devolver ETFs de país.
        for t in PRIORS_M5["universo"]:
            if t != "XLK":
                pasos = np.arange(len(data.panel[t]))
                data.panel[t]["close"] = 100 + 0.001 * pasos + 0.01 * np.sin(pasos / 8)
    elif caso == "falla":
        def fallar(*args, **kwargs):
            raise RuntimeError("proveedor no disponible")
        monkeypatch.setattr("quantagent.trading.pipeline_m5.candidatos_en_vivo", fallar)
    else:
        for t in PRIORS_M5["universo"]:
            data.panel[t]["close"] = np.linspace(200, 50, len(data.panel[t]))
    args = dict(data_provider=data, regimen_path=base / "regimen.csv", base_dir=base, narrador=None)
    resultado = correr_semanal(as_of, **args)
    cierres = pd.DataFrame({s: data.panel[s].set_index("timestamp")["close"] for s in PRIORS_M5["universo"]})
    esperado = regimen_semanal(cierres, PRIORS_M5, peso_macro=0.0).tail(1)
    assert (base / "regimen.csv").read_text() == esperado.to_csv(index=False)
    assert pd.read_csv(base / "m5-candidatos-semanales.csv").empty
    sombra = pd.read_csv(base / "m5-sombra-semanal.csv")
    np.testing.assert_array_equal(sombra.peso_h1, sombra.peso_h3)
    if caso == "risk-off":
        assert resultado["estado"] == "risk-off" and sombra.peso_h1.sum() == 0
    else:
        assert sombra.peso_h1.sum() == pytest.approx(1)
    descartes = pd.read_csv(base / "m5-candidatos-descartes.csv")
    if caso in ("sin_fundamentales", "falla"):
        assert len(descartes) == 1 and pd.isna(descartes.iloc[0].simbolo)
        assert ("fundamentales" if caso == "sin_fundamentales" else "proveedor no disponible") in descartes.iloc[0].motivo
    assert pd.read_csv(base / "m5-narrativa.csv").iloc[0].tipo == "nota_semanal"
    originales = {p: p.read_bytes() for p in base.glob("*.csv")}
    correr_semanal(as_of, **args)
    assert {p: p.read_bytes() for p in originales} == originales


def test_fundamentales_vigente_verifica_y_respalda(tmp_path):
    assert fundamentales_vigente(tmp_path) == "fundamentales-sp500-2026-10"
    frames = {"AAA": pd.DataFrame({"val": [1]}, index=pd.to_datetime(["2026-10-01"]))}
    for dia in ("28", "29", "30"):
        write_snapshot(f"fundamentales-sp500-2026-10-{dia}", frames, base_dir=tmp_path)
    (tmp_path / "fundamentales-sp500-2026-10-30" / "AAA.parquet").write_bytes(b"corrupto")
    assert fundamentales_vigente(tmp_path) == "fundamentales-sp500-2026-10-29"
    (tmp_path / "fundamentales-sp500-2026-10-29" / "AAA.parquet").unlink()
    assert fundamentales_vigente(tmp_path) == "fundamentales-sp500-2026-10-28"
