"""Jobs semanal y diario de M5.

El régimen se conserva en ``local-data/m5-regimen-semanal.csv`` (una fila por
sesión de decisión), separado de la DB ``local-data/m5.db`` de intents y órdenes.
Todas las fechas de decisión salen de ``as_of`` y la sesión NYSE cerrada.
"""

import logging
from decimal import Decimal
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from quantagent.fabrica.regimen import regimen_semanal
from quantagent.fabrica.transversal import PRIORS_M5
from quantagent.models import ActivePosition, Environment, Order, OrderSide, OrderType, TradeIntentRecord
from quantagent.portfolio.manager import PortfolioManager
from quantagent.strategy.arquetipo_a_strategy import ArquetipoAStrategy
from quantagent.strategy.arquetipo_b_strategy import ArquetipoBStrategy
from quantagent.strategy.indicadores import atr
from quantagent.strategy.trade_intent import TradeIntent
from quantagent.trading.bucket_m5 import CAPITAL_INICIAL, crear_position_sizer, crear_risk_manager
from quantagent.trading.cartera_paper import sesiones_cerradas_nyse, ultima_sesion_cerrada_nyse
from quantagent.trading.intent_persistence import save_trade_intent
from quantagent.trading.narrativa_m5 import explicar_intent, guardar_narrativa, llm_por_defecto, nota_semanal
from quantagent.trading.position_monitor import PositionMonitor
from quantagent.trading.stop_management import propuesta_breakeven, propuesta_chandelier

logger = logging.getLogger(__name__)
UNIVERSO_M5 = tuple(PRIORS_M5["universo"])


def _panel_cierres(proveedor, sesiones):
    inicio = sesiones[0].to_pydatetime()
    fin = (sesiones[-1] + pd.Timedelta(days=1)).to_pydatetime()
    columnas = {}
    for simbolo in UNIVERSO_M5:
        df = proveedor.get_ohlc(simbolo, "1d", inicio, fin)
        if df.empty:
            continue
        fechas = pd.to_datetime(df["timestamp"]).dt.normalize()
        series = pd.Series(df["close"].to_numpy(), index=fechas)
        columnas[simbolo] = series[~series.index.duplicated(keep="last")]
    return pd.DataFrame(columnas).reindex(index=sesiones, columns=UNIVERSO_M5)


def _resolver_narrador(narrador):
    return llm_por_defecto() if narrador == "auto" else narrador


def _narrativa_path(narrativa_path, regimen_path):
    return narrativa_path or Path(regimen_path).parent / "m5-narrativa.csv"


def _narrar_intent(intent, narrador, narrativa_path, regimen_path, fecha, simbolo, strategy_id):
    """Texto del LLM sobre un intent ya guardado; nunca altera ni propaga errores al pipeline."""
    try:
        texto = explicar_intent(intent, _resolver_narrador(narrador))
        guardar_narrativa(_narrativa_path(narrativa_path, regimen_path), "intent",
                          f"{fecha.date()}:{simbolo}:{strategy_id}", texto)
    except Exception:
        logger.warning("Narrativa de intent M5 no guardada", exc_info=True)


def correr_semanal(as_of, *, data_provider, db: Session | None = None,
                   regimen_path="local-data/m5-regimen-semanal.csv", narrador="auto",
                   narrativa_path=None):
    """Calcula H1 al cierre semanal y hace upsert CSV por fecha, sin depender del reloj del sistema."""
    ultima = ultima_sesion_cerrada_nyse(as_of)
    sesiones = sesiones_cerradas_nyse(as_of, n_sesiones=504)
    cierres = _panel_cierres(data_provider, sesiones).dropna(how="all").ffill()
    tabla = regimen_semanal(cierres, PRIORS_M5, peso_macro=0.0)
    fila = tabla.loc[tabla["fecha"].dt.normalize() == ultima.normalize()].tail(1)
    if fila.empty:
        return None
    path = Path(regimen_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anterior = None
    if path.exists():
        previo = pd.read_csv(path, parse_dates=["fecha"])
        previas = previo[previo["fecha"].dt.normalize() < ultima.normalize()]
        if not previas.empty:
            anterior = previas.sort_values("fecha").iloc[-1]
        tabla = pd.concat([previo[previo["fecha"].dt.normalize() != ultima.normalize()], fila])
    else:
        tabla = fila
    tabla.sort_values("fecha").to_csv(path, index=False)
    try:
        llm = _resolver_narrador(narrador)
        texto = nota_semanal(fila.iloc[0], anterior, llm)
        guardar_narrativa(_narrativa_path(narrativa_path, path), "nota_semanal",
                          pd.Timestamp(fila.iloc[0]["fecha"]).date().isoformat(), texto)
    except Exception:
        logger.warning("Narrativa semanal M5 no guardada", exc_info=True)
    return fila.iloc[0].to_dict()


def _regimen_actual(path, fecha):
    tabla = pd.read_csv(path, parse_dates=["fecha"])
    filas = tabla[tabla["fecha"].dt.normalize() <= fecha.normalize()]
    return None if filas.empty else filas.sort_values("fecha").iloc[-1]


def _seguir_stops(db, broker, data_provider, panel, fecha):
    monitor = PositionMonitor(db, environment=Environment.PAPER)
    for position in db.query(ActivePosition).filter(
            ActivePosition.is_active.is_(True), ActivePosition.quantity > 0,
            ActivePosition.environment == Environment.PAPER).all():
        if position.symbol not in panel:
            continue
        bars = panel[position.symbol].dropna()
        bars = bars[bars.index >= pd.Timestamp(position.decision_timestamp).normalize()]
        if bars.empty:
            continue
        inicio = pd.Timestamp(position.decision_timestamp).to_pydatetime()
        fin = (fecha + pd.Timedelta(days=1)).to_pydatetime()
        candles = data_provider.get_ohlc(position.symbol, "1d", inicio, fin)
        if len(candles) < 14:
            continue
        candles = candles[pd.to_datetime(candles["timestamp"]).dt.normalize() >= bars.index[0]]
        atr_value = float(atr(candles).iloc[-1])
        if pd.isna(atr_value) or atr_value <= 0:
            continue
        extremo = float(candles["high"].max() if position.side == OrderSide.BUY else candles["low"].min())
        proposed = propuesta_chandelier(extremo, atr_value, position.side)
        breakeven = propuesta_breakeven(float(position.entry_price), float(position.stop_loss),
                                         float(bars.iloc[-1]), position.side)
        if breakeven is not None:
            proposed = max(proposed, breakeven) if position.side == OrderSide.BUY else min(proposed, breakeven)
        stop_anterior = position.stop_loss
        decision = monitor.actualizar_stop(position, proposed)
        if decision.stop == stop_anterior or not position.broker_stop_order_id:
            continue
        stop_order = db.query(Order).filter(Order.alpaca_order_id == position.broker_stop_order_id).first()
        if stop_order is not None and hasattr(broker, "replace_stop"):
            broker.replace_stop(position.broker_stop_order_id, decision.stop)
            stop_order.stop_price = decision.stop
            db.commit()


def correr_diario(as_of, *, data_provider, db: Session, broker,
                  portfolio_manager=None, regimen_path="local-data/m5-regimen-semanal.csv",
                  estrategias=None, narrador="auto", narrativa_path=None):
    """Evalúa arquetipos sobre ETFs habilitados; crea intents idempotentes y órdenes stop protegidas."""
    fecha = ultima_sesion_cerrada_nyse(as_of)
    timestamp = pd.Timestamp(as_of)
    timestamp = timestamp.tz_convert("UTC").tz_localize(None) if timestamp.tzinfo else timestamp
    ses = sesiones_cerradas_nyse(as_of, n_sesiones=504)
    regimen = _regimen_actual(regimen_path, fecha)
    if regimen is None or regimen["estado"] == "risk-off":
        return []
    habilitados = [s for s in str(regimen["habilitados"]).split(",") if s in UNIVERSO_M5]
    panel = _panel_cierres(data_provider, ses)
    _seguir_stops(db, broker, data_provider, panel, fecha)
    manager = portfolio_manager or PortfolioManager(CAPITAL_INICIAL, db=db, clock=lambda: fecha.date())
    manager.set_date(fecha.date())
    risk = crear_risk_manager(manager, db=db, clock=lambda: fecha.date())
    sizer = crear_position_sizer()
    strategies = estrategias or (("arquetipo-a-pullback", ArquetipoAStrategy()),
                                 ("arquetipo-b-breakout", ArquetipoBStrategy()))
    creados = []
    for simbolo in habilitados:
        serie = panel[simbolo].dropna()
        if serie.empty:
            continue
        candles = data_provider.get_ohlc(simbolo, "1d", serie.index[0].to_pydatetime(),
                                         (fecha + pd.Timedelta(days=1)).to_pydatetime())
        candles = candles[pd.to_datetime(candles["timestamp"]).dt.normalize().isin(serie.index)]
        records = candles.to_dict("records")
        if not records:
            continue
        precio = float(records[-1]["close"])
        atr_value = float(atr(pd.DataFrame(records)).iloc[-1])
        for strategy_id, strategy in strategies:
            signal = strategy.generate_signal(records, simbolo, "1d", precio)
            if signal is None or signal.decision.upper() != "LONG":
                continue
            razonamiento = dict(session_date=str(fecha.date()), source="pipeline_m5")
            intent = TradeIntent(
                decision=signal.decision, confidence=signal.confidence,
                entry_price=signal.entry_price or precio, stop_loss=signal.stop_loss,
                take_profit=signal.take_profit, reasoning=signal.reasoning,
                exit_policy=signal.exit_policy, entry_order=signal.entry_order,
                entry_expiry_bars=signal.entry_expiry_bars,
                bucket_id="m5", strategy_id=strategy_id, symbol=simbolo,
                order_type="stop" if signal.entry_order == "stop" else "market",
                atr_at_signal=atr_value, reasoning_metadata=razonamiento,
            )
            duplicado = next((r for r in db.query(TradeIntentRecord).filter(
                TradeIntentRecord.bucket_id == "m5", TradeIntentRecord.symbol == simbolo,
                TradeIntentRecord.strategy_id == strategy_id).all()
                if (r.reasoning_metadata or {}).get("session_date") == str(fecha.date())), None)
            if duplicado:
                continue
            qty = sizer.calculate_size(simbolo, intent.confidence, float(intent.entry_price),
                                       manager.get_total_value(), stop_price=float(intent.stop_loss))
            valido, motivo = risk.validate_trade(simbolo, OrderSide.BUY, qty,
                                                  float(intent.entry_price), float(intent.stop_loss))
            if qty <= 0:
                motivo = "tamano_cero"
                valido = False
            if not valido:
                record = save_trade_intent(db, intent, "rechazado_riesgo", motivo.split(":")[0])
                record.created_at = record.updated_at = timestamp.to_pydatetime()
                db.commit()
                creados.append(record)
                _narrar_intent(intent, narrador, narrativa_path, regimen_path, fecha, simbolo, strategy_id)
                continue
            entrada = Order(symbol=simbolo, side=OrderSide.BUY, quantity=Decimal(str(qty)),
                            order_type=OrderType.STOP if intent.order_type == "stop" else OrderType.MARKET,
                            price=Decimal(str(intent.entry_price)) if intent.order_type == "market" else None,
                            stop_price=Decimal(str(intent.entry_price)) if intent.order_type == "stop" else None,
                            environment=Environment.PAPER, created_at=timestamp.to_pydatetime(),
                            comment=f"m5; vence={intent.entry_expires_bars} barras")
            db.add(entrada)
            db.flush()
            broker.submit_order(entrada)
            db.add(entrada)
            proteccion = Order(symbol=simbolo, side=OrderSide.SELL, quantity=Decimal(str(qty)),
                               order_type=OrderType.STOP, stop_price=Decimal(str(intent.stop_loss)),
                               environment=Environment.PAPER, created_at=timestamp.to_pydatetime(),
                               comment=f"proteccion intent={simbolo}:{strategy_id}")
            db.add(proteccion)
            db.flush()
            broker.submit_order(proteccion)
            record = save_trade_intent(db, intent, "aprobado", order_id=entrada.id)
            record.created_at = record.updated_at = timestamp.to_pydatetime()
            db.commit()
            creados.append(record)
            _narrar_intent(intent, narrador, narrativa_path, regimen_path, fecha, simbolo, strategy_id)
    return creados
