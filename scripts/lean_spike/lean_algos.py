"""QuantAgent-ryaw (spike): port de sma-cross y rsi a QCAlgorithm. Corre dentro de LEAN (Python 3.11).

NO EJECUTADO todavia: escrito contra la API de LEAN sin tener el motor a mano.

Replica lo que hace scripts/bt_adapter.py --fill next-open con las estrategias del proyecto:
- la senal se evalua con el cierre de cada vela diaria y la orden de mercado llena en la apertura siguiente;
- tamano = equity * fraccion * confianza / cierre de la senal, en acciones enteras (LEAN no opera
  fracciones de equity; el control si);
- costos por lado: slippage como fraccion del precio y comision como fraccion del nocional. Los defaults
  son el perfil `etf` de COSTOS_PERFILES (quantagent/settings.py): 0.0005 y 0.

Parametros (seccion "parameters" del config.json): cuenta (cash|margin), slippage, comision, fraccion.
Los tags de las ordenes llevan lo que closedTrades no trae: `stop=<nivel>` en la entrada y el motivo en la
salida. closed_trades_to_csv.py los lee.
"""

from collections import deque

from AlgorithmImports import *


class ComisionPorcentual(FeeModel):
    """Comision como fraccion del nocional, por lado."""

    def __init__(self, pct):
        super().__init__()
        self.pct = pct

    def get_order_fee(self, parameters):
        valor = abs(parameters.order.get_value(parameters.security))
        return OrderFee(CashAmount(valor * self.pct, "USD"))


class SpikeBase(QCAlgorithm):
    CUENTA = "cash"
    HISTORIA = 30  # velas minimas antes de evaluar (required_history_bars de la estrategia)

    def initialize(self):
        self.set_start_date(2007, 1, 3)
        self.set_end_date(2018, 12, 31)  # tramo de desarrollo: nada despues
        self.set_cash(100000)
        cuenta = self.get_parameter("cuenta", self.CUENTA)
        self.slippage = float(self.get_parameter("slippage", 0.0005))
        self.comision = float(self.get_parameter("comision", 0.0))
        self.fraccion = float(self.get_parameter("fraccion", 1.0))
        self.set_brokerage_model(
            BrokerageName.ALPACA, AccountType.CASH if cuenta == "cash" else AccountType.MARGIN)
        equity = self.add_equity("SPY", Resolution.DAILY, data_normalization_mode=DataNormalizationMode.RAW)
        equity.set_fee_model(ComisionPorcentual(self.comision))
        equity.set_slippage_model(ConstantSlippageModel(self.slippage))
        if cuenta != "cash":
            equity.set_leverage(1)
        self.spy = equity.symbol
        self.set_benchmark(lambda _: 0)  # sin benchmark: no hay mas datos que SPY
        self.cierres = deque(maxlen=self.HISTORIA)
        self.velas = 0
        self.pos = None  # posicion abierta o con entrada pendiente

    def on_data(self, data):
        bar = data.bars.get(self.spy)
        if bar is None:
            return
        self.velas += 1
        self.cierres.append(float(bar.close))
        if self.velas < self.HISTORIA:
            return
        cierre = float(bar.close)
        if self.pos is not None:
            motivo = self.motivo_de_salida(cierre)
            if motivo:
                self.cerrar(motivo)
            else:
                self.pos["velas"] += 1
        if self.pos is None:
            self.evaluar_entrada(cierre)

    def cerrar(self, motivo):
        self.transactions.cancel_open_orders(self.spy)
        self.liquidate(self.spy, tag=motivo)
        self.pos = None

    def entrar(self, largo, confianza, cierre, stop, take):
        equity = float(self.portfolio.total_portfolio_value)
        cantidad = int(equity * self.fraccion * confianza / cierre)
        if largo:  # como OrderManager._cap_entry_qty: lo que alcanza a pagar con slippage y comision
            cantidad = min(cantidad, int(equity / (cierre * (1 + self.slippage) * (1 + self.comision))))
        if cantidad < 1:
            return
        ticket = self.market_order(self.spy, cantidad if largo else -cantidad, tag=f"stop={stop:.8f}")
        self.pos = {"largo": largo, "cantidad": cantidad, "stop": stop, "take": take, "velas": 0,
                    "extremo": None, "entrada": ticket.order_id}


class SmaCrossSpike(SpikeBase):
    """quantagent/strategy/sma_cross_strategy.py: cruce 50/200, solo largos, sin stop ni take profit."""

    CUENTA = "cash"
    FAST, SLOW = 50, 200
    HISTORIA = SLOW + 1

    def medias(self, atras=0):
        cierres = list(self.cierres)
        cierres = cierres[:len(cierres) - atras]
        return sum(cierres[-self.FAST:]) / self.FAST, sum(cierres[-self.SLOW:]) / self.SLOW

    def motivo_de_salida(self, cierre):
        rapida, lenta = self.medias()
        return "SMA_CROSS_DOWN" if rapida < lenta else None

    def evaluar_entrada(self, cierre):
        rapida, lenta = self.medias()
        rapida_ayer, lenta_ayer = self.medias(atras=1)
        if rapida_ayer <= lenta_ayer and rapida > lenta:
            self.entrar(True, 1.0, cierre, 0.01, 1_000_000_000.0)

    def on_order_event(self, event):
        if self.pos and event.order_id == self.pos["entrada"] and event.status == OrderStatus.INVALID:
            self.pos = None


class RsiSpike(SpikeBase):
    """quantagent/strategy/rsi_strategy.py: RSI 14 de medias simples, largo < 30 y corto > 70, stop 2 %, take 3 %.

    El stop y el take profit son ordenes de LEAN (stop de mercado y limite) que se mandan cuando llena la
    entrada, con el modelo de llenado por defecto. La salida por regla (TradingStrategy.should_exit: stop,
    take profit y trailing 5 % contra el cierre) va a mercado en la apertura siguiente.
    """

    CUENTA = "margin"  # 105 de los 187 trades del control son cortos
    PERIODO, SOBREVENTA, SOBRECOMPRA = 14, 30.0, 70.0
    STOP, TAKE, TRAILING = 0.02, 0.03, 0.05

    def rsi(self):
        cierres = list(self.cierres)[-(self.PERIODO + 1):]
        difs = [b - a for a, b in zip(cierres, cierres[1:])]
        ganancia = sum(d for d in difs if d > 0) / self.PERIODO
        perdida = sum(-d for d in difs if d < 0) / self.PERIODO
        return 100 - 100 / (1 + ganancia / (perdida or 1e-10))

    def motivo_de_salida(self, cierre):
        p = self.pos
        signo = 1 if p["largo"] else -1
        if signo * (cierre - p["stop"]) <= 0:
            return "STOP_LOSS"
        if signo * (cierre - p["take"]) >= 0:
            return "TAKE_PROFIT"
        if p["extremo"] is None or signo * (cierre - p["extremo"]) > 0:
            p["extremo"] = cierre
        if signo * (cierre - p["extremo"] * (1 - signo * self.TRAILING)) < 0:
            return "TRAILING_STOP"
        return None

    def evaluar_entrada(self, cierre):
        rsi = self.rsi()
        if rsi < self.SOBREVENTA:
            self.entrar(True, 1.0 - rsi / self.SOBREVENTA, cierre,
                        round(cierre * (1 - self.STOP), 8), round(cierre * (1 + self.TAKE), 8))
        elif rsi > self.SOBRECOMPRA:
            self.entrar(False, (rsi - self.SOBRECOMPRA) / (100 - self.SOBRECOMPRA), cierre,
                        round(cierre * (1 + self.STOP), 8), round(cierre * (1 - self.TAKE), 8))

    def on_order_event(self, event):
        p = self.pos
        if p is None:
            return
        if event.order_id == p["entrada"]:
            if event.status == OrderStatus.INVALID:
                self.pos = None
            elif event.status == OrderStatus.FILLED:
                salida = -p["cantidad"] if p["largo"] else p["cantidad"]
                # el stop primero: si la vela toca los dos niveles, el control ejecuta el stop
                p["ordenes"] = [
                    self.stop_market_order(self.spy, salida, p["stop"], tag="STOP_LOSS").order_id,
                    self.limit_order(self.spy, salida, p["take"], tag="TAKE_PROFIT").order_id,
                ]
        elif event.order_id in p.get("ordenes", []) and event.status == OrderStatus.FILLED:
            self.transactions.cancel_open_orders(self.spy)
            self.pos = None
