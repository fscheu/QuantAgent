#!/usr/bin/env python
"""QuantAgent-832.1: la estrategia RSI del proyecto portada a backtesting.py (oraculo de cross-validation).

Uso:  python scripts/crossval_rsi.py [--out crossval_trades.csv]

Corre RSI mean-reversion sobre tests/fixtures/spy-90d.csv con backtesting.py (dependencia de
desarrollo, nunca de runtime: nada bajo quantagent/ la importa) y escribe un CSV con las mismas 10
columnas, en el mismo orden, que `python -m quantagent.cli backtest run ... --out` (COLUMNS de
quantagent/backtesting/export.py). Este script NO compara contra el motor del proyecto.

Antes de simular imprime una tabla de parametros proyecto vs port. La columna del proyecto se LEE en
runtime (RSIMeanReversionStrategy, quantagent.settings, default de Backtest, PaperBroker); la del port
son las constantes PORT_PARAMS que usa la Strategy. Si difieren el script sale con codigo 1. El
regimen de cross-validation es sin slippage: el script fuerza TRADING_SLIPPAGE_PCT=0 antes de importar
settings, y para comparar hay que correr el motor igual: `TRADING_SLIPPAGE_PCT=0 python -m quantagent.cli ...`.
(Por eso la fila slippage solo prueba que el motor corrido asi toma 0; el default del proyecto es 0.0005.)

Como se alinea con el motor (leido en quantagent/backtesting/backtest.py::_analyze_and_trade):
- Senal con el close de la vela; orden rellenada a ese mismo close (trade_on_close=True; las ordenes
  de apertura/cierre son de mercado "opuestas", no Trade.close(), porque esas rellenan en el OPEN de
  la vela siguiente). Sin slippage ni comision. No se usan sl=/tp= de backtesting.py (miran high/low);
  como el motor, SL/TP/trailing se evaluan solo contra el close, en el orden SL, TP, trailing.
- Con posicion abierta no se mira la senal (una senal opuesta se ignora); al cerrar por regla se evalua
  la senal de esa misma vela. Hace falta historia >= required_history_bars (30) velas.
- RSI de RSIMeanReversionStrategy._calculate_rsi: media simple de 14 (no Wilder), perdida 0 -> 1e-10.
- Tamano: equity * base_position_pct * confianza / precio, confianza = 1 - rsi/30 (long) o
  (rsi-70)/30 (short). backtesting.py solo opera unidades enteras: los precios se dividen por 2**27
  (exacto en float) y se operan unidades = qty * 2**27; qty y precios del CSV se vuelven a escalar.
- Cierre forzado al final al close de la ultima vela: se agrega una vela centinela plana (solo para
  que el broker procese las ordenes de la ultima vela real; ningun trade usa su timestamp).
- stop_loss/take_profit se redondean a 8 decimales (el motor los lee de Numeric(18,8)).

COLUMNAS SIN EQUIVALENTE EN backtesting.py (quedan vacias en el CSV):
- stop_loss: el motor lo toma de ActivePosition.stop_loss; aca el stop lo evalua el codigo de la
  Strategy y backtesting.py no lo conoce (no se usa sl=), asi que el stats._trades no lo trae.
- exit_reason: backtesting.py no registra el motivo de cierre (STOP_LOSS/TAKE_PROFIT/TRAILING_STOP/
  backtest_end), solo precio y hora.

DIFERENCIAS DE SEMÁNTICA CONOCIDAS (motor del proyecto vs port; ninguna se corrige en el motor):
1. Cantidad: el motor opera qty float (guardada con 8 decimales, Numeric(18,8)); backtesting.py solo
   unidades enteras, asi que el port la cuantiza a 2**-27 (~7.5e-9) por redondeo. Efecto: ruido ~1e-8
   relativo en qty y pnl; sin sesgo.
2. Rechazos de riesgo: el motor valida cada orden en trading/risk_manager.py::validate_trade; el port no
   valida nada. CONFIRMADO que no hay rechazos en esta corrida: scripts/crossval_compare.py corre el
   mismo CLI envolviendo validate_trade con un contador (los rechazos no se persisten ni se loguean en el
   CLI): 454 llamadas (2 por trade), 0 rechazos, y el CSV de trades es identico al de la corrida normal.
3. Perdida diaria del motor: RiskManager.get_daily_pnl usa date.today() del reloj real. CONFIRMADO que
   no dispara aca: el termino realizado (Trade.closed_at >= hoy 00:00) es 0 porque el fixture es de
   ene-mar 2026 y hoy es posterior; solo cuenta el no realizado de la posicion abierta (<=5 % del
   portfolio con SL 2 %). Margen minimo al limite: 4918.53 USD sobre ~5000; breaker activo: 0 veces.
4. Valor de portfolio para el tamano: motor = PortfolioManager.get_total_value(); port = Strategy.equity.
   CONFIRMADO: qty (que depende de ese valor) coincide en los 227 trades (max |dif| 8.5e-9) y la
   equity en las 2160 velas (max |dif| 5.8e-5, redondeo a 4 decimales del CSV).
5. Redondeo de DB: el motor persiste en Numeric(18,8) y compara con el valor releido; el port redondea
   SL/TP a 8 decimales pero no qty/pnl. SIN CONFIRMAR en general (solo se corrio SQLite; Postgres podria
   redondear distinto en empates exactos): en SQLite no hubo efecto, los 227 exit_time coinciden.
6. PnL de salida: el motor lo calcula en Decimal con qty a 8 decimales y lo copia de la orden de cierre
   (_sync_linked_trade_exit); el port usa el PnL de backtesting.py (size * (exit - entry) en float).
   Ruido ~1e-8; sin sesgo.
7. Ultima vela: el motor evalua la senal tambien en la ultima vela y despues cierra todo con
   'backtest_end' al ultimo close (_close_remaining_positions); el port lo reproduce con la vela
   centinela. CONFIRMADO el cierre forzado (el trade 227 sale 'backtest_end' en la ultima vela con el
   mismo exit_time y precio en ambos). SIN CONFIRMAR la apertura en la ultima vela: el fixture no la
   ejercita (ultima entrada 2026-03-31T17:00) y probarla exigiria otro fixture.
8. Ventana de datos: el motor pasa a la estrategia las velas de los ultimos 7 dias (ventana por
   calendario), el port el historial completo; con RSI de 14 periodos el valor es el mismo salvo que la
   ventana tuviera huecos (el fixture es continuo 24/7).
9. Hora/precio de fill: backtesting.py rellena las ordenes de mercado (trade_on_close) al close de la
   vela previa a su procesamiento y les pone esa hora; tests/test_crossval_rsi.py verifica que
   entry_time/exit_time son velas del fixture y que los precios son el close de esa vela. El motor usa
   timestamp=current_date. Es comportamiento interno de la libreria (acotada a >=0.6.6,<0.7).
10. Orden del CSV: el motor ordena por Trade.opened_at; el port por entry_time. Sin empates posibles
   (una posicion a la vez).
11. Capital inicial: el CLI del motor no pasa initial_capital, rige el default de Backtest.__init__
   (100000.0), no settings.TRADING_INITIAL_CASH (que solo usa StrategyAssembler.DEFAULTS); la tabla lee
   el default de Backtest. Hoy ambos valen 100000.0; si divergieran, la tabla seguiria al CLI.
12. Trailing stop: igual que TradingStrategy._check_trailing_stop (extremo = primer close evaluado).
   CONFIRMADO que no dispara: 0 salidas TRAILING_STOP en el motor (113 SL, 113 TP, 1 backtest_end) y es
   inalcanzable: con TP en 1.03E el extremo es < 1.03E, el nivel de trailing < 0.9785E < 0.98E (el SL, que
   se evalua antes); en short es simetrico. El port no se instrumento: sus 227 salidas coinciden en hora.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import math
import os
import sys
from pathlib import Path

# Regimen de cross-validation: sin slippage. Debe fijarse ANTES de importar quantagent.settings.
os.environ["TRADING_SLIPPAGE_PCT"] = "0"

import pandas as pd  # noqa: E402

try:
    from backtesting import Backtest, Strategy  # noqa: E402
except ImportError:  # pragma: no cover
    sys.exit("Falta backtesting.py: pip install -e '.[dev]' (o pip install backtesting)")

from quantagent import settings  # noqa: E402
from quantagent.backtesting.backtest import Backtest as ProjectBacktest  # noqa: E402
from quantagent.backtesting.export import COLUMNS  # noqa: E402
from quantagent.strategy.registry import build_strategy  # noqa: E402
from quantagent.trading.paper_broker import PaperBroker  # noqa: E402

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "spy-90d.csv"
SYMBOL = "SPY"
SCALE = float(2**27)  # potencia de 2: dividir/multiplicar precios es exacto en float

# Parametros del port (lo que la Strategy usa de verdad). La tabla los compara con el proyecto.
PORT_PARAMS = {
    "rsi_period": 14,
    "oversold_threshold": 30.0,
    "overbought_threshold": 70.0,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.03,
    "trailing_stop_pct": 0.05,
    "initial_cash": 100000.0,
    "base_position_pct": 0.05,
    "slippage_pct": 0.0,
    "commission": 0.0,
}


def project_params() -> dict:
    """Parametros del motor del proyecto, leidos del codigo (no literales)."""
    strat = build_strategy("RSIMeanReversionStrategy")  # igual que quantagent/cli/backtest.py
    broker = PaperBroker(slippage_pct=settings.TRADING_SLIPPAGE_PCT)  # igual que StrategyAssembler
    sig = inspect.signature(ProjectBacktest.__init__)
    return {
        "rsi_period": strat.rsi_period,
        "oversold_threshold": strat.oversold_threshold,
        "overbought_threshold": strat.overbought_threshold,
        "stop_loss_pct": strat.stop_loss_pct,
        "take_profit_pct": strat.take_profit_pct,
        "trailing_stop_pct": strat.trailing_stop_pct,
        # el CLI no pasa initial_capital: rige el default de Backtest, no TRADING_INITIAL_CASH
        "initial_cash": sig.parameters["initial_capital"].default,
        "base_position_pct": settings.TRADING_BASE_POSITION_PCT,
        "slippage_pct": broker.slippage_pct,
        "commission": 0.0 if broker.commission_model == "none" else math.nan,
        "_required_history_bars": strat.required_history_bars,
    }


def print_param_table(proj: dict) -> bool:
    print(f"{'parametro':<22}{'proyecto':>12}{'port':>12}  igual")
    ok = True
    for name, port_value in PORT_PARAMS.items():
        same = proj[name] == port_value
        ok &= same
        print(f"{name:<22}{proj[name]:>12}{port_value:>12}  {'SI' if same else 'NO'}")
    return ok


def rsi(close, period: int):
    """Mismo calculo que RSIMeanReversionStrategy._calculate_rsi (media simple, perdida 0 -> 1e-10)."""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return (100 - (100 / (1 + rs))).values


class RsiPort(Strategy):
    n_real_bars = 0  # sin la vela centinela
    min_history = 30
    intrabar = False  # True (solo QuantAgent-832, experimento): SL/TP nativos de backtesting.py (high/low)

    def init(self):
        self.rsi = self.I(rsi, self.data.Close * SCALE, PORT_PARAMS["rsi_period"])
        self.pos = None  # {"units": +long/-short, "stop", "tp", "extreme"} espeja ActivePosition

    def _real(self, scaled: float) -> float:
        return float(scaled) * SCALE

    def _order(self, units: int, **sl_tp):
        (self.buy if units > 0 else self.sell)(size=abs(units), **sl_tp)

    def _exit_triggered(self, price: float) -> bool:
        p, long = self.pos, self.pos["units"] > 0
        if (price <= p["stop"]) if long else (price >= p["stop"]):
            return True
        if (price >= p["tp"]) if long else (price <= p["tp"]):
            return True
        pct = PORT_PARAMS["trailing_stop_pct"]
        if long:  # el extremo arranca en None y se fija en la primera vela evaluada, como el motor
            p["extreme"] = price if p["extreme"] is None or price > p["extreme"] else p["extreme"]
            return price < p["extreme"] * (1 - pct)
        p["extreme"] = price if p["extreme"] is None or price < p["extreme"] else p["extreme"]
        return price > p["extreme"] * (1 + pct)

    def next(self):
        n = len(self.data)
        if n > self.n_real_bars:  # vela centinela: solo procesa las ordenes de la ultima vela real
            return
        if self.intrabar and self.pos and self.position.size == 0:
            self.pos = None  # la libreria cerro el trade dentro de la vela por sl=/tp=
        held = self.position.size
        assert held == (self.pos["units"] if self.pos else 0), "el broker cancelo o altero una orden"
        if n >= self.min_history:
            price = self._real(self.data.Close[-1])
            if self.pos and self._exit_triggered(price):
                self._order(-self.pos["units"])
                self.pos = None
            if self.pos is None:
                self._maybe_enter(price)
        if n == self.n_real_bars and self.pos:  # cierre forzado ('backtest_end') al ultimo close
            self._order(-self.pos["units"])

    def _maybe_enter(self, price: float):
        r = float(self.rsi[-1])
        lo, hi = PORT_PARAMS["oversold_threshold"], PORT_PARAMS["overbought_threshold"]
        if r < lo:
            sign, conf = 1, 1.0 - (r / lo)
        elif r > hi:
            sign, conf = -1, (r - hi) / (100 - hi)
        else:
            return
        sl, tp = PORT_PARAMS["stop_loss_pct"], PORT_PARAMS["take_profit_pct"]
        qty = self.equity * PORT_PARAMS["base_position_pct"] * conf / price
        units = int(round(qty * SCALE))
        if units < 1:
            return
        stop, take = round(price * (1 - sl * sign), 8), round(price * (1 + tp * sign), 8)
        self._order(sign * units, **({"sl": stop / SCALE, "tp": take / SCALE} if self.intrabar else {}))
        self.pos = {"units": sign * units, "stop": stop, "tp": take, "extreme": None}


def load_fixture(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    raw = raw[["open", "high", "low", "close", "volume"]].astype(float)
    last = raw.index[-1]
    step = raw.index[1] - raw.index[0]
    sentinel = pd.DataFrame(
        {"open": raw.close.iloc[-1], "high": raw.close.iloc[-1], "low": raw.close.iloc[-1],
         "close": raw.close.iloc[-1], "volume": 0.0},
        index=[last + step],
    )
    df = pd.concat([raw, sentinel])
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]] / SCALE
    df.columns = [c.capitalize() for c in df.columns]
    return df


def run_port_stats(history_bars: int, intrabar: bool = False):
    df = load_fixture(FIXTURE)
    bt = Backtest(df, RsiPort, cash=PORT_PARAMS["initial_cash"], commission=0.0, margin=1.0,
                  trade_on_close=True, hedging=False, exclusive_orders=False, finalize_trades=False)
    return bt.run(n_real_bars=len(df) - 1, min_history=history_bars, intrabar=intrabar)


def run_port(history_bars: int) -> pd.DataFrame:
    return run_port_stats(history_bars)["_trades"].sort_values("EntryTime")


def write_csv(trades: pd.DataFrame, out: Path) -> None:
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for t in trades.itertuples():
            w.writerow([
                t.EntryTime.isoformat(), t.ExitTime.isoformat(), SYMBOL, "buy" if t.Size > 0 else "sell",
                abs(t.Size) / SCALE, t.EntryPrice * SCALE, t.ExitPrice * SCALE,
                "", t.PnL, "",  # stop_loss y exit_reason: sin equivalente en backtesting.py
            ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="crossval_trades.csv", help="CSV de salida")
    args = ap.parse_args()

    proj = project_params()
    if not print_param_table(proj):
        print("ERROR: los parametros del port difieren del proyecto", file=sys.stderr)
        return 1
    trades = run_port(proj["_required_history_bars"])
    write_csv(trades, Path(args.out))
    print(f"Trades: {len(trades)}")
    print(f"CSV escrito en {args.out}")
    return 0 if len(trades) else 1


if __name__ == "__main__":
    sys.exit(main())
