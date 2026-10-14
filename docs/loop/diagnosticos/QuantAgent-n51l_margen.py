"""QuantAgent-n51l: primera entrada que backtesting.py cancela por margen con costos cripto (rsi, llenado al cierre).

Uso único, fixture sintético (spy-2y-1d), sin datos reales ni Postgres:
    DATABASE_URL=sqlite:///$PWD/nube-test.db python docs/loop/diagnosticos/QuantAgent-n51l_margen.py
No cambia el motor ni el adaptador: captura el motor envolviendo Backtest.run y la orden cancelada
envolviendo _process_orders del broker de la librería.
"""
import csv
import io
import os
import runpy
import sys
import tempfile
from pathlib import Path

from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
import compare_control as c  # noqa: E402

from quantagent.backtesting import backtest as motor  # noqa: E402

capturado = []
run_original = motor.Backtest.run
motor.Backtest.run = lambda self, *a, **k: (capturado.append(self), run_original(self, *a, **k))[1]
tasas = c.settings.COSTOS_PERFILES["cripto"]
s, cm = tasas["slippage_pct"], tasas["commission_pct"]
SCALE = c.port.SCALE

t = runpy.run_path(str(ROOT / "tests/test_compare_control.py"))
with tempfile.TemporaryDirectory() as d:
    os.environ["QUANTAGENT_SNAPSHOT_DIR"] = d
    os.environ["QUANTAGENT_PRUEBA_DESDE"] = os.environ["QUANTAGENT_RESERVA_DESDE"] = "2030-01-01"
    t["write_snapshot"]("t", {"SPY": t["_daily_frame"]()})
    res = CliRunner().invoke(t["backtest_group"], ["compare", "--strategies", "rsi", "--symbols", "SPY", "--snapshot",
                                                  "t", "--costos", "cripto", "--out", d + "/cmp.csv", "--trades-dir", d + "/tr"])
    assert res.exit_code == 0, res.output
    motor_trades = list(csv.DictReader(open(d + "/tr/rsi-SPY.csv")))
    c.settings.TRADING_BASE_POSITION_PCT = 1.0
    df = c.port.load_fixture(snapshot="t", symbol="SPY")

    cancelada = []
    Strat = c.bt_adapter.StrategyAdapter
    init_original = Strat.init

    def init(self):
        init_original(self)
        broker, proc = self._broker, self._broker._process_orders

        def proc_spy():
            antes = [(o, o.size, broker._cash, broker.margin_available) for o in broker.orders if not o.parent_trade]
            cerrados = len(broker.closed_trades)
            proc()
            for o, size, cash, margen in antes:  # orden de entrada que no abrió nada ni cerró nada
                if o not in broker.orders and len(broker.closed_trades) == cerrados and \
                        not any(tr.entry_bar == broker._i - 1 for tr in broker.trades):
                    cancelada.append((broker._i - 1, size, cash, margen))
        broker._process_orders = proc_spy
    Strat.init = init
    stats = c.bt_adapter.run_adapter(c.build_strategy(c.cli_aliases()["rsi"]), df,
                                     symbol="SPY", timeframe="1d", fill="close", **tasas)

trades = stats["_trades"]
bars = stats["_strategy"].bars
eng = capturado[-1].get_equity_curve()
print(f"trades: motor={len(motor_trades)} adaptador={len(trades)} intentos cancelados={len(cancelada)}")
i, units, cash, margen = cancelada[0]
px = float(bars["close"].iloc[i])
ts = bars["timestamp"].iloc[i]
qty = abs(units) / SCALE
lado = "compra" if units > 0 else "venta en corto"
print(f"\nprimera cancelada: vela {i} ({ts:%F}), {lado} de {qty:.8f} a close {px:.4f}")
m = next(r for r in motor_trades if r["entry_time"][:10] == f"{ts:%F}")
print("motor, trade:", m)
print("motor, cuenta (vela anterior y vela de entrada):\n", eng.iloc[i - 1:i + 1].to_string())
llenado = float(m["entry_price"])
print(f"motor: efectivo antes de entrar {eng['cash'].iloc[i] - qty * llenado * (1 - cm):.2f}, "
      f"llenado {llenado:.6f}, nocional {qty * llenado:.2f}, comisión {qty * llenado * cm:.2f}")
# _Broker._process_orders: adjusted_price + commission(size, price) / |size|, contra margin_available * leverage
ajustado = px * (1 + (s if units > 0 else -s)) + px * cm
print(f"librería: margin_available {margen:.2f}; exige {qty:.8f} x {ajustado:.6f} = {qty * ajustado:.2f} "
      f"(nocional {qty * px:.2f}, spread {qty * px * s * (1 if units > 0 else -1):+.2f}, "
      f"comisión {qty * px * cm:+.2f}); faltante {qty * ajustado - margen:.2f} -> cancela")
episodios = [b for k, (b, *_) in enumerate(cancelada) if k == 0 or b != cancelada[k - 1][0] + 1]
print("primer intento de cada racha de cancelaciones:", [f"{bars['timestamp'].iloc[b]:%F}" for b in episodios])
print("lados:", {("compra" if u > 0 else "corto") for _, u, *_ in cancelada})

# Contraprueba: la misma corrida con apalancamiento 1 + comisión (margin = 1 / (1 + c)).
Strat.init = init_original
Original = c.bt_adapter.Backtest
for m_ in (1.0, 1 / (1 + cm)):
    c.bt_adapter.Backtest = lambda *a, _m=m_, **k: Original(*a, **{**k, "margin": _m})
    st = c.bt_adapter.run_adapter(c.build_strategy(c.cli_aliases()["rsi"]), df.copy(), symbol="SPY", timeframe="1d",
                                  fill="close", **tasas)
    prt = list(csv.DictReader(io.StringIO(c.trades_to_csv(c.bt_adapter.trade_rows(st, "SPY")))))
    print(f"margin={m_:.6f}:", c.compare_trades_with_reasons(motor_trades, prt))
