"""QuantAgent-jzb6: alinea por fecha la curva de equity del motor y la del adaptador (llenado al cierre).

Uso único, fixture sintético (spy-2y-1d, fifty-two-week-high), sin datos reales ni Postgres:
    DATABASE_URL=sqlite:///$PWD/nube-test.db python docs/loop/diagnosticos/QuantAgent-jzb6_curvas.py
No cambia el motor ni el adaptador: captura la instancia del motor envolviendo Backtest.run.
"""
import os
import runpy
import sys
import tempfile
from pathlib import Path

import pandas as pd
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
import compare_control as c  # noqa: E402

from quantagent.backtesting import backtest as motor  # noqa: E402

capturado = []
run_original = motor.Backtest.run
motor.Backtest.run = lambda self, *a, **k: (capturado.append(self), run_original(self, *a, **k))[1]

t = runpy.run_path(str(ROOT / "tests/test_compare_control.py"))
with tempfile.TemporaryDirectory() as d:
    os.environ["QUANTAGENT_SNAPSHOT_DIR"] = d
    os.environ["QUANTAGENT_PRUEBA_DESDE"] = os.environ["QUANTAGENT_RESERVA_DESDE"] = "2030-01-01"
    t["write_snapshot"]("t", {"SPY": t["_daily_frame"]()})
    res = CliRunner().invoke(t["backtest_group"], ["compare", "--strategies", "fifty-two-week-high", "--symbols", "SPY",
                                                  "--snapshot", "t", "--out", d + "/cmp.csv", "--trades-dir", d + "/tr"])
    assert res.exit_code == 0, res.output
    c.settings.TRADING_BASE_POSITION_PCT = 1.0
    stats = c.bt_adapter.run_adapter(
        c.build_strategy("FiftyTwoWeekHighStrategy"), c.port.load_fixture(snapshot="t", symbol="SPY"),
        symbol="SPY", timeframe="1d", commission_pct=c.settings.TRADING_COMMISSION_PCT,
        slippage_pct=c.settings.TRADING_SLIPPAGE_PCT)

eng = capturado[-1].get_equity_curve().set_index("date")
eng.index = pd.to_datetime(eng.index).tz_localize(None).normalize()
ad = stats["_equity_curve"].iloc[:stats["_strategy"].n_real_bars]
ad.index = pd.to_datetime(ad.index).tz_localize(None).normalize()
trades = stats["_trades"]
print(f"puntos: motor={len(eng)} ({eng.index[0]:%F}..{eng.index[-1]:%F}) adaptador={len(ad)} ({ad.index[0]:%F}..{ad.index[-1]:%F})")
print(f"equity inicial: motor={eng['equity'].iloc[0]:.2f} adaptador={ad['Equity'].iloc[0]:.2f}")
j = eng.join(ad[["Equity"]], how="outer")
j["dif"] = j["equity"] - j["Equity"]
print("fechas solo en motor:", list(j.index[j["Equity"].isna()].strftime("%F"))[:5])
print("fechas solo en adaptador:", list(j.index[j["equity"].isna()].strftime("%F"))[:5])
malas = j[j["dif"].abs() > 0.01]
if malas.empty:
    sys.exit("sin diferencias > 0.01 USD en fechas comunes")
f = malas.index[0]
i = ad.index.get_loc(f)
cerrado = stats["_strategy"].bars  # precios reales del adaptador
abiertos = trades[(trades.EntryBar <= i) & (trades.ExitBar > i)]
qty = abiertos.Size.sum() / c.port.SCALE
px = float(cerrado["close"].iloc[i])
print(f"\nprimera diferencia > 0.01 USD: {f:%F} (dif {j.loc[f, 'dif']:.4f})")
print(f"motor:     equity={j.loc[f, 'equity']:.4f} cash={j.loc[f, 'cash']:.4f} posicion={j.loc[f, 'positions_value']:.4f}")
# La librería escribe Equity[i] antes de next(): en la vela de la entrada todavía no tiene posición.
print(f"adaptador: equity={ad['Equity'].iloc[i]:.4f} cash={ad['Equity'].iloc[i]:.4f} cantidad=0 posicion=0 "
      f"(la orden de {qty:.8f} a close {px:.4f} se llena al procesar la vela {i + 1})")
print("\ncontexto (±3 velas):")
print(j.iloc[max(0, j.index.get_loc(f) - 3):j.index.get_loc(f) + 4].to_string())
print("\ntrades adaptador:")
print(trades[["EntryBar", "ExitBar", "EntryTime", "ExitTime", "Size", "EntryPrice", "ExitPrice", "PnL", "Commission"]].to_string())
print("\nresumen dif: max abs", round(j["dif"].abs().max(), 4), "en", j["dif"].abs().idxmax().strftime("%F"))

# Hipótesis: con trade_on_close la orden de la vela i se llena al procesar la i+1, así que Equity[i] no
# descuenta el slippage ni la comisión de los llenados al cierre de i. El motor sí (equity tras la orden).
fills = list(trades.EntryBar) + list(trades.ExitBar)
velas = sorted(ad.index.get_loc(x) for x in malas.index)
print("velas con dif > 0.01:", velas, "| todas son de llenado:", set(velas) <= set(fills))
costo = pd.Series(0.0, index=ad.index)
s, cm = c.settings.TRADING_SLIPPAGE_PCT, c.settings.TRADING_COMMISSION_PCT
for tr in trades.itertuples():
    for b in (tr.EntryBar,):  # en las salidas no hay diferencia (ver abajo)
        costo.iloc[b] += abs(tr.Size) / c.port.SCALE * float(cerrado["close"].iloc[b]) * (s + cm)
corr = ad["Equity"] - costo
print("max abs dif motor vs adaptador corregido:", round((eng["equity"] - corr).abs().max(), 6))
per = (len(eng) - 1) * 365.25 * 86400 / (eng.index[-1] - eng.index[0]).total_seconds()


def sh(e):
    r = e.pct_change().dropna()
    return round(float(r.mean() / r.std() * per ** 0.5), 4)


print("Sharpe tasa 0: motor", sh(eng["equity"]), "adaptador", sh(ad["Equity"]), "adaptador corregido", sh(corr))
print("dif en velas de salida:", [round(float(j["dif"].iloc[b]), 6) for b in trades.ExitBar])
b = int(trades.ExitBar.iloc[0])
print("\nprimera salida, velas", b, b + 1, "(motor y adaptador; close):")
print(j.iloc[b:b + 2].assign(close=cerrado["close"].iloc[b:b + 2].values).to_string())
print("motivos de salida:", [stats["_strategy"].info[e]["reason"] for e in trades.EntryBar])
