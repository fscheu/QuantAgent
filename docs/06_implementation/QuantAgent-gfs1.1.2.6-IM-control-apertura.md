# QuantAgent-gfs1.1.2.6 — control con apertura

Diagnóstico sobre etf-1d-2026-10, RSI 2007–2018, costos etf, tamaño 1.
Grupo última fila (11 celdas): RSI/IWM, 2018-12-31; anterior 2018-12-28 sin salida por regla; tampoco la hay al final.
Motor: backtest_end, precio/close=0.9995, precio/open=1.000845369; adaptador: backtest_end, precio/open=0.9995, precio/close=0.998156440.
Causa: `finalize_trades=True` repite broker.next con el OHLC final y liquida al open; `_close_remaining_positions` liquida al close.
Propuesta en adaptador (≤100 líneas): liquidación final explícita al close con slippage y motivo backtest_end, sin volver a procesar entradas/stops.
Debe descartar salidas por regla emitidas en la última vela: no tienen apertura siguiente; explica también el posible motivo de regla en la última fila (XLK).
Grupo salidas intermedias (5 celdas): RSI/EFA, fila 32, 2008-10-16; anterior 2008-10-15 genera LONG, sin posición ni salida pendiente.
Entrada y salida ocurren el 16: tp/open=0.999514602, stop/open=0.950994475, low/open=0.948486243; la apertura ya supera TP y luego se toca SL.
Motor: TAKE_PROFIT en apertura, precio/open=0.9995; adaptador: STOP_LOSS, precio/open=0.950518978 (nivel SL con slippage).
Causa: el motor comprueba el gap también en la vela de entrada; `apply_opening_gap` omite esa vela y la librería prioriza SL intrabar.
Propuesta en adaptador (≤100 líneas): después de ejecutar entradas, aplicar gap a las nuevas posiciones antes de sus SL/TP intrabar, conservando prioridad SL si ambos niveles se tocan sin gap.
Ambas propuestas replican el motor con apertura; no cambian sus números. No se corrigió código ni se verificó individualmente el resto de las 16 celdas.

Reproducción desde la raíz (usa DATABASE_URL de tests; no abre 2019):
`timeout 5m python -m quantagent.cli backtest compare --strategies rsi --symbols IWM,EFA --snapshot etf-1d-2026-10 --from 2007-01-01 --to 2018-12-31 --costos etf --out /tmp/gfs126.csv --trades-dir /tmp/gfs126-trades && python -c 'from pathlib import Path; exec(Path("docs/06_implementation/QuantAgent-gfs1.1.2.6-IM-control-apertura.md").read_text().split("```python\n")[1].split("```")[0])'`

```python
import csv, sys
sys.path.insert(0, "scripts")
import bt_adapter as b
from quantagent.strategy.registry import build_strategy, cli_aliases
b.settings.TRADING_BASE_POSITION_PCT = 1.
for symbol, rownum in (("IWM", -1), ("EFA", 31)):
    d = b.port.load_fixture(snapshot="etf-1d-2026-10", symbol=symbol, from_date="2007-01-01", to_date="2018-12-31")
    strategy = build_strategy(cli_aliases()["rsi"])
    events = {}
    original = strategy.should_exit
    def trace(pos, price, window):
        result = original(pos, price, window)
        events[str(window.timestamp.iloc[-1].date())] = result
        return result
    strategy.should_exit = trace
    stats = b.run_adapter(strategy, d, symbol=symbol, timeframe="1d", fill="next-open", commission_pct=0, slippage_pct=.0005)
    motor = list(csv.DictReader(open(f"/tmp/gfs126-trades/rsi-{symbol}.csv")))[rownum]
    adapter = b.trade_rows(stats, symbol)[rownum]
    i = d.index.get_loc(adapter.exit_time)
    o, c = float(d.Open.iloc[i])*b.SCALE, float(d.Close.iloc[i])*b.SCALE
    print(symbol, "vela", d.index[i].date(), "anterior", d.index[i-1].date(), "señal", events.get(str(d.index[i-1].date())))
    print("motor", motor["exit_reason"], "precio/open", round(float(motor["exit_price"])/o, 9), "precio/close", round(float(motor["exit_price"])/c, 9))
    print("adaptador", adapter.exit_reason, "precio/open", round(adapter.exit_price/o, 9), "precio/close", round(adapter.exit_price/c, 9))
    print("última", i == len(d)-2, "señal actual", events.get(str(d.index[i].date())))
    assert (motor["exit_reason"], adapter.exit_reason) == (("backtest_end", "backtest_end") if symbol == "IWM" else ("TAKE_PROFIT", "STOP_LOSS"))
    assert abs(float(motor["exit_price"])/(c if symbol == "IWM" else o)-.9995) < 1e-8 and abs(adapter.exit_price/o-(.9995 if symbol == "IWM" else .950518978)) < 1e-8
    if symbol == "EFA":
        w = stats["_strategy"].bars.iloc[i-30:i]
        signal = build_strategy(cli_aliases()["rsi"]).generate_signal(w.to_dict(orient="records"), symbol, "1d", float(w.close.iloc[-1]))
        print("entrada misma vela", str(adapter.entry_time) == str(adapter.exit_time), "señal previa", signal.decision, "stop/open", round(signal.stop_loss/o, 9), "tp/open", round(signal.take_profit/o, 9), "low/open", round(float(d.Low.iloc[i])/float(d.Open.iloc[i]), 9))
```
