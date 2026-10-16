# QuantAgent-gfs1.1.2.3 — cantidad en apertura

Diagnóstico sin cambios de código; reemplaza el bloqueo de #188. RSI/SPY,
snapshot etf-1d-2026-10, 2007–2018, apertura, costos etf, tamaño 1, confianza.
Ambos producen 187 trades; fila 1: corto, entrada 2007-02-15T00:00:00.

Sea E el capital inicial (100% efectivo), C el cierre de la señal (2007-02-14),
O la apertura de ejecución y f=0.09921573120608494 la confianza de esa señal.
O/C=1.0004119761399575; E/C=984.4209703305041; E/O=984.0155793904489.

| Aspecto, fila 1 | Motor | Adaptador next-open |
|---|---|---|
| Base de capital | E, equity al ejecutar en O, sin posiciones ni costos previos | E, equity al emitir la señal en C, sin posiciones ni costos previos |
| Referencia del tamaño | O: `_analyze_and_trade` → `_execute_strategy_signal` → `execute_decision` | C: `StrategyAdapter._maybe_enter` |
| Cantidad antes de persistir/cuantizar | E × f / O = 97.62982522740272 | E × f / C = 97.67004638594463 |
| Cantidad exportada | 97.62982523 | 97.67004638910294 |
| Redondeo | Numeric(18,8), 8 decimales | round(qty × 2**27) / 2**27 |
| Costos en el tamaño | Comisión 0; slippage 0.0005 excluido; corto sin recorte de efectivo | Comisión 0; slippage 0.0005 excluido; `cap_open_entries` solo recorta largos |

Ambos venden a O × (1−0.0005); el precio ejecutado incluye slippage, el divisor del tamaño usa O sin slippage en el motor.
Causa: el adaptador conserva la cantidad calculada en C al llenar en O; el motor la calcula en O.
La razón qty_adaptador/qty_motor antes de redondear es O/C; la diferencia no es ruido de cuantización.

Corrección propuesta (estimación ≤100 líneas, adaptador): guardar confianza y fracción de tamaño con la orden pendiente;
en `cap_open_entries`, después de procesar cierres, recalcular ambos lados con equity marcada a O y divisor O,
cuantizar a 2**27 y aplicar el recorte de costos de compra existente. Mantener cierre sin cambios.
Probar una apertura distinta de C en cada lado y repetir RSI/SPY; no afirmar coincidencia global antes de hacerlo.
El motor sigue dimensionando con la referencia del llenado; no cambiar tolerancias ni referencias.

Reproducción (no imprime precios absolutos; compara la fila del motor con el cálculo real del adaptador):

```bash
timeout 5m python -m quantagent.cli backtest compare --strategies rsi --symbols SPY --snapshot etf-1d-2026-10 --from 2007-01-01 --to 2018-12-31 --costos etf --out /tmp/gfs123.csv --trades-dir /tmp/gfs123-trades && python -c 'import sys,csv; sys.path.insert(0,"scripts"); import bt_adapter as b; from quantagent.strategy.registry import build_strategy,cli_aliases; b.settings.TRADING_BASE_POSITION_PCT=1.; d=b.port.load_fixture(snapshot="etf-1d-2026-10",symbol="SPY",from_date="2007-01-01",to_date="2018-12-31"); s=b.run_adapter(build_strategy(cli_aliases()["rsi"]),d,symbol="SPY",timeframe="1d",fill="next-open",commission_pct=0,slippage_pct=.0005); t=s["_trades"].sort_values("EntryTime").iloc[0]; i=int(t.EntryBar); c=float(d.Close.iloc[i-1])*b.SCALE; o=float(d.Open.iloc[i])*b.SCALE; w=s["_strategy"].bars.iloc[i-30:i]; f=build_strategy(cli_aliases()["rsi"]).generate_signal(w.to_dict(orient="records"),"SPY","1d",c).confidence; e=list(csv.DictReader(open("/tmp/gfs123-trades/rsi-SPY.csv"))); qm=float(e[0]["qty"]); qa=abs(t.Size)/b.SCALE; assert len(e)==len(s["_trades"])==187; assert e[0]["side"]=="sell" and t.Size<0; assert abs(qm-b.INITIAL_CASH*f/o)<5e-9; assert qa==round(b.INITIAL_CASH*f/c*b.SCALE)/b.SCALE; print("qty motor/adaptador:",qm,qa); print("O/C, E/C, E/O:",o/c,b.INITIAL_CASH/c,b.INITIAL_CASH/o); print("PASS: 187 trades; fila 1 dimensionada en O / C")'
```
