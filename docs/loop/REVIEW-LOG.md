# Registro de revisión del loop

Una línea por entrega revisada. El loop copia acá la línea `R:` del PR anterior al hacer la siguiente entrega.
Formato: `PLAN-CONTINUACION.md` §5.

```text
2026-09-29 · #1 · R: cierre vi: decido: merge   (previa al loop; vi: vacío, no pasaría el gate)
2026-09-29 · #2 · R: hyt vi: líneas comentadas ok decido: merge
2026-10-04 · #4 · R: e35 vi:ok Al borrado del archivo decido: merge
2026-10-04 · #6 · R: bv8 vi: de acuerdo con el cambio decido: merge
2026-10-05 · #7 · R: fdi vi: testeado en la VM, resultado ok: ── loop/QuantAgent-fdi @ 5965aa1c ── Trades: 7 Win rate: 71.43% Profit factor: 1.44 Sharpe ratio: 2.10 Total PnL: 94.02 decido: merge
2026-10-05 · #9 · R: fiu vi: revise el código del test y ejecute en la vm, test pass ok decido: merge
2026-10-05 · #10 · R: 3km vi: ~/repos/projects/QuantAgent/scripts/loop/try.sh loop/QuantAgent-3km -- 'python -m pytest -q tests/test_data_provider.py -k YahooIntradayWindow 2>&1 | tail -3' ── loop/QuantAgent-3km @ 4bf01bbf ── tests/test_data_provider.py ...... [100%] ================= 6 passed, 18 deselected, 6 warnings in 1.17s ================= decido: merge
2026-10-05 · #12 · R: 11v vi:ejecute y revise el csv. creo que para futuro hay que redondear los valores a 4 decimales maximo porque mas de eso no tiene sentido, inclusive cuando son valores en % como 0,13. Generar un ticket para hacerlo despues con menor prioridad, ahora merge decido: merge
2026-10-05 · #13 · R: 89e.1 vi: registros duplicados en la base para la misma operación no quiero, es un bug decido: merge
2026-10-06 · #14 · R: 89e vi: revise la tabla de tardes después de ejecutar el loop y lo veo ok decido: merge
2026-10-06 · #15 · R: hx0.5 vi: el PNL no sé recalcula. Encontramos que El slippage estaba sobredimensionado. Agregamos otros tickets a la cola. decido: merge
2026-10-06 · #18 · R: hx0.8 vi: try.sh en la VM da PnL por trade 227/227 y métricas sin cambio (227 / -4717.33) decido: merge
2026-10-06 · #23 · PM: hx0.9 vi: corrí en base limpia las tres variantes: 0.05% da 227 / 12120.98, TRADING_SLIPPAGE_PCT=0.01 vuelve a 227 / -4717.33 y slippage 0 da 13084.98 (monótono, 113 ganadores en las tres); recalc_metrics 227/227; suite 818 passed decido: merge
```

