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
2026-10-06 · #24 · PM: hx0.7 vi: tabla corregida y recalculada por mi lado desde la equity curve en base limpia: 0.57 / 1.37 / 1.13 con 0.05% y 0.40 / 0.99 / 0.82 con 1.00%; el fixture spy-90d tiene 2160 velas = 90 días × 24 h, fines de semana incluidos decido: escalo ¿qué anualización para 1h: A 252×6.5 (Sharpe 0.57), B 365×24 (1.37) o C 252×24 (1.13)? Recomiendo B condicional: 8760 cuando el backtest corre sin filtro de horario (fixtures 24/7) y 1638 cuando market_hours_filter está activo. Contestá con R: hx0.7 vi: ... decido: merge (acepta B condicional) o cambio <opción>.
2026-10-06 · #24 · R: hx0.7 vi: tabla 0.57 / 1.37 / 1.13, el fixture spy-90d es 24/7 decido: cambio agregar opción D: periodos por año derivados de los datos (velas observadas / años cubiertos) y adoptarla   (Fede mergeó el doc tal cual; la opción D se implementa en hx0.4)
2026-10-06 · #25 · PM: hx0.1 vi: rompí la fórmula del short en manager.py y la de total_return_pct y el test nuevo falla en los dos casos; equity_final recalculada desde el CSV de equity da inicial + 12120.98; suite 819 passed decido: merge
2026-10-06 · #30 · R: hx0.2 vi: 113 ganadores, 114 perdedores, 0 neutros. Además mostrar n/a en lugar de inf decido: merge
2026-10-07 · #32 · PM: hx0.6 vi: corrí el recálculo sobre la salida real de rsi/spy-90d en base limpia y coincide con el engine: Sharpe 0.57 y max drawdown 0.090128; con --periods-per-year 8760 da 1.37, igual que mi cálculo de ayer; suite 825 passed decido: merge
2026-10-07 · #35 · PM: y8z vi: metí un random real en el precio de ejecución del broker y verify falla con Metric mismatch en total_pnl, sharpe_ratio y max_drawdown; la base del entorno queda intacta (mismo md5 antes y después); rsi/spy-90d da OK reproducible; suite 830 passed; ojo: fifty-two-week-high y triple-screen hacen 0 trades en spy-smoke decido: merge
2026-10-07 · #34 · R: hx0.3 vi: Sharpe 8.74 y drawdown 0.001204, trades y PnL sin cambio decido: merge
```
