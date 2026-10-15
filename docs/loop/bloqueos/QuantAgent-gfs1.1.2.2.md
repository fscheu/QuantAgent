# QuantAgent-gfs1.1.2.2: partición bloqueada por jerarquía

Inventario previo: 75 construcciones de Backtest en 15 archivos
`tests/test_backtest*.py`, además de invocaciones del CLI y oráculos/golden.
Adaptarlas explícitamente sin defaults globales requiere varias entregas:
el conjunto excede 10 archivos y se estima en 130–180 líneas.
No se editó código ni expectativas ni se regeneraron fixtures.

Los tres intentos de `bd create --parent QuantAgent-gfs1.1.2.2`
fallaron: `maximum hierarchy depth (3) exceeded for parent
QuantAgent-gfs1.1.2.2`. No se creó ningún hijo ni se cambió la cola.

División propuesta, cada entrega con revisión probar y máximo 100 líneas:
1. Motor (54 construcciones candidatas, 8 archivos, estimación 80 líneas):
   test_backtest, integration, end_to_end_timeframe_isolation, market_hours,
   holiday_gap, lookback_window_4w4, position_monitor, run_isolation.
2. Métricas (19 construcciones candidatas, 5 archivos, 40 líneas):
   phase4_metrics, metrics_run_scoping, sharpe_audit, pnl_audit, win_rate_audit.
3. CLI y oráculos/golden (estimación 90 líneas, inventariar antes de editar):
   cierre explícito en referencias; conservar pruebas del default/apertura.
   Golden RSI/SPY hasta 2018 exige 187 trades / -9141.21 sin números nuevos.

Aceptación por entrega: expectativas idénticas, cierre explícito donde se
depende de él, tests de archivos modificados y smoke exit 0; la tercera
también corre los dirigidos y oráculos completos indicados en el padre.

Decisión que destraba: autorizar tickets hermanos bajo QuantAgent-gfs1.1.2
(o tickets sin padre con dependencias), y reemplazar la fila del padre
en §4 por esos tickets. No se altera la jerarquía sin esa decisión.

Validación de esta entrega documental: scripts/loop/smoke.sh exit 0,
9 trades / PnL 542.36. No hay tests modificados; no se declara que se
cumplieron los criterios de implementación del ticket.
