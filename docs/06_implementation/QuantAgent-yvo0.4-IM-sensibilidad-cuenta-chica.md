# Sensibilidad de cuenta chica contra SPY

`quantagent backtest cuenta-chica` ejecuta la estrategia solicitada para USD 1.000, 2.500 y 5.000 por defecto. `--capitales` permite reemplazar esos importes. Presenta tarjetas de texto con Sharpe, retorno, costos estimados pagados, símbolos con entradas que quedaron en cero, mayor concentración y diferencia de retorno contra SPY comprar y mantener.

La hipótesis de esta entrega es la lectura mínima del ticket: la estrategia configurada sobre el único símbolo de un fixture o snapshot. El benchmark SPY usa las mismas velas de ese activo; por eso los fixtures de prueba se identifican como SPY. Para cuenta chica, ambos lados usan acciones enteras, slippage, comisión y spread efectivos, costo fijo de entrada y, si el snapshot tiene dividendos implícitos, retención y costo fiscal. El CSV de `backtest run` no cambia.

El benchmark de cuenta chica se calcula en `buy_and_hold_cuenta_chica`, separado del comando CLI. El spread efectivo está incluido en `r["slippage_pct"]` por el motor y se aplica al benchmark; el renglón de costos de la estrategia suma ese mismo costo por lado junto con comisión, impuestos y costo fijo. `_engine_config` recibe el capital de cada tarjeta.

Validación solicitada: `NUMBA_DISABLE_JIT=1 pytest tests/test_buy_and_hold.py tests/test_backtest_cli.py tests/test_cli_help.py`.
