# QuantAgent-d4cq.32 — Tanda secuencial

`backtest hipotesis tanda hipotesis/ --snapshot <snapshot> --carpeta <salida>` procesa los YAML por nombre y deja `corrida.log` e `informe.html` por ID; sin estado disponible queda el log. El informe muestra Sharpe en exceso de DGS3MO cuando hay curvas guardadas; también se puede pedir con `backtest hipotesis informe <id> --archivo <yaml> --carpeta-equity <dir> --snapshot <snapshot>`. Sin esos datos, informa que el diagnóstico no está disponible. Un YAML inválido deja `<nombre>.log` en la salida. Los cortes y errores no interrumpen la tanda; termina con exit 0 y `resumen.md` desde su inicio UTC.

Validación: `NUMBA_CACHE_DIR=/tmp/quantagent-numba-cache python -m pytest -q tests/test_hipotesis_tanda.py` (cortes, excepción, YAML inválido y resumen sin estados anteriores).
