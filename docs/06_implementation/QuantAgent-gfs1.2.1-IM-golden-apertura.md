# Golden RSI / spy-90d con apertura siguiente

QuantAgent-gfs1.2.1 migra únicamente el golden corto al default del CLI.
El test no pasa `--llenado`: verifica métricas y CSV byte a byte, además
de conservar el diagnóstico por fila. No cambia el motor ni los oráculos legacy.
`.gitattributes` conserva CRLF en este CSV al hacer checkout, igual que el CLI.

| Métrica | Cierre anterior | Apertura nueva |
|---|---:|---:|
| Trades | 340 | 227 |
| Win rate | 33.24% | 49.78% |
| Profit factor | 1.35 | 3.81 |
| Sharpe ratio | 9.46 | 32.48 |
| Total PnL | 4571.27 | 13532.99 |
| Max drawdown | 0.001414 | 0.000879 |

Costos sin cambio: slippage 0.05% por lado, comisión 0.00% por lado.
Valores observados con el fixture versionado; no son una proyección.

## Regeneración

Desde la raíz, con el entorno Python activo y DATABASE_URL de tests migrada:

`python -m quantagent.cli backtest run --strategy rsi --fixture spy-90d --out tests/fixtures/golden/rsi-spy-90d-trades.csv --equity-out /tmp/gfs121-equity.csv`

Validar sin regenerar: `python -m pytest -q tests/test_golden_rsi_spy_90d.py`.
El test ejecuta el mismo CLI en una SQLite temporal limpia, sin cambiar la
base configurada en el entorno. Gate adicional: `scripts/loop/smoke.sh`.
El golden SPY 2007–2018 y el inventario legacy quedan para los hijos siguientes
de [la partición](../loop/particiones/QuantAgent-gfs1.2.md).
