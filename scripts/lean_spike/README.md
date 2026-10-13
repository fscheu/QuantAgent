# Spike de LEAN (QuantAgent-ryaw)

Código del spike del §9 de `docs/04_decisions/2026-10-12-DC-quantconnect-y-pivot-lean.md`. Vive en la rama
`spike/lean`, sin PR. No toca el motor, el CLI ni los tests.

Estado al 2026-10-13: los pasos 1 a 3 y 5 están escritos; **LEAN todavía no corrió** (la notebook no tiene Docker
usable). `lean_algos.py`, `config.json`, `run_lean.sh` y `closed_trades_to_csv.py` están sin ejecutar contra LEAN.

| Archivo | Qué hace | Probado |
|---|---|---|
| `write_lean_data.py` | SPY diario 2007-01-03..2018-12-31 del snapshot `etf-1d-2026-10` a `equity/usa/daily/spy.zip`, más factor file y map file mínimos | Sí: 3020 velas. Falta ver que LEAN lo lea |
| `lean_algos.py` | `SmaCrossSpike` y `RsiSpike` (QCAlgorithm, Python) | Solo la lógica, con un simulador de prueba que no es LEAN |
| `config.json`, `run_lean.sh` | Launcher con config propio, sin CLI ni login, contenedor sin red | No |
| `closed_trades_to_csv.py` | `closedTrades` del JSON de resultados al CSV de `--out` | No |
| `compare_trades.py` | "Coinciden N de M" por fechas y lado, desvío de PnL y `crossval_compare.compare_trades` | Sí, control contra control |

## Cómo se corre

Todo lo que se genera va a una carpeta fuera del repo (`$W`); los datos del snapshot no se commitean.

```bash
export QUANTAGENT_SNAPSHOT_DIR=<carpeta con etf-1d-2026-10>
R="--snapshot etf-1d-2026-10 --symbol SPY --from 2007-01-03 --to 2018-12-31"
# controles (perfil etf de COSTOS_PERFILES, 100% del capital)
TRADING_BASE_POSITION_PCT=1 python scripts/bt_adapter.py --strategy sma-cross $R --fill next-open --costos etf --out $W/control/adapter_sma-cross.csv
python -m quantagent.cli backtest run --strategy sma-cross $R --costos etf --tamano-posicion 1.0 --out $W/control/motor_sma-cross.csv
# datos y corrida
python scripts/lean_spike/write_lean_data.py --out $W/data
scripts/lean_spike/run_lean.sh SmaCrossSpike cash $W
scripts/lean_spike/run_lean.sh RsiSpike margin $W
# comparación
python scripts/lean_spike/closed_trades_to_csv.py $W/results/SmaCrossSpike/SmaCrossSpike.json --out $W/lean_sma-cross.csv
python scripts/lean_spike/compare_trades.py $W/control/adapter_sma-cross.csv $W/lean_sma-cross.csv --detalle
```
