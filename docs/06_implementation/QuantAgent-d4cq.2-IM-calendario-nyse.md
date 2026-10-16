# QuantAgent-d4cq.2: reutilizar NYSE

Se comparte una instancia de NYSE; se conserva `valid_days`, la sesión siguiente y la caché por fecha.
Sin cambios de señales, SQL, llenado ni referencias. Sigue al [perfil](QuantAgent-d4cq.1-IM-perfil-motor.md).
Medido sin cProfile, en serie, SPY 2007–2018, snapshot `etf-1d-2026-10`, Postgres del loop y defaults del CLI.
Cada proceso reconstruyó su caché temporal de fuentes Matplotlib; los tiempos incluyen ese arranque variable.

| Estrategia | Antes s | Después s | Trades | Win rate | PF | Sharpe ratio | PnL |
|---|---:|---:|---:|---:|---:|---:|---:|
| rsi | 56.20 | 35.24 | 187 | 43.32% | 0.86 | -11.86 | -447.29 |
| fifty-two-week-high | 35.64 | 29.93 | 5 | 40.00% | 2.02 | -95.89 | 78.75 |
| sma-cross | 43.13 | 29.26 | 5 | 80.00% | 13.58 | -2.19 | 5294.87 |
| momentum-12m | 531.17 | 28.58 | 3 | 66.67% | 20.59 | -1.64 | 6586.89 |

Las cinco métricas son idénticas antes/después (comparadas con `diff`); momentum mejora 18.6 veces.
Son defaults de `backtest run`, no el tamaño de posición de `compare`: no reemplazan el número de referencia RSI.
Repetir antes y después del cambio, con el mismo entorno y sin alterar `DATABASE_URL`:

```bash
for s in rsi fifty-two-week-high sma-cross momentum-12m; do timeout 15m /usr/bin/time -f "PARED $s: %e s" python -m quantagent.cli backtest run --strategy "$s" --snapshot etf-1d-2026-10 --symbol SPY --from 2007-01-01 --to 2018-12-31 || exit; done
```
