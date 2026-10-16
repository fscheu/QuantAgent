# QuantAgent-d4cq.1: perfil del motor propio

Medición: 2026-10-14, lote `motor-del-filtro`. Diagnóstico; ningún cambio en `quantagent/`.
Contexto: [decisión de motor](../04_decisions/QuantAgent-d4cq-DC-motor-de-la-fabrica.md).
SPY, snapshot `etf-1d-2026-10`, rango pedido 2007-01-01–2018-12-31, 3.020 velas.
Postgres de tests del loop, sin cambiar `DATABASE_URL`; corridas en serie, parámetros y costos default del CLI, llenado al cierre.
Los tiempos de pared incluyen arranque y **sobrecarga de cProfile**: no son comparables directamente con los 14/115 s sin perfilador del PM.
La caché de fuentes de Matplotlib no era escribible y se reconstruyó en un directorio temporal por proceso.

## Reproducir

Desde la raíz, con el entorno Python activo y el snapshot disponible (hasta 15 minutos por estrategia):

```bash
for s in rsi fifty-two-week-high sma-cross momentum-12m; do timeout 15m /usr/bin/time -f "PARED $s: %e s" python -m cProfile -o /tmp/d4cq-$s.prof -m quantagent.cli backtest run --strategy "$s" --snapshot etf-1d-2026-10 --symbol SPY --from 2007-01-01 --to 2018-12-31 && python -c "import pstats; pstats.Stats('/tmp/d4cq-$s.prof').strip_dirs().sort_stats('cumulative').print_stats(10)" || exit; done
```

Los perfiles binarios quedan en `/tmp`, no se versionan. `try.sh` usa SQLite: permite repetir el diagnóstico, pero sus tiempos SQL no son los de esta tabla.

## Pared y top 10 por tiempo acumulado

| Estrategia | Pared (s) | `Backtest.run` (s) | Trades |
|---|---:|---:|---:|
| rsi | 37.86 | 29.333 | 187 |
| fifty-two-week-high | 39.49 | 31.732 | 5 |
| sma-cross | 37.61 | 29.605 | 5 |
| momentum-12m | 519.43 | 511.451 | 3 |

Top 10 **sin filtros**, orden descendente por `cumtime` (segundos). Las filas son llamadas anidadas: sus tiempos no se suman.
`click/core.py`, `matplotlib/font_manager.py` y los dos `backtest.py` se distinguen por ruta; las líneas identifican funciones con el mismo nombre.

| Función (archivo:línea) | rsi | fifty-two-week-high | sma-cross | momentum-12m |
|---|---:|---:|---:|---:|
| `click/core.py:1629(__call__)` | 30.277 | 31.863 | 29.741 | 511.595 |
| `click/core.py:1484(main)` | 30.277 | 31.863 | 29.741 | 511.595 |
| `click/core.py:1998(invoke)` | 30.276 | 31.862 | 29.741 | 511.595 |
| `click/core.py:1401(invoke)` | 30.275 | 31.861 | 29.739 | 511.593 |
| `click/core.py:857(invoke)` | 30.275 | 31.861 | 29.739 | 511.593 |
| `quantagent/cli/backtest.py:406(run_backtest)` | 30.275 | 31.861 | 29.739 | 511.593 |
| `quantagent/cli/backtest.py:266(correr_par)` | 30.252 | 31.838 | 29.717 | 511.568 |
| `quantagent/backtesting/backtest.py:326(run)` | 29.333 | 31.732 | 29.605 | 511.451 |
| `quantagent/backtesting/backtest.py:807(_analyze_and_trade)` | 29.183 | 31.560 | 29.461 | 511.228 |
| `matplotlib/font_manager.py:1269(addfont)` (puesto 10 en las primeras tres) | 22.807 | 23.096 | 27.215 | — |
| `quantagent/strategy/momentum_12m_strategy.py:61(_monthly_return)` (puesto 10 en momentum) | — | — | — | 472.138 |

## Causa por estrategia

- **rsi:** `_analyze_and_trade` (puesto 9) reparte su costo entre persistencia e historial; `_get_history_df` suma 6.312 s, `update_candle_tracking` 5.412 s y `generate_signal` 3.207 s.
- **fifty-two-week-high:** `_analyze_and_trade` reconstruye la ventana SQL por vela; `_get_history_df` suma 18.130 s (57% del motor), incluyendo `_rows_to_df` 5.451 s; `generate_signal` solo 3.463 s.
- **sma-cross:** `_analyze_and_trade` (puesto 9) reconstruye el mismo historial SQL, 14.378 s (49% del motor), frente a 1.003 s en `generate_signal` y 0.789 s en `should_exit`; el seguimiento de posición agrega 6.389 s.
- **momentum-12m:** `_monthly_return`, puesto 10, llama al calendario NYSE por fecha; `is_last_session_of_month` suma 471.461 s (92% del motor), con 2.768 llamadas a `valid_days` y `holidays`.

`addfont` (puesto 10 en las primeras tres) corresponde a construir la caché de fuentes al importar Matplotlib; es costo de arranque compartido, no una causa específica de una estrategia.
El perfil extendido se obtiene cambiando `print_stats(10)` por `print_stats(25)`; allí aparecen `_get_history_df` y las funciones de calendario mencionadas, que desglosan el costo de `_analyze_and_trade`.
`_get_history_df` está en `quantagent/backtesting/backtest.py:986`, `_rows_to_df` en `quantagent/data/provider.py:376`, y `update_candle_tracking` en `quantagent/trading/position_monitor.py:100`.
El calendario se crea dentro de `is_last_session_of_month` (`momentum_12m_strategy.py:13`): la caché actual guarda fechas, pero cada fecha nueva crea otro calendario y recalcula sus feriados.
No se reprodujo la lentitud de 115 s en las otras dos estrategias; no hay evidencia para atribuirles la misma causa que a momentum.

## Corrección propuesta para d4cq.2 (sin implementarla)

Reutilizar una instancia de NYSE en `momentum_12m_strategy.py`, manteniendo el cálculo de la sesión siguiente y la caché por fecha; no reemplazarlo por la siguiente vela del dataset.
Estimación: menos de 20 líneas de producción más tests de fin de mes, feriado y mismo resultado con fechas truncadas; entrega total de hasta 100 líneas.
Al evitar los 469.752 s perfilados de `market_calendar.py:556(holidays)`, el techo ideal pasa de 519 a unos 50 s (90% menos); es una estimación, no una mejora medida.
El criterio de 3 veces más rápido debe verificarse sin perfilador, antes/después, junto con métricas y oráculos sin cambios. No propongo tocar SQL en esa entrega.

Validación: cuatro perfiles con exit 0; `timeout 5m scripts/loop/smoke.sh` exit 0, 9 trades, PnL 542.36. Sin tests dirigidos: no se modificaron módulos ni archivos de tests.
