# QuantAgent-gfs1: llenado por defecto

Estado: opción A decidida por Fede (2026-10-14), registrada por el PM en el comentario de 2026-10-15 01:06 UTC de `QuantAgent-gfs1`.
Snapshot `etf-1d-2026-10`, pedido 2007-01-01–2018-12-31 (sesiones desde 2007-01-03); costos `etf`, tamaño 100%, regla confianza, parámetros default, sin orden mínima. PnL en USD; sin precios.
Fuente: `compare` + `compare_control`; 48/48 controles al cierre coinciden. Sharpe tasa 0: apertura = motor al cierre + (adaptador apertura − adaptador cierre), ajuste de L7; no es una corrida del motor propio en apertura. Solo rsi/GLD cambia trades (243→242).
A (elegida): default en apertura siguiente y cierre opcional, alineado con el filtro M3; esta tabla mide sensibilidad, no regenera referencias.
B (descartada): conservar el cierre por defecto; el filtro usa apertura siguiente y el veredicto sigue exigiendo ambos llenados.
Con A cambian los oráculos de M1 y la referencia rsi/SPY 187 / −9141.21 (sensibilidad: 187 / −15620.14). El PM creó `QuantAgent-gfs1.1` (default) y `QuantAgent-gfs1.2` (regeneración), entregas ≤100 líneas. Esta entrega no cambia código. Ver [plan M3](../02_planning/QuantAgent-d4cq-PL-m3-fabrica-de-estrategias.md).
Reproducir tabla (VM con `QUANTAGENT_SNAPSHOT_DIR` configurado; para una fila usar `--strategies rsi --symbols SPY`): `timeout 6000 python -m quantagent.cli backtest compare --strategies rsi,fifty-two-week-high,sma-cross,momentum-12m --symbols SPY,IWM,EFA,EEM,GLD,TLT,SHY,DBC,XLE,XLF,XLK,XLU --snapshot etf-1d-2026-10 --from 2007-01-01 --to 2018-12-31 --costos etf --tamano-posicion 1 --out /tmp/gfs1.csv --trades-dir /tmp/gfs1-trades && timeout 3600 python scripts/compare_control.py --tabla /tmp/gfs1.csv --trades-dir /tmp/gfs1-trades --snapshot etf-1d-2026-10 --costos etf --tamano-posicion 1 --out /tmp/gfs1-control.csv`.

| Estrategia | ETF | Trades cierre | PnL cierre | Sharpe cierre | Trades apertura | PnL apertura | Sharpe apertura |
|---|---|---:|---:|---:|---:|---:|---:|
| rsi | SPY | 187 | -9141.21 | -0.22 | 187 | -15620.14 | -0.46 |
| rsi | IWM | 236 | -2483.54 | -0.03 | 236 | -8302.27 | -0.17 |
| rsi | EFA | 202 | 2307.0 | 0.07 | 202 | -12274.19 | -0.37 |
| rsi | EEM | 230 | 19940.67 | 0.44 | 230 | 4951.99 | 0.16 |
| rsi | GLD | 243 | -9907.46 | -0.22 | 242 | -9505.26 | -0.23 |
| rsi | TLT | 157 | -1610.16 | -0.04 | 157 | -1302.88 | -0.03 |
| rsi | SHY | 10 | -6239.42 | -0.85 | 10 | -5674.73 | -0.76 |
| rsi | DBC | 268 | -3796.6 | -0.07 | 268 | -5413.52 | -0.12 |
| rsi | XLE | 294 | 2519.9 | 0.07 | 294 | -5784.81 | -0.14 |
| rsi | XLF | 247 | -4205.51 | -0.08 | 247 | -18169.74 | -0.48 |
| rsi | XLK | 199 | -17601.25 | -0.44 | 199 | -17239.06 | -0.52 |
| rsi | XLU | 177 | 12021.11 | 0.31 | 177 | 9530.2 | 0.27 |
| fifty-two-week-high | SPY | 5 | 1559.59 | 0.31 | 5 | 1503.02 | 0.3 |
| fifty-two-week-high | IWM | 6 | 1982.25 | 0.22 | 6 | 1946.14 | 0.22 |
| fifty-two-week-high | EFA | 5 | -36.28 | 0.0 | 5 | -324.63 | -0.03 |
| fifty-two-week-high | EEM | 3 | 44.96 | 0.01 | 3 | -49.21 | -0.01 |
| fifty-two-week-high | GLD | 7 | 4290.08 | 0.36 | 7 | 3533.1 | 0.3 |
| fifty-two-week-high | TLT | 9 | 1937.69 | 0.14 | 9 | 2938.15 | 0.21 |
| fifty-two-week-high | SHY | 1 | 1090.09 | 0.83 | 1 | 1050.02 | 0.79 |
| fifty-two-week-high | DBC | 3 | -344.98 | -0.04 | 3 | -96.43 | -0.01 |
| fifty-two-week-high | XLE | 5 | -1146.6 | -0.17 | 5 | -1428.23 | -0.21 |
| fifty-two-week-high | XLF | 9 | 3576.42 | 0.25 | 9 | 2902.68 | 0.21 |
| fifty-two-week-high | XLK | 8 | 1785.52 | 0.17 | 8 | 2586.72 | 0.24 |
| fifty-two-week-high | XLU | 9 | -1851.83 | -0.22 | 9 | -1664.98 | -0.2 |
| sma-cross | SPY | 5 | 135055.28 | 0.67 | 5 | 129894.03 | 0.65 |
| sma-cross | IWM | 6 | 85780.14 | 0.41 | 6 | 72105.29 | 0.37 |
| sma-cross | EFA | 7 | 16052.55 | 0.16 | 7 | 13991.8 | 0.15 |
| sma-cross | EEM | 9 | 1115.41 | 0.08 | 9 | 11323.37 | 0.13 |
| sma-cross | GLD | 6 | 34290.76 | 0.28 | 6 | 31096.31 | 0.26 |
| sma-cross | TLT | 10 | 33764.37 | 0.27 | 10 | 30842.06 | 0.25 |
| sma-cross | SHY | 6 | 3350.18 | 0.49 | 6 | 3019.57 | 0.44 |
| sma-cross | DBC | 7 | -16248.7 | -0.09 | 7 | -15173.73 | -0.08 |
| sma-cross | XLE | 6 | 41325.61 | 0.27 | 6 | 36644.77 | 0.25 |
| sma-cross | XLF | 6 | 130009.52 | 0.56 | 6 | 125325.5 | 0.55 |
| sma-cross | XLK | 6 | 175355.22 | 0.7 | 6 | 167247.35 | 0.68 |
| sma-cross | XLU | 9 | 69170.02 | 0.44 | 9 | 65250.75 | 0.42 |
| momentum-12m | SPY | 3 | 150596.33 | 0.66 | 3 | 148553.54 | 0.66 |
| momentum-12m | IWM | 5 | 112479.5 | 0.48 | 5 | 114466.18 | 0.49 |
| momentum-12m | EFA | 7 | 19532.57 | 0.18 | 7 | 23253.48 | 0.2 |
| momentum-12m | EEM | 9 | 1838.22 | 0.09 | 9 | -2842.92 | 0.07 |
| momentum-12m | GLD | 12 | 19833.22 | 0.18 | 12 | 20178.43 | 0.18 |
| momentum-12m | TLT | 7 | 36495.84 | 0.27 | 7 | 33372.13 | 0.25 |
| momentum-12m | SHY | 3 | 12292.32 | 0.81 | 3 | 11989.48 | 0.79 |
| momentum-12m | DBC | 10 | -29167.32 | -0.16 | 10 | -31009.56 | -0.18 |
| momentum-12m | XLE | 8 | 5557.85 | 0.12 | 8 | 8417.27 | 0.13 |
| momentum-12m | XLF | 5 | 79033.52 | 0.42 | 5 | 74672.42 | 0.41 |
| momentum-12m | XLK | 2 | 217467.97 | 0.73 | 2 | 222158.42 | 0.74 |
| momentum-12m | XLU | 7 | 69311.32 | 0.41 | 7 | 66038.52 | 0.4 |
