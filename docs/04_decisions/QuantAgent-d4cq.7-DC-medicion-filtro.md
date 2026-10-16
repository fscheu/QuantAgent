# F09.3: medición del filtro, 2007–2018

Medido en la VM el 2026-10-15, sobre `etf-1d-2026-10`: cuatro estrategias M2 × doce ETFs, 48 claves únicas por tabla. Defaults del registro, capital 100.000, costos ETF, tamaño 1. Solo ajuste: ninguna fecha posterior a 2018-12-31. Referencia: [propuesta de motor](QuantAgent-d4cq-DC-motor-de-la-fabrica.md).

## Resultado y decisión pendiente

| Medición | Resultado |
|---|---:|
| Filtro, pared externa (imports, señales, primer JIT, simulación y CSV) | 966,95 s |
| Filtro, pared interna desde STARTED | 965,288828 s |
| bar_signals, suma de 48 celdas | 943,046044 s |
| run_filter, suma de 48 celdas | 16,953472 s |
| Motor propio, apertura, sin control, pared externa | 763,87 s |
| Motor propio, cierre, sin control, pared externa | 812,58 s |
| Spearman Sharpe tasa 0, filtro contra apertura | 0,893329 |
| Spearman Sharpe tasa 0, filtro contra cierre | 0,903133 |
| Trades totales filtro / motor apertura / motor cierre | 2460 / 2612 / 2681 |

Decidir: **A o B**. Con la regla de Fede del 2026-10-15, corresponde proponer **A**: B requiere que el filtro total sea menor que el motor con apertura, y fue 26,6% más lento. El chequeo de calidad del PM (Spearman apertura ≥0,8) pasa. Fede confirma al mergear el lote; esta entrega no selecciona ni cambia motores. La regla vigente reemplaza el umbral histórico de un minuto.

Cinco mayores diferencias absolutas de trades por llenado; delta = filtro − motor:

| Apertura, celda | Delta | Cierre, celda | Delta |
|---|---:|---|---:|
| rsi/SHY | +55 | rsi/XLE | -60 |
| rsi/XLE | -44 | rsi/SHY | +55 |
| rsi/XLF | -32 | rsi/EEM | -35 |
| rsi/IWM | -30 | rsi/XLF | -34 |
| rsi/EEM | -23 | rsi/GLD | -31 |

Las otras tres estrategias tienen delta cero en las 36 celdas de cada llenado. No se diagnosticó la divergencia RSI en esta entrega.

## Reproducir

Desde la raíz del checkout, con la venv activa y `QUANTAGENT_SNAPSHOT_DIR` apuntando al snapshot local. Una corrida por proceso, en serie: filtro, apertura, cierre. Caché Numba inicialmente vacía en `/tmp` (incluye primera compilación); caché de archivos del sistema sin vaciar, sin precalentamiento deliberado. Tiempos observados, sin garantía de repetición exacta; conteos y correlaciones sí deben repetirse. El total incluye carga de datos y arranque; las sumas por fase no los incluyen.

Estos comandos repiten los ejecutados, usando directorios de caché nuevos para cada reproducción. Los CSV y logs quedan en `/tmp`; no contienen evidencia versionada ni se suben precios/trades al PR.

```bash
export MPLCONFIGDIR="$(mktemp -d /tmp/f09-mpl.XXXXXX)" NUMBA_CACHE_DIR="$(mktemp -d /tmp/f09-numba.XXXXXX)" PYTHONPATH=.
timeout 1800 /usr/bin/time -f 'Pared filtro: %e s' python scripts/medir_filtro.py --out /tmp/f09-filtro.csv > /tmp/f09-filtro.log 2>&1
timeout 2400 /usr/bin/time -f 'Pared motor apertura: %e s' python -m quantagent.cli backtest compare --strategies rsi,fifty-two-week-high,sma-cross,momentum-12m --symbols SPY,IWM,EFA,EEM,TLT,SHY,GLD,DBC,XLE,XLF,XLK,XLU --snapshot etf-1d-2026-10 --from 2007-01-01 --to 2018-12-31 --costos etf --tamano-posicion 1 --llenado apertura --out /tmp/f09-apertura.csv > /tmp/f09-apertura.log 2>&1
timeout 2400 /usr/bin/time -f 'Pared motor cierre: %e s' python -m quantagent.cli backtest compare --strategies rsi,fifty-two-week-high,sma-cross,momentum-12m --symbols SPY,IWM,EFA,EEM,TLT,SHY,GLD,DBC,XLE,XLF,XLK,XLU --snapshot etf-1d-2026-10 --from 2007-01-01 --to 2018-12-31 --costos etf --tamano-posicion 1 --llenado cierre --out /tmp/f09-cierre.csv > /tmp/f09-cierre.log 2>&1
tail -2 /tmp/f09-filtro.log
tail -1 /tmp/f09-apertura.log
tail -1 /tmp/f09-cierre.log
python scripts/medir_filtro.py --out /tmp/f09-filtro.csv --apertura /tmp/f09-apertura.csv --cierre /tmp/f09-cierre.csv
python -c 'import pandas as pd; print("Trades filtro / apertura / cierre:", [int(pd.read_csv(p).trades.sum()) for p in ("/tmp/f09-filtro.csv", "/tmp/f09-apertura.csv", "/tmp/f09-cierre.csv")])'
```

Validación: `env MPLCONFIGDIR=/tmp/f09-mpl NUMBA_CACHE_DIR=/tmp/f09-numba python -m pytest -q tests/test_medir_filtro.py`: **15 passed**; `scripts/loop/smoke.sh` con el mismo entorno de caché: **exit 0**, 9 trades / PnL 561,18. Sin esas variables, este sandbox rechaza la caché de Numba en la venv de solo lectura.

## Decisión de Fede (2026-10-15)

**B.** El filtro de las 48 celdas corre en vectorbt. La simulación de vectorbt tarda 17 s; los 943 s de señales son de `bar_signals`, que le pasa a la estrategia toda la historia en cada vela. Condición: QuantAgent-d4cq.7.4 (L9) hace que `bar_signals` pase solo `required_history_bars` velas, como el motor, y con eso el filtro total tiene que quedar por debajo del motor con apertura (764 s) con Spearman apertura >= 0.8. Si no se cumple, se pasa a A sin volver a consultar.
