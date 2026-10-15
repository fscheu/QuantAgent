# QuantAgent-gfs1.2.2 — referencias SPY 2007–2018

RSI, snapshot `etf-1d-2026-10`, costos etf, posición 100%, regla confianza.
Los datos privados quedan fuera del repo; el golden explícito se genera en `tmp_path`
y se compara byte a byte con otra corrida del CLI sin `--llenado`.
El adaptador independiente controla cada trade; los escalares siguientes fijan la regresión.

| Referencia | Cierre anterior | Apertura default |
|---|---:|---:|
| Trades | 187 | 181 |
| PnL | -9141.21 | -21841.01 |
| Sharpe | -0.80 | -1.23 |
| TAKE_PROFIT / STOP_LOSS | 81 / 106 | 67 / 114 |

La apertura cambia el precio y el momento de entrada, dimensiona con ese precio y
reescala los niveles relativos de stop/objetivo. También evalúa gaps en la vela de entrada.
Esos cambios alteran las salidas y las próximas entradas: no es solo desplazar los mismos trades.
La caída de PnL viene acompañada de menos objetivos y más stops; no es una medición fuera de muestra.

Reproducir CLI y recálculo (solo ajuste 2007–2018):
```bash
python -m quantagent.cli backtest run --strategy rsi --snapshot etf-1d-2026-10 --symbol SPY --from 2007-01-03 --to 2018-12-31 --costos etf --tamano-posicion 1.0 --out /tmp/gfs122-open.csv --equity-out /tmp/gfs122-equity.csv
python scripts/recalc_metrics.py /tmp/gfs122-open.csv --equity /tmp/gfs122-equity.csv
```
Ambos dieron `Trades: 181`, `Total PnL: -21841.01`, `Sharpe ratio: -1.23`;
el recálculo dio `PnL por trade: 181/181 filas coinciden`.
La regresión cierre corre separada del golden de apertura en `tests/test_golden_snapshot.py`.
Control de las 48 celdas, sin abrir prueba ni reserva:
```bash
python -m quantagent.cli backtest compare --strategies rsi,fifty-two-week-high,sma-cross,momentum-12m --symbols SPY,IWM,EFA,EEM,TLT,SHY,GLD,DBC,XLE,XLF,XLK,XLU --snapshot etf-1d-2026-10 --from 2007-01-03 --to 2018-12-31 --costos etf --tamano-posicion 1.0 --out /tmp/gfs122-compare.csv --trades-dir /tmp/gfs122-trades
python scripts/compare_control.py --tabla /tmp/gfs122-compare.csv --trades-dir /tmp/gfs122-trades --snapshot etf-1d-2026-10 --costos etf --tamano-posicion 1.0
```
Resultado: compare y control exit 0; apertura `48/48`, RSI/SPY `coinciden 181 de 181`.
Pendientes: sweep/walkforward conservan cierre; oráculos legacy se resuelven en `QuantAgent-gfs1.2.3`.
Verificación dirigida: `python -m pytest -q -s tests/test_golden_snapshot.py tests/test_recalc_metrics.py`:
22 passed (377.74 s). `scripts/loop/smoke.sh`: exit 0, 9 trades / PnL 561.18.
