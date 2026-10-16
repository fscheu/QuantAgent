# H07: punto de control L9

Corridas del 2026-10-16: `hipotesis/rsi-reversion-corta.yaml` y
`hipotesis/sma-tendencia.yaml`, snapshot `etf-1d-2026-10`, solo 2007–2018.
Ambos archivos cargaron y pasaron admisión (incluido truncamiento).
Dos combinaciones por hipótesis; doce ETFs, costos `etf`, capital 100.000.
El CLI recorta las velas antes de admitir y filtrar: no evalúa 2019+.

| Activo | RSI Sharpe fuera de muestra | SMA Sharpe fuera de muestra |
|---|---:|---:|
| SPY | 0.172371 | 0.371159 |
| IWM | -0.00825005 | 0.597319 |
| EFA | 0.283529 | -0.235299 |
| EEM | 0.650597 | -0.204756 |
| TLT | 0.269706 | 0.454131 |
| SHY | -0.464271 | 0.491952 |
| GLD | -0.126471 | -0.402143 |
| DBC | -0.491896 | -0.124985 |
| XLK | 0.192253 | 0.683665 |
| XLE | 0.263057 | 0.00709314 |
| XLF | 0.361701 | 0.479563 |
| XLU | 0.349396 | 0.480644 |

RSI: **PASA (4 de 12 bajo 0,10)**, **164.95 s**, exit 0.
SMA: **PASA (5 de 12 bajo 0,10)**, **150.87 s**, exit 0.
Registro aislado `/tmp/h07.40xeWA`: RSI **0 → 12**, SMA **0 → 12**;
24 intentos etapa `1`, uno por activo e hipótesis. Los tiempos incluyen admisión.

Qué se aprendió:
1. Ambas hipótesis superan el umbral provisorio; esto no valida operar ni habilita etapas 2–4.
2. El resultado depende del activo: GLD y DBC quedan bajo 0,10 en ambas.
3. SMA genera solo 3–10 trades por activo, frente a 67–233 de RSI; pasar no prueba estabilidad estadística.

## Reproducción

Desde la raíz, con `QUANTAGENT_SNAPSHOT_DIR` apuntando a los snapshots de la VM.
El registro temporal conserva los intentos y no modifica el compartido.

```bash
d=$(mktemp -d /tmp/h07.XXXXXX); ln -s "$QUANTAGENT_SNAPSHOT_DIR/etf-1d-2026-10" "$d/etf-1d-2026-10"; export QUANTAGENT_SNAPSHOT_DIR="$d" NUMBA_CACHE_DIR=/tmp/quantagent-numba MPLCONFIGDIR=/tmp/quantagent-mpl; echo "Intentos antes: 0 por hipótesis"; for h in rsi-reversion-corta sma-tendencia; do timeout 30m /usr/bin/time -f "Tiempo: %e s" python -m quantagent.cli backtest hipotesis filtrar hipotesis/$h.yaml --snapshot etf-1d-2026-10 || exit; done; python -c "from quantagent.cli.sweep import contar_intentos; print([(h, contar_intentos(h)) for h in (\"rsi-reversion-corta\", \"sma-tendencia\")])"
```

Tests dirigidos: `NUMBA_CACHE_DIR=/tmp/quantagent-numba MPLCONFIGDIR=/tmp/quantagent-mpl python -m pytest -q tests/test_hipotesis.py`: **5 passed, 1.92 s**.
Smoke: mismas variables y `timeout 10m scripts/loop/smoke.sh`: exit 0,
9 trades / PnL 561.18. No cambian código, umbrales ni candado.
`try.sh` se intentó: exit 2 al escribir `.git/FETCH_HEAD` (solo lectura).
El comando interior se ejecutó directamente en este worktree.
