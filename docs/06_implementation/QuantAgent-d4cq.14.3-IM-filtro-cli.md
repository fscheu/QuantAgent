# H06.3: evidencia del filtro RSI en la VM

Corrida del 2026-10-16 sobre `etf-1d-2026-10`, exclusivamente 2007–2018.
Hipótesis: `tests/fixtures/hipotesis/rsi-doce-etfs.yaml`; dos combinaciones
(`rsi_period` 7/14, sobreventa 30), doce ETFs, costos `etf`, capital 100.000.
El CLI recorta cada activo antes de admisión y filtro; no evalúa 2019+.

| activo | sharpe_fuera_muestra | trades |
|---|---:|---:|
| SPY | 0.172371 | 161 |
| IWM | -0.00825005 | 233 |
| EFA | 0.283529 | 143 |
| EEM | 0.650597 | 196 |
| TLT | 0.269706 | 133 |
| SHY | -0.464271 | 67 |
| GLD | -0.126471 | 192 |
| DBC | -0.491896 | 171 |
| XLK | 0.192253 | 141 |
| XLE | 0.263057 | 150 |
| XLF | 0.361701 | 200 |
| XLU | 0.349396 | 135 |

Salida real: `PASA (4 de 12 bajo 0,10)`. Tiempo del CLI: **158.06 s**, exit 0.
Registro aislado: **0 → 12** intentos para `rsi-dos-combinaciones-doce-etfs`,
todos etapa `1`, un símbolo distinto por fila, desde 2007-01-01 hasta 2018-12-31.
No se modifica el registro compartido de la VM. El tiempo depende de la caché Numba.

Reproducción desde la raíz del checkout (con `QUANTAGENT_SNAPSHOT_DIR` apuntando
al directorio de snapshots de la VM); el directorio temporal conserva `intentos.csv`:

```bash
d=$(mktemp -d /tmp/h063.XXXXXX); ln -s "$QUANTAGENT_SNAPSHOT_DIR/etf-1d-2026-10" "$d/etf-1d-2026-10"; export QUANTAGENT_SNAPSHOT_DIR="$d" NUMBA_CACHE_DIR=/tmp/quantagent-numba MPLCONFIGDIR=/tmp/quantagent-mpl; echo "Intentos antes: 0"; timeout 30m /usr/bin/time -f "Tiempo: %e s" python -m quantagent.cli backtest hipotesis filtrar tests/fixtures/hipotesis/rsi-doce-etfs.yaml --snapshot etf-1d-2026-10 && python -c "from quantagent.cli.sweep import contar_intentos; print(\"Intentos después:\", contar_intentos(\"rsi-dos-combinaciones-doce-etfs\"))"
```

Verificación dirigida: `NUMBA_CACHE_DIR=/tmp/quantagent-numba MPLCONFIGDIR=/tmp/quantagent-mpl python -m pytest -q tests/test_cli_hipotesis.py tests/test_admision.py tests/test_fabrica_filtro.py`:
**12 passed, 66 warnings in 18.85s**. Sin esas variables, el aislamiento impide
crear la caché de Numba en el entorno instalado; no es un fallo del filtro.
Smoke con las mismas variables: exit 0, 9 trades, PnL 561.18.
No cambian código, umbrales ni candado. El veredicto es provisorio y no habilita operar.
