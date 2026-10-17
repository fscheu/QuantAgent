# Resumen de la tanda

`python -m quantagent.cli backtest hipotesis resumen --desde 2026-02-01 --out resumen.md`
lee estados e intentos de `QUANTAGENT_SNAPSHOT_DIR`. Sin `--desde` incluye todo.
El inicio UTC es inclusivo; se filtran las filas antes de tomar el último estado por hipótesis.
Cuenta ese estado final, agrupa hasta cinco causas sin prefijos como `SPY:` y suma
los intentos del período de las hipótesis incluidas. No modifica los CSV.
Verificación: `NUMBA_CACHE_DIR=/tmp/quantagent-numba-cache python -m pytest -q tests/test_estado.py tests/test_hipotesis_resumen.py`.
