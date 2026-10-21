# Bloqueo: QuantAgent-lrf8

Intenté implementar el uso de `source` en `DataProvider`, los loaders y las dos consultas offline del backtest. Los tests dirigidos nuevos pasaron, pero la verificación requerida no quedó verde; retiré el código para no entregar una regresión sin resolver.

Fallos observados:
- `NUMBA_DISABLE_JIT=1 python -m pytest -q tests/test_data_provider.py -k m1_references_unchanged`: 1 falla, rsi/spy-90d da 339 trades frente a 340; las otras dos referencias pasan.
- `NUMBA_DISABLE_JIT=1 timeout 30m python -m pytest -q -m oraculo`: 3 fallas y 10 errores. `test_golden_rsi_spy_90d` da 226/227 trades; los tests de crossval fallan al convertir `exit_price=''` a float.
- `scripts/loop/smoke.sh`: falla al importar `vectorbt` por caché Numba en el entorno instalado. Con `NUMBA_DISABLE_JIT=1 scripts/loop/smoke.sh` termina con exit 0.

Destraba: confirmar/corregir la referencia de RSI y los fallos de crossval, o decidir explícitamente que estos resultados no bloquean T5. Después se puede retomar el cambio mínimo y verificar de nuevo.
