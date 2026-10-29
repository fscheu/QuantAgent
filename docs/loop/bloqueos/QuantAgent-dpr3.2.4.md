# Bloqueo: QuantAgent-dpr3.2.4

Intenté traducir y completar la ayuda de `quantagent/cli/sweep.py`; `profile.py` ya tiene ayudas completas en inglés.

Verificaciones ejecutadas:

- `timeout 300s python -m pytest -q tests/test_backtest_sweep.py tests/test_profile_cli.py`: error de colección en `test_backtest_sweep.py` por importación circular de `quantagent.cli.sweep` y `quantagent.fabrica.hipotesis`.
- `timeout 300s python -m pytest -q tests/test_profile_cli.py`: 25 passed.
- `scripts/loop/smoke.sh`: falla al importar vectorbt porque Numba no puede cachear `set_seed_nb` desde `.venv` (`no locator available`).

Decisión para destrabar: corregir el entorno/importación circular y volver a ejecutar los tests dirigidos y smoke; después aplicar la traducción pendiente.
