# Datos locales (no versionados)

Esta carpeta contiene archivos y datos locales necesarios para el desarrollo, backtesting y validación de M2 que no se publican en el repositorio git por términos de licencia o tamaño.

**Regla:** Todo dato local nuevo va en esta carpeta (`data/local/`) y se anota en este README en la misma entrega. Ningún dato local se comitea al repositorio git (todos los archivos excepto este `README.md` están ignorados en `.gitignore`).

---

## Inventario de archivos locales

### 1. `etf-1d-2026-10/`
- **Qué es:** Snapshot Parquet de 12 ETFs diarios (2007-01-03 a 2026-10-09) con su `manifest.json`.
- **Quién lo usa:** `backtest run --snapshot etf-1d-2026-10`, `scripts/crossval_compare.py`, `scripts/bt_adapter.py`, oráculos y tests de regresión.
- **Comando de verificación:**
  ```bash
  python -m quantagent.cli data snapshot verify --name etf-1d-2026-10
  ```
- **¿Se puede regenerar igual?:** No. Yahoo recalcula los precios históricos en cada dividendo o split nuevo; recrearlo daría hashes sha256 distintos e invalidaría el manifiesto versionado en `tests/fixtures/snapshots/etf-1d-2026-10.json`. Conservar en solo lectura (`chmod -R 555`).

### 2. `golden/etf-1d-2026-10/rsi-SPY-2007-2018-etf-100.csv`
- **Qué es:** Resultado golden de referencia de trades para RSI sobre SPY diario 2007-2018 con `--costos etf` y 100% de capital (QuantAgent-688, PR #112).
- **Quién lo usa:** `tests/test_golden_snapshot.py`.
- **Comando de verificación / generación:**
  - Verificación:
    ```bash
    python -m pytest -q -m snapshot tests/test_golden_snapshot.py
    ```
  - Generación (si hiciera falta recrear):
    ```bash
    python -m quantagent.cli backtest run --strategy rsi --snapshot etf-1d-2026-10 --symbol SPY --from 2007-01-03 --to 2018-12-31 --costos etf --tamano-posicion 1.0 --out data/local/golden/etf-1d-2026-10/rsi-SPY-2007-2018-etf-100.csv
    ```
- **¿Se puede regenerar igual?:** Sí, siempre que se cuente con el snapshot `etf-1d-2026-10` y el motor conserve su comportamiento. Solo lectura.

### 3. `reserva-accesos.log`
- **Qué es:** Registro de aperturas de la reserva out-of-sample que `backtest run --abrir-reserva` escribe en esta carpeta (`quantagent/cli/backtest.py`). Cada línea registra fecha/hora UTC y el comando ejecutado al acceder a datos a partir de `QUANTAGENT_RESERVA_DESDE` (default 2023-01-01).
- **Quién lo usa:** Auditoría de accesos a la reserva en backtests y revisiones del PM.
- **¿Se puede regenerar igual?:** No se borra ni se regenera. Es el historial acumulativo de accesos a la reserva.

---

## Configuración de entorno

La variable de entorno `QUANTAGENT_SNAPSHOT_DIR` apunta a esta carpeta. `loop/config.env` (`LOOP_SNAPSHOT_DIR`), el valor por defecto de `scripts/loop/run_nightly.sh` y `scripts/loop/try.sh` apuntan a la ruta absoluta de `data/local/` del checkout principal, para que el loop y los worktrees puedan acceder a estos datos sin duplicar archivos.
