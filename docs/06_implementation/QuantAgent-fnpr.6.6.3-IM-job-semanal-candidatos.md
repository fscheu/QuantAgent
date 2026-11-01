# M5-55c: candidatos vigentes y sombra semanal

Implementación de [la agenda M5, §6.7](../03_design/QuantAgent-M5-DS-motor-top-down.md#67-agenda), ticket `QuantAgent-fnpr.6.6.3`.

`correr_semanal` conserva el cálculo, CSV y retorno del régimen. Después ejecuta `candidatos_en_vivo` con los habilitados de esa fecha; en risk-off deja la selección vacía. Agrega los parámetros opcionales `base_dir="local-data"` (snapshots) y `composicion="sp500-2026-10"`. El CLI no requiere cambios.

`fundamentales_vigente(base_dir)` recorre los snapshots diarios de más reciente a más antiguo y elige el primero que pasa `verify_snapshot`; si ninguno pasa devuelve `fundamentales-sp500-2026-10`. Si falta ese respaldo o falla el screener, registra el motivo con símbolo vacío y conserva H1 como H3. No refresca fundamentales ni consulta servicios adicionales.

Los archivos nuevos se guardan junto a `regimen_path`, por defecto dentro de `local-data/`:

- `m5-candidatos-semanales.csv`: columnas `COLS_CANDIDATOS`.
- `m5-candidatos-descartes.csv`: `fecha,simbolo,motivo`.
- `m5-sombra-semanal.csv`: `fecha,activo,peso_h1,peso_h3`, incluidos pesos cero de los 14 ETFs para representar efectivo.

Los tres reemplazan todas las filas de la fecha de decisión, incluso ante una selección vacía. Mantienen las demás fechas. La nota semanal sigue el flujo existente; no hay órdenes ni cambios al job diario.

La falla de idempotencia era real: el lector CSV predeterminado de pandas perdía precisión decimal en las semanas conservadas. Al recalcular una semana se recuperaban sus decimales originales, mientras las otras se degradaban otra vez. El régimen ya tenía ese comportamiento antes de este ticket; los nuevos CSV lo repetían. Ambos upserts ahora leen con `float_precision="round_trip"`, sin cambiar las columnas ni el formato inicial del régimen. Los nuevos CSV usan orden estable por fecha para conservar el orden de candidatos y activos dentro de cada semana.

La prueba también detectó que `guardar_narrativa` movía al final una nota idéntica al repetir una semana después de cargar otra anterior. Ahora retorna sin escribir si ya existe exactamente una fila con la misma clave, tipo y texto. Los cambios de texto siguen reemplazando la nota. La aserción byte a byte se conserva para todos los CSV, incluida la narrativa, tanto en la repetición inmediata como con otra semana cargada.

La sombra usa `señales_transversales` con la métrica, máscara y regla de H1, `pesos` semanal y `pesos_h3`, igual que `scripts/m5/h3.py`. Para registrar la decisión del viernes, agrega la próxima sesión NYSE sin precios: el rebalanceo usa la selección vigente y la volatilidad hasta el cierre de decisión. H3 sustituye los ETFs sectoriales con candidatos de rank ≤3; sin candidatos mantiene el ETF. EEM, EWZ y SPY permanecen como ETFs. En risk-off todos los pesos son cero.

Validación (intérprete indicado por el ticket; sin red, caché y temporales dentro del worktree):

```bash
mkdir -p .cache/numba .cache/tmp
NUMBA_CACHE_DIR="$PWD/.cache/numba" TMPDIR="$PWD/.cache/tmp" PYTHONPATH=. /home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m pytest -q tests/test_pipeline_m5.py tests/test_regimen.py tests/test_candidatos_vivo.py tests/test_cartera*.py tests/test_cli_*.py tests/test_narrativa_m5.py
NUMBA_CACHE_DIR="$PWD/.cache/numba" TMPDIR="$PWD/.cache/tmp" PATH=/home/azureuser/repos/projects/QuantAgent/.venv/bin:$PATH PYTHONPATH=. bash scripts/loop/smoke.sh
git diff --check
```

Los tests usan reloj inyectado, proveedor de fixture y snapshots temporales; comparan el CSV del régimen byte a byte, candidatos con el screener real y pesos con las mismas funciones del backtest. Cubren repetición, reemplazo por selección vacía, falla del proveedor, ausencia de fundamentales, snapshots corruptos y efectivo en risk-off. El LLM externo se desactiva en los tests del pipeline.

El primer intento de la suite sin `NUMBA_CACHE_DIR` terminó con cuatro errores de colección por falta de ubicación escribible para la caché de Numba en `vectorbt`; no fue una falla de lógica del ticket.

Resultado final: **124 passed, 0 failed, 303 warnings**, en 446,55 s, exit 0. Incluye todos los archivos pedidos y `tests/test_narrativa_m5.py` por el ajuste al upsert de notas. Smoke: exit 0, 9 trades, win rate 55,56 %, PnL 561,18; matplotlib usó un temporal dentro del worktree al no poder escribir su caché habitual. `git diff --check`: exit 0.

Huecos: no se ejecutaron proveedores ni LLM externos, por la restricción de no usar red. La equivalencia con el script histórico se prueba con sus funciones de cálculo sobre precios sintéticos, sin correr el backtest histórico completo.

Entrega sin stage ni commit, pendiente de revisión e integración por la sesión revisora. Sin desvíos de alcance.
