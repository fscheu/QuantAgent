# QuantAgent-fnpr.6.6.5 — Digest M5 con acciones

El digest M5 agrega candidatos por sector/rank, sectores habilitados sin candidatos con motivos de descartes, conteos por símbolo de descartes por precio/fundamentales y hasta tres ejemplos. Lee los CSV junto a la DB seleccionada, filtra por el régimen vigente y no recupera candidatos de semanas anteriores cuando la selección vigente está vacía. Las posiciones y los intents muestran cantidades con hasta cuatro decimales; conserva stops, heat y el mensaje de ausencia de actividad.

`digest_m5_semanal.py` recibe proveedor y raíz de snapshots inyectables. El CLI construye el mismo `DataProvider` Alpaca del ciclo M5 y carga las credenciales del bucket cuando existe el CSV de sombra. No depende del job diario de M5-55d. La reconciliación existente recorre todos los símbolos y compara cantidades con `Decimal`: no requiere cambios para acciones.

La sombra encadena retornos ponderados cierre a cierre entre decisiones semanales y hasta la última sesión NYSE cerrada. Excluye decisiones en el cierre final, conserva efectivo en pesos cero y no rellena precios ausentes. Un precio de extremo faltante invalida esa estrategia/intervalo y el acumulado que lo contiene, sin afectar las demás estrategias. 1/N reparte entre los 14 ETFs de M5. Si no hay decisión para el inicio de la semana actual, el semanal indica `sin dato`. El pipeline conserva equity = capital inicial + P&L no realizado; usa `EquitySnapshot` en las fechas de referencia y muestra `sin dato` si faltan, sin escribir snapshots.

Límites de los artefactos existentes: los retornos cierre a cierre incluyen el gap hasta la apertura siguiente y no modelan costos; no son una simulación de fills en esa apertura. Los CSV no registran el snapshot de fundamentales usado por el screener: se informa el snapshot vigente elegido por `fundamentales_vigente`, no una atribución histórica. El respaldo mensual se identifica por mes (la edad cuenta desde el primer día). Los descartes no tienen sector: fuera de una falla global, los motivos mostrados para sectores sin candidatos son los descartes de la semana. No se modifica el pipeline para persistir información adicional.

Manual: [operations.md](../manual/operations.md), sección Daily digest, actualizado. La referencia CLI se regeneró sin diferencias: no cambió la ayuda ni las opciones.

Validación sin red, con temporales dentro del worktree:

```bash
mkdir -p .cache/numba .cache/tmp .cache/matplotlib
PYTHONPATH=. NUMBA_CACHE_DIR=$PWD/.cache/numba TMPDIR=$PWD/.cache/tmp MPLCONFIGDIR=$PWD/.cache/matplotlib PYTHONDONTWRITEBYTECODE=1 /home/azureuser/repos/projects/QuantAgent/.venv/bin/python scripts/gen_cli_reference.py --output-dir docs/manual/reference/cli > .cache/cli-reference.log
PYTHONPATH=. NUMBA_CACHE_DIR=$PWD/.cache/numba TMPDIR=$PWD/.cache/tmp MPLCONFIGDIR=$PWD/.cache/matplotlib PYTHONDONTWRITEBYTECODE=1 /home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m pytest -q --basetemp=$PWD/.cache/pytest-final -p no:cacheprovider tests/test_digest*.py tests/test_trading_digest.py tests/test_cli_help.py tests/test_cli_reference.py
PYTHONPATH=. NUMBA_CACHE_DIR=$PWD/.cache/numba TMPDIR=$PWD/.cache/tmp MPLCONFIGDIR=$PWD/.cache/matplotlib PYTHONDONTWRITEBYTECODE=1 /home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m pytest -q --basetemp=$PWD/.cache/pytest-cli -p no:cacheprovider tests/test_cli_*.py
PYTHONPATH=. NUMBA_CACHE_DIR=$PWD/.cache/numba TMPDIR=$PWD/.cache/tmp MPLCONFIGDIR=$PWD/.cache/matplotlib PYTHONDONTWRITEBYTECODE=1 /home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m ruff check --no-fix quantagent/digest_m5.py quantagent/digest_m5_semanal.py quantagent/cli/reconcile.py tests/test_digest_m5_semanal.py
PATH=/home/azureuser/repos/projects/QuantAgent/.venv/bin:$PATH PYTHONPATH=. NUMBA_CACHE_DIR=$PWD/.cache/numba TMPDIR=$PWD/.cache/tmp MPLCONFIGDIR=$PWD/.cache/matplotlib PYTHONDONTWRITEBYTECODE=1 bash scripts/loop/smoke.sh
git diff --check
PYTHONPATH=. NUMBA_CACHE_DIR=$PWD/.cache/numba PYTHONDONTWRITEBYTECODE=1 /home/azureuser/repos/projects/QuantAgent/.venv/bin/python -c 'import pathlib, shutil; root = pathlib.Path.cwd(); assert root.name == "QuantAgent-fnpr.6.6.5"; shutil.rmtree(root / ".cache"); shutil.rmtree(root / ".ruff_cache")'
```

Los tests nuevos ejecutan la lectura de CSV reales temporales, la elección de snapshot verificado, la consulta SQL de posiciones/intents y el cálculo ponderado sobre series sintéticas, sin mocks. Cubren selección de fechas/ranks, fallback sectorial/global, descartes y antigüedad; H1 +10% semanal/+21% acumulado, H3 0%/+15%, pérdida de una pata, exclusión de la decisión del viernes final y acciones fraccionarias sin actividad posterior. No existen `tests/test_m5_53*.py` en esta base. Sin consultas reales a Alpaca por la restricción de red.

Digest + ayuda + referencia: **32 passed, 0 failed, 1 warning**, 51,92 s. Suite ampliada `tests/test_cli_*.py`: **70 passed, 0 failed, 276 warnings**, 531,82 s. Ruff: `All checks passed!`. `git diff --check`: exit 0. Smoke: exit 0, 9 trades, win rate 55,56 %, PnL 561,18. La primera pasada de Ruff señaló únicamente orden/formato de imports; corregido y verificado de nuevo.

Entrega sin stage ni commit, pendiente de revisión e integración. No se usó `bd`, no se modificó el ticket, no se hizo push ni se abrió PR.
