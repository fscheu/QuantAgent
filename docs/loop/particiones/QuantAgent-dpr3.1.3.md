# Partición de QuantAgent-dpr3.1.3

El test `tests/test_vje_scheduler_heartbeat_backend.py` mezcla cinco pruebas del servicio `DbHandle` de
Streamlit con pruebas de `TradingScheduler` que siguen vigentes. Borrar todas las pruebas en una entrega
superaría el objetivo de 100 líneas; borrar el archivo entero también eliminaría cobertura del scheduler.

Se divide el trabajo en tres entregas secuenciales, registradas en la cola de §4 de
`PLAN-CONTINUACION.md`:

1. `QuantAgent-dpr3.1.3.1`: borrar `app.py` (120 líneas), los dos `__init__.py` (1 línea cada uno) y
   `utils/ui.py` (13 líneas); son cuatro borrados completos y suman 0 líneas en `diff_size.sh`.
2. `QuantAgent-dpr3.1.3.2`: retirar las tres primeras pruebas de `DbHandle` (líneas 176–241 del archivo
   actual: 66 líneas modificadas); conservar el import porque quedan dos pruebas.
3. `QuantAgent-dpr3.1.3.3`: retirar las dos pruebas restantes (líneas 285–311: 27 líneas) y el import
   (1 línea), y borrar `services/db.py` (153 líneas, borrado completo y 0 líneas en `diff_size.sh`). Se
   conservan las pruebas de `TradingScheduler`.

Cada hijo tiene criterio binario y queda acotado a menos de 100 líneas modificadas; los borrados de archivos
completos no suman líneas según `scripts/loop/diff_size.sh`.
