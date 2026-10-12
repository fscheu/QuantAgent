# QuantAgent-5f1f: entorno del implementador bloqueado

El cambio solicitado recorta compras `next-open` al efectivo disponible al
ejecutar, incluyendo slippage y comisión. Se intentó en `scripts/bt_adapter.py`
y `tests/test_bt_adapter.py`; los ocho tests pasaron en 110.27 s, incluido
el gap de 100 a 100.5 al 100 % del capital. No se verificó la tabla 4×12.
El código queda en `loop-wip/QuantAgent-5f1f`, commit `6fd2dd84`;
este PR no lo integra. Comando ejecutado: `python -m pytest -q tests/test_bt_adapter.py`.

La preparación de la rama falló con:

`git checkout -b loop/QuantAgent-5f1f`

`fatal: unable to update HEAD`, por `HEAD.lock: Read-only file system` en
`/home/baiscf/repos/QuantAgent/.git/worktrees/20261012T203645Z/`.
Se pudo conservar el código usando metadatos Git aislados dentro del worktree.

La línea obligatoria de Probar también falló con exit 127:

`~/repos/projects/QuantAgent/scripts/loop/try.sh loop/QuantAgent-5f1f -- 'python -m pytest -q tests/test_bt_adapter.py'`

`/home/baiscf/repos/projects/QuantAgent/scripts/loop/try.sh: No such file or directory`.
El checkout real está en `/home/baiscf/repos/QuantAgent`.

La suite sobre la base sin cambios de código también falló (exit 1):
`python -m pytest -q -m "not slow and not api and not oraculo"`.
Resultado: 20 failed, 942 passed, 21 skipped, 15 errors en 679.30 s.
Error: `sqlite3.OperationalError: attempt to write a readonly database`,
en logging, migraciones, cartera y PnL. No se cambió `DATABASE_URL`.

Decisión que destraba: ajustar la ruta de Probar al checkout real y permitir
escribir los metadatos Git y la base de tests al lanzar el implementador. Reintentar
el ticket con ese entorno; verificar tabla 4×12 solo hasta 2018-12-31 y volver
a correr suite, smoke, verificador independiente y la línea exacta de Probar.
