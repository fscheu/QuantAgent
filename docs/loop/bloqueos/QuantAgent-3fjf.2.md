# Bloqueo: QuantAgent-3fjf.2

El archivo `tests/trading/test_scheduler_position_monitor.py` ya llama a
`strategy.should_exit`; solo conserva el nombre eliminado en los nombres de
las pruebas. Las 16 pruebas dirigidas pasan.

No se pudo completar la entrega porque `scripts/loop/smoke.sh` termina con
exit 1 al importar `vectorbt`: Numba informa `cannot cache function
'set_seed_nb': no locator available` para el paquete instalado en `.venv`.
`scripts/loop/try.sh` tampoco se pudo reproducir: intenta escribir
`.git/FETCH_HEAD` y falla porque `.git` es de solo lectura; la rama de prueba
no está disponible en este worktree.

Destraba: ejecutar el smoke en un worktree con `.git` escribible y un entorno
donde la caché de Numba de `vectorbt` pueda inicializarse. Luego renombrar las
pruebas y repetir test dirigido, smoke y `try.sh`.
