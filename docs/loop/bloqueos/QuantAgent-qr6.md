# Bloqueo · QuantAgent-qr6

Durante la implementación, el test dirigido `python -m pytest -q tests/test_strategy_assembler.py`
terminó con 4 passed. Ese código se retiró porque el smoke requerido no quedó verde.

No se pudo completar el smoke requerido. `scripts/loop/smoke.sh` falla al importar `vectorbt`: Numba
no puede cachear `vectorbt/utils/random_.py` del `.venv` compartido y termina con
`RuntimeError: cannot cache function 'set_seed_nb': no locator available`.

La línea `~/repos/projects/QuantAgent/scripts/loop/try.sh loop/QuantAgent-qr6 -- 'scripts/loop/smoke.sh'`
tampoco se pudo ejecutar en este worktree aislado: la escritura de `.git/FETCH_HEAD` está bloqueada y
`loop/QuantAgent-qr6` todavía no existe en origin.

Destraba: corregir el entorno de dependencias/cache de Numba en la VM y volver a correr el smoke y
`try.sh` sobre la rama de entrega. No se conserva implementación parcial.
