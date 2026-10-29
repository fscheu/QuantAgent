# Bloqueo: QuantAgent-dpr3.6.2

Intenté agregar los enlaces al glosario en `criteria.md`, `decisions.md` y `reference/cli/README.md`.

`scripts/loop/smoke.sh` falla antes de validar la CLI: `numba` lanza `RuntimeError: cannot cache function 'set_seed_nb': no locator available` desde `vectorbt/utils/random_.py` instalado en el `.venv`.

Decisión que destraba: corregir la configuración/instalación del entorno de `vectorbt` para que el smoke vuelva a pasar; después aplicar y verificar los enlaces.
