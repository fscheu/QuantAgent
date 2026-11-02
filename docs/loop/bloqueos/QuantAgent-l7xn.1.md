# Bloqueo QuantAgent-l7xn.1

Intenté agregar el job de CI web y corrí `cd web && npm ci && npm run build && npm test`: build y 62 tests pasaron.

`scripts/loop/smoke.sh` falla al importar el CLI por el entorno Python: Numba no puede cachear `vectorbt.utils.random_.set_seed_nb` desde `site-packages` (`RuntimeError: no locator available`). El fallo ocurre antes de ejecutar una comprobación del workflow.

Decisión para destrabar: corregir o provisionar el entorno Python del smoke en la VM y reintentar QA-01. El cambio del workflow quedó descartado en esta entrega.
