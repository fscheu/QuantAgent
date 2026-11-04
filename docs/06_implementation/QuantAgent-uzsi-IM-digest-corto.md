QuantAgent-uzsi: cartera y M5 entregan un resumen de 8 líneas y detalle completo por sección en mensajes de hasta 3900 caracteres, sin recortes; `--enviar-telegram TITULO` envía cada mensaje por stdin a Hermes en orden (HERMES_BIN o PATH) y falla con stderr y código no cero ante un error; stdout conserva todos los mensajes separados. Propuesta de unidades pendiente de confirmación de Fede, sin aplicar:

```diff
--- paper-digest@.service
+++ paper-digest@.service
@@
-... | hermes send -t telegram -s "Paper %i"
+... --enviar-telegram "Paper %i"
--- m5-digest.service
+++ m5-digest.service
@@
-... | hermes send -t telegram -s "Paper %i"
+... --enviar-telegram "Paper %i"
```

Validación: `NUMBA_DISABLE_JIT=1 /home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m pytest tests/test_trading_digest.py tests/test_digest_m5.py tests/test_digest_m5_semanal.py tests/test_cli_reference.py tests/test_cli_help.py -q`: 43 passed, 1 warning. Beads: `bd show QuantAgent-uzsi` no puede abrir SQLite; `bd --no-db --no-daemon show QuantAgent-uzsi` no encuentra el ticket en el JSONL de este worktree. Entrega local pendiente de revisión e integración humana.
