# Bloqueo — QuantAgent-dpr3.7

La guía del embudo y su prueba de comandos se implementaron y los tests dirigidos y `scripts/loop/smoke.sh` pasaron. La revisión independiente rechazó la verificación de CLI dos veces.

En la primera revisión, el test solo comprobaba nombres de grupo/comando. Tras una corrección, usa `shlex` y `make_context`, pero mantiene `resilient_parsing=True` y no comprueba que el recorrido consumió toda la ruta. Click puede suprimir errores de parseo en ese modo, por lo que no demuestra que las invocaciones documentadas sean válidas.

Para destrabar: corregir el test para que el parseo estricto de Click rechace opciones inválidas y para que toda ruta de comando se consuma, o acordar un criterio de aceptación distinto para validar los comandos documentados. El protocolo de esta entrega no permite otra ronda de corrección tras el segundo FAIL.
