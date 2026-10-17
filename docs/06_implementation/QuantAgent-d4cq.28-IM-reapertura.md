# QuantAgent-d4cq.28: cortar la reapertura antes del filtro

`correr` consulta `fecha_apertura(id, "prueba")` después de evaluar la admisión y antes
de registrar su resultado. Si ya existe fecha, imprime el aviso y sale con código 1,
sin escribir intentos ni estados, incluso si la nueva admisión rechaza el YAML.
`filtrar` conserva el registro de intentos. La lectura del candado es compartida con
`abierto` y `abrir`; la reserva conserva su comportamiento.

Verificación: `python -m pytest -q tests/test_cli_hipotesis.py tests/test_candado.py tests/test_hipotesis_correr_alarma.py tests/test_hipotesis_correr_error.py` y `scripts/loop/smoke.sh`.
