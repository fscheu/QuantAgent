# Entrada M5 rechazada por gatillo superado

`correr_diario` captura excepciones del envío de la orden de entrada, marca la orden como `REJECTED` y persiste el intent como `rechazado_riesgo`. El motivo es `gatillo_superado` cuando el mensaje contiene `must be greater than current price`; para otros errores de envío usa `entrada_rechazada`. No envía la protección y continúa con el símbolo siguiente.

El manejo existente de rechazos de la orden de protección no cambia.

Validación: `NUMBA_CACHE_DIR=/tmp/claude-1000/nd03/numba .venv/bin/python -m pytest tests/test_pipeline_m5.py tests/trading -q`.
