# Calendario mensual del ciclo

El ciclo mensual propone cinco hipótesis el día 1, inicia el embudo el día 8 y revisa paper el día 25. Las fechas se definen en `quantagent/default_config.py`.

`calendario_ciclos(desde, cantidad=6)` genera meses consecutivos, incluso al cruzar de diciembre a enero, y combina sus fechas ISO con el estado persistido del registro: pendiente, abierto o cerrado. Los ciclos cerrados incluyen la cantidad de hipótesis promovidas del recibo.

`quantagent registro calendario [DESDE]` muestra seis ciclos desde el mes indicado (o el mes actual) y el corte consecutivo sin etapa 3 del registro. Al alcanzar seis ciclos, indica que toca revisar el enfoque. El comando no modifica el registro.

Validación agregada en `tests/test_registro.py`: cruce anual, fechas configuradas, estados persistidos, ciclo inválido y salida CLI. No se ejecutó pytest porque el entorno indicado no lo tiene.
