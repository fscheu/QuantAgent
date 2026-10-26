# Diagnóstico del conteo de operaciones F1

`rebalanceo-fin-de-mes` cambia los pesos objetivo durante los últimos cinco días hábiles de cada mes. En el fixture sintético 2007–2018 hay 288 fechas con cambios, equivalentes a 576 cambios de objetivo por activo (288 SPY y 288 IEF). Sin embargo, `filtrar_activo` cuenta 0 trades para SPY y 0 para IEF.

La causa es que la estrategia implementa `posiciones_objetivo` y devuelve `None` en `generate_signal`. Etapa 1 llama `filtrar_activo`, que convierte esas señales genéricas en entradas y salidas mediante `bar_signals`; por eso no ve los rebalanceos. El corte por menos de 30 trades es un artefacto de conteo, no evidencia de ausencia de operaciones.

El conteo correcto debe derivarse de órdenes por activo al pasar de pesos vigentes a pesos objetivo, y sumar las ejecuciones de esos cambios para la cartera. No alcanza con contar fechas de rebalanceo: si el objetivo de un activo no cambia, no genera una operación. En este fixture el conteo esperado sería 576 órdenes y el filtro cuenta 0. No se implementó el arreglo ni se repitió F1.

Reproducción: `python -m pytest -q tests/test_fabrica_filtro.py -k pesos_f1_tiene_rebalances`.
