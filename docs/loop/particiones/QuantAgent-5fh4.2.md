# Partición de QuantAgent-5fh4.2

La integración completa excede el objetivo de 100 líneas porque combina regla única, backtest/replay y
paridad con paper, goldens, crossval y oráculos. Se divide en entregas secuenciales:

1. **QuantAgent-5fh4.2.1:** unir la evaluación intravela y al cierre en una función pura de posición y vela.
   Aceptación: `tests/test_exit_rule_divergence.py` conserva razón y precio en los cuatro casos.
2. **QuantAgent-5fh4.2.2:** delegar los caminos de backtest/replay en esa función y comprobar invariantes.
   Aceptación: divergencia, goldens, crossval y oráculos pasan con números invariantes.

La segunda entrega depende de la primera; ambas conservan las referencias y los precios actuales.
