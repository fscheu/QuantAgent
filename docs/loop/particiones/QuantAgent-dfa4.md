# Partición de QuantAgent-dfa4

El cambio pedido supera el objetivo de 100 líneas al incluir la mudanza de la función, su reexportación,
la integración en paper y la prueba de igualdad. Se divide en dos entregas revisables:

1. **QuantAgent-dfa4.2:** mover `check_intrabar_stops` a un módulo neutral y reexportarla desde `backtest.py`.
   Aceptación: `test_intrabar_stops.py` y crossval intravela pasan con los mismos resultados.
2. **QuantAgent-dfa4.1:** paper evalúa la función en la última vela cerrada y cierra por el camino existente.
   Depende de dfa4.2. Aceptación: el caso 2 cierra a 90 en backtest y paper cuando la vela cierra a 95.

Cada hijo queda por debajo de 100 líneas y conserva los archivos relevantes del ticket original.
