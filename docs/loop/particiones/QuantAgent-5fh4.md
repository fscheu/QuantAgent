# Partición de QuantAgent-5fh4

La función pura única excede el objetivo de 100 líneas al reunir la regla al cierre, la ejecución intravela,
la delegación del backtest y la paridad con paper, además de preservar el estado de trailing y las referencias.
Se divide en dos entregas secuenciales:

1. **QuantAgent-5fh4.1:** extraer a un módulo neutral la regla actual de salida al cierre y hacer que
   `TradingStrategy.should_exit` delegue en ella. Aceptación: pruebas dirigidas conservan razón y precio.
2. **QuantAgent-5fh4.2:** integrar la salida intravela y al cierre en la función pura
   `(posición, vela) -> (razón, precio)`; backtest delega y se verifica la paridad con paper. Depende de
   `QuantAgent-5fh4.1`. Aceptación: test de divergencia, goldens, crossval y oráculos verdes con números invariantes.

Ambas entregas conservan los archivos relevantes del ticket padre y no ajustan las referencias.
