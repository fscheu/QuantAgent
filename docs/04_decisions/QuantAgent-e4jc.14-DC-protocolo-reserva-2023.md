# Protocolo para la reserva 2023+

**Ticket:** `QuantAgent-e4jc.14`  
**Decisión de referencia:** [R01, decisión 5](./QuantAgent-R01-DC-decisiones-investigacion-edge.md#decisiones)

La reserva queda limitada a estas hipótesis, escritas antes de abrir los datos:

1. `cartera-transversal`
2. F1 (`rebalanceo-fin-de-mes`), solo si pasa la etapa 2
3. F2 (`tendencia-larga-corta`), solo si pasa la etapa 2

`cambio-de-mes` queda excluida. El tope es tres hipótesis; no se agregan otras a esta reserva.

Para cada hipótesis admitida, aplicar el criterio C0 y reportar M1 (Sharpe en exceso de la tasa corta). Cada intento sobre la reserva cuenta en N, también si el resultado no pasa el criterio.

Se permite una sola apertura de la reserva por hipótesis. Registrar antes de abrir el nombre de la hipótesis y su elegibilidad; después, registrar la fecha, el intento contado en N y el resultado de C0 y M1. No volver a abrir esa hipótesis para repetir o cambiar el análisis.

Este protocolo ejecuta la decisión de R01 sin definir una escalera de incubación. Ver también el [índice de decisiones](./README.md).
