# Partición QuantAgent-y62

El cambio completo requiere comparar posiciones PAPER de la base contra Alpaca, persistir y mostrar divergencias, exponer un comando CLI y registrar un job periódico. La suma de lógica, CLI, scheduler y tests de cuatro escenarios supera el objetivo de 100 líneas.

Entregas, cada una con objetivo acotado y aceptación binaria:

- `QuantAgent-y62.1`: comparación en `PositionMonitor`. Tests cubren concordancia, cantidad distinta y posiciones locales/remotas huérfanas; cada divergencia se persiste como WARNING visible en Logs, sin divergencia no escribe y ninguna ruta corrige. Revisión: probar.
- `QuantAgent-y62.2`: comando CLI sobre la comparación. Con cliente simulado informa divergencias con motivo y no crea ni corrige posiciones. Revisión: probar.
- `QuantAgent-y62.3`: job periódico en `TradingScheduler`, solo para Alpaca paper. El test confirma registro/ejecución y ausencia de mutaciones. Revisión: probar.

Orden: `.1`, luego `.2` y `.3` pueden avanzar en paralelo una vez cerrada `.1`.
