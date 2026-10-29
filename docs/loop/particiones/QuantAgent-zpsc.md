# Partición de QuantAgent-zpsc

La entrega completa cruza ejecución y validación de posiciones en `OrderManager`/`RiskManager` con el ciclo de monitoreo del scheduler. Sumados los cambios y pruebas de integración, supera el objetivo de 100 líneas. Se divide en dos entregas con aceptación independiente:

1. `QuantAgent-zpsc.1`: abrir un corto con SELL, validarlo por tamaño y concentración, y cerrarlo con BUY dejando la cartera plana. Mantener intacto long-only. Revisión: leer.
2. `QuantAgent-zpsc.2`: integrar señal SHORT y ciclo paper del scheduler sobre el soporte anterior. Abrir, monitorear y cerrar con PaperBroker dejando la cartera plana. Depende de `.1`. Revisión: probar.

El ticket padre queda como contenedor; no se modifica código en esta partición.
