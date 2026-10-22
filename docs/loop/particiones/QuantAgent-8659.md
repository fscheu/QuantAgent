# Partición QuantAgent-8659

Medición Alpaca paper: compra market DAY de 1.000.000 SPY con buying power de USD 400.000 fue rechazada con HTTP 403 / `40310000` / `insufficient buying power`; no quedó orden abierta. La propuesta es reservar 5% de efectivo al dimensionar compras en el punto compartido `OrderManager._cap_entry_qty`.

Entregas, cada una con límite de 100 líneas:

- `QuantAgent-8659.1`: caso determinista de señal al cierre y fill next-open con gap; test verifica la regla. Criterio: el caso demuestra el límite de efectivo y pasa con la reserva. Revisión: probar.
- `QuantAgent-8659.2`: reserva de 5% compartida para entrada y reversión; preservar ventas. Criterio: compra con gap de hasta 5% no excede efectivo y tests dirigidos pasan. Revisión: leer.

La propuesta requiere decisión de Fede antes de implementar.
