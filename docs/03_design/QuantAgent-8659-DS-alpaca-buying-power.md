# Buying power en compras paper

## Medición real

Medición solicitada por el PM el 2026-10-22 contra `https://paper-api.alpaca.markets`, con la cuenta paper activa. El poder de compra informado por la cuenta fue USD 400.000. Envié una compra market DAY de 1.000.000 acciones de SPY, que supera ese límite ampliamente. Alpaca respondió HTTP 403, código `40310000`, mensaje `insufficient buying power` y `cost_basis` USD 762.060.003. La orden fue rechazada; no quedó orden abierta para cancelar. No se imprimieron ni guardaron credenciales.

Esto confirma que un exceso grande se rechaza, no que Alpaca siempre rechace en vez de llenar parcialmente un exceso pequeño. La medición no caracterizó el umbral exacto ni un gap de apertura real.

## Propuesta

Reservar 5% del efectivo local al limitar compras en `OrderManager._cap_entry_qty`, en el mismo punto compartido por entrada y apertura de reversión. Calcular la cantidad máxima con 95% del efectivo y los costos del broker ya contemplados. Ventas cortas no cambian. La reserva es una protección acotada: no garantiza poder de compra ante gaps mayores a 5%, cambios del buying power o posiciones/cargos que difieran del portfolio local; esos rechazos deben seguir visibles como tales.

No elegir nocional: el contrato vigente expresa órdenes en acciones y el precio de señal no fija el precio ejecutable. No reintentar tras un rechazo en esta etapa: requiere consultar buying power/fills y manejar duplicados y estados del adapter. La regla propuesta se mantiene en un único punto compartido.

## Entregas propuestas

- `QuantAgent-8659.1`: reproducir con un caso determinista el gap entre cierre de señal y apertura siguiente que excede efectivo; test del resultado de la regla. Hasta 100 líneas.
- `QuantAgent-8659.2`: aplicar reserva de 5% en la regla compartida y cubrir entrada, reversión y ventas con tests dirigidos. Hasta 100 líneas.

La reproducción y la regla quedan pendientes de aprobación de Fede (`decidir`).
