# Bloqueo: QuantAgent-av1p.2

La prueba con tres activos desde efectivo inicial pasó, pero no demuestra un rebalanceo general desde posiciones existentes. La verificación independiente mantuvo FAIL tras una corrección.

`RiskManager.validate_trade` rechaza compras sobre una posición LONG existente y limita cada entrada al 10% del valor del portfolio. `execute_decision_with_order` aplica esa validación, por lo que el rebalanceador no puede ajustar libremente pesos existentes ni abrir posiciones objetivo mayores. La reserva de efectivo tampoco permite resolver por sí sola compras dependientes de ventas anteriores si el broker/portfolio no refleja esos fills.

Para destrabar: definir una ruta autorizada de órdenes de rebalanceo que preserve validación de efectivo, reserva del 5% y prohibición de cortos, y probar una transición real de tres activos con ventas antes de compras.

La implementación incompleta y sus tests fueron retirados; no se modificó `RiskManager`.
