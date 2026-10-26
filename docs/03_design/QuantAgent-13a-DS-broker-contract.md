# Contrato de `Broker`

`quantagent.trading.paper_broker.Broker` es el límite entre `OrderManager` y el ejecutor. `submit_order(Order)` devuelve la misma orden con estado, cantidad acumulada y fills conocidos; `place_order` conserva el nombre que hoy usa `OrderManager` y delega en `submit_order`. Un rechazo se devuelve como `OrderStatus.REJECTED` con motivo en `comment`; los errores de transporte/API se propagan como excepción. No se interpreta una respuesta pendiente como fill.

## Órdenes y fills

El modelo interno representa `MARKET`, `LIMIT`, `STOP` y `STOP_LIMIT`. Para acciones, Alpaca admite esos cuatro tipos y `TRAILING_STOP`; este último no existe en el modelo y queda fuera del contrato hasta agregarlo explícitamente. El adapter debe rechazar tipos sin representación, no traducirlos silenciosamente. `LIMIT` requiere `price`; `STOP` requiere `stop_price`; `STOP_LIMIT` requiere ambos.

Una orden puede pasar por `PENDING`, `PARTIALLY_FILLED`, `FILLED`, `CANCELLED` o `REJECTED`. Cada fill se agrega una vez a `Order.fills`; `filled_quantity` es la suma acumulada y `average_fill_price` es el promedio ponderado. `cancel_order(id)` informa si la cancelación se aceptó; una orden ya ejecutada no se presenta como cancelada. El modelo actual no conserva ID externo del broker: persistir/reconciliar ese identificador requiere un cambio de esquema separado.

Las órdenes de QuantAgent se expresan en `quantity` (acciones), no en dólares. Alpaca también permite `notional` para órdenes de acciones, pero `qty` y `notional` son excluyentes; el adapter de este contrato envía `quantity` como `qty` y no infiere nocional a partir de `price`, que puede ser una referencia y no un precio ejecutable. Los rechazos por poder de compra se representan con estado `REJECTED` y el texto del broker en `comment`.

## Cuenta, posiciones y datos

`get_account()` devuelve un mapeo de campos del broker, incluyendo efectivo y poder de compra; `get_balance()` queda por compatibilidad con el broker actual. `get_positions()` devuelve posiciones por símbolo. `PaperBroker` no es fuente de cuenta ni posiciones: ambas consultas siguen siendo responsabilidad del portfolio local.

Para operar, se usa Alpaca SIP; Yahoo queda para backtests y desarrollo sin credenciales, según la decisión de [fuente de datos para operar](../04_decisions/QuantAgent-7nle-DC-datos-para-operar-m3.md). La interfaz del broker no obtiene velas ni incorpora streaming. La disponibilidad de tipos, fracciones y restricciones de nocional depende del activo y `time_in_force`; el adapter debe respetar la respuesta de Alpaca y dejar su detalle de restricciones en sus pruebas.

## Implementación actual

`PaperBroker.submit_order` llena la orden completa de inmediato y simula slippage/comisión. No simula fills parciales ni rechazos de buying power; esos estados forman parte del contrato para brokers externos. Las pruebas del adapter Alpaca deben cubrirlos al implementar `QuantAgent-yme`.
