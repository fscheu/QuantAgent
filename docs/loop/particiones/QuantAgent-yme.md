# Partición de QuantAgent-yme

La implementación completa supera el objetivo de 100 líneas: el adapter cubre órdenes y fills, estados,
cancelación, cuenta, posiciones, configuración y pruebas del SDK; además requiere registro y una comprobación
real contra paper. Se divide en cuatro entregas secuenciales:

| Hijo | Objetivo | Criterio binario de aceptación | Revisión |
|---|---|---|---|
| QuantAgent-yme.1 | Envío/cancelación y fills de órdenes market/limit contra cliente simulado | Tests cubren envío, cancelación, fill completo/parcial y rechazo sin red | probar |
| QuantAgent-yme.2 | Consultas de cuenta/posiciones y mapeo de estados del contrato Broker | Tests cubren estados, datos consultados y propagación de errores API | leer |
| QuantAgent-yme.3 | Registrar AlpacaBroker en StrategyAssembler con configuración paper | Selector construye paper y bloquea live; tests pasan | probar |
| QuantAgent-yme.4 | Validación de envío y cancelación en paper real | Orden limit lejana se envía y cancela en paper-api con evidencia reproducible | probar |

La prueba real queda en un hijo separado porque requiere credenciales de la VM y acceso al servicio externo.
La rama de implementación no accede a archivos de secretos.
