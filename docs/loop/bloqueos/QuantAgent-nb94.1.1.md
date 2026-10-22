# Bloqueo: QuantAgent-nb94.1.1

El ticket exige que `validar_propuesta` requiera `rol` y rechace roles fuera de una lista cerrada.
El comentario más nuevo del padre `QuantAgent-nb94.1` (R01, 2026-10-22) dice que `rol: diversificador`
queda fuera del pedido y de los campos de la propuesta. La definición de §5.2 todavía incluye `rol`.

No implementé cambios de código porque no queda definido si `rol` se elimina por completo o si se
mantiene con un conjunto permitido que excluya `diversificador`. Esa decisión determina el criterio
de aceptación de esta entrega y el formato de las propuestas.

Para destrabar: actualizar §5.2 y el ticket hijo para que coincidan, indicando explícitamente si
`rol` debe ser requerido y cuáles valores admite.
