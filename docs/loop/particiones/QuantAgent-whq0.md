# Partición de QuantAgent-whq0

El alcance completo excede el objetivo de 100 líneas: requiere derivar tres advertencias estructurales,
propagarlas a tres salidas y demostrar que no cambian el veredicto de admisión. Se divide en entregas chicas:

| Hijo | Objetivo | Criterio binario | Revisión |
|---|---|---|---|
| QuantAgent-whq0.1 | Mostrar advertencias en etapa 0 y pedido, sin bloquear | Ambas salidas incluyen las advertencias y el pase se mantiene posible | leer |
| QuantAgent-whq0.2 | Derivar `desvio_de_la_cita`, `muestra_solapada` y `overlay_solo_largo` desde metadatos y composición | Cada condición activa emite su código y la propuesta fiel sin condiciones no emite ninguno | leer |
| QuantAgent-whq0.3 | Mostrar advertencias en informe y cubrir las cuatro carteras conocidas | El informe las muestra; las cuatro carteras disparan lo esperado y una propuesta fiel no | probar |

La tabla de cola de `PLAN-CONTINUACION.md` reemplaza S07 por estos tres hijos. El padre queda como contenedor.
