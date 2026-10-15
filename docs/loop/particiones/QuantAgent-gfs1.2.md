# QuantAgent-gfs1.2: referencias con apertura siguiente

Entrega de partición, sin cambios de código ni valores esperados.
Padre en `in_progress` hasta integrar los hijos y resolver el alcance legacy.

## Estimación previa

Estimación total: 205 líneas sin fixtures ni valores esperados.
Golden spy-90d: 65; snapshot y regresión cierre con auditoría: 95;
inventario y decisión sobre controles legacy: 45.
Supera el objetivo de 100: no se implementa en esta corrida.

## Hijos en orden

| Ticket | Objetivo | Aceptación binaria | Revisión |
|---|---|---|---|
| QuantAgent-gfs1.2.1 | Golden spy-90d al default apertura | CSV exacto, métricas verificadas y comando de regeneración | probar |
| QuantAgent-gfs1.2.2 | Snapshot apertura, regresión cierre y tabla de métricas | Control apertura 48/48, cierre 187 / -9141.21 y recálculo independiente coincide | probar |
| QuantAgent-gfs1.2.3 | Inventario y decisión sobre controles legacy | Seis archivos cubiertos, oráculos verdes y decisión del PM para cada referencia al cierre | decidir |

Cada hijo depende del anterior; la cola L8 reemplaza la fila del padre.
Los comandos dirigidos y archivos permitidos están en cada ticket.
El segundo hijo genera el golden nuevo dentro del worktree temporal del test:
no sobrescribe el golden externo, cuya escritura está fuera del aislamiento.

## Alcance pendiente explícito

`crossval_rsi.py` fija `trade_on_close=True`: quitar `--llenado cierre`
de los golden no migra ese port. El tercer hijo identifica qué migrar y
propone hijos de implementación antes de cerrar el padre si hace falta;
no autoriza modificar scripts fuera del alcance original.
Sweep y walkforward quedan pendientes, como permite el comentario del PM
del 2026-10-15. No se cambia el motor ni el punto de control histórico L7.
La regla del PM para aprobar números nuevos sigue siendo apertura 48/48
y cierre 187 / -9141.21 (comentario de Fede del 2026-10-15).
