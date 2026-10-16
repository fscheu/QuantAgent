# QuantAgent-d4cq.7: medición del filtro en tres entregas

Partición previa a implementar: estimación 240 líneas, superior al objetivo de 100.
El padre sigue `in_progress`; esta entrega no ejecuta ni modifica el motor.

| Hijo | Objetivo | Líneas estimadas | Aceptación binaria | Revisión |
|---|---|---:|---|---|
| QuantAgent-d4cq.7.1 | Script de señales y simulación sobre las 48 celdas | 90 | CSV de 48 claves únicas, métricas y tiempos separados; test demuestra exclusión de velas posteriores a 2018 | leer |
| QuantAgent-d4cq.7.2 | Comparación contra CSV del motor con apertura y cierre | 90 | Spearman +1/-1, empates, cruce por clave y rechazo de tablas incompletas; ambos Spearman y top cinco por llenado | leer |
| QuantAgent-d4cq.7.3 | Corrida VM e informe de decisión reproducible | 60 | Tiempos comparables, ambos Spearman, top cinco y comandos reales reproducidos por el PM | decidir |

Los hijos dependen en cadena; reemplazan F09 en la cola L8.
Cada ticket contiene archivos permitidos, criterios y tests dirigidos.
No se agrega infraestructura: se reutilizan snapshot, bar_signals, run_filter
y los CSV de backtest compare. No se usa compare_control para esta medición.

## Regla vigente (comentarios del padre)

Comparar contra `sharpe_tasa_0` del motor corrido directamente con cada llenado.
Medir pared de `compare` con apertura sin control y total del filtro, además de
señales y simulación. Incluir arranque/JIT y documentar caché para no ocultar costo.
B requiere filtro más rápido que motor apertura; desaparece el umbral de un minuto.
El PM mantiene Spearman apertura >= 0,8: si falla, escala; Fede confirma al integrar.
Solo 2007–2018, costos ETF y tamaño 1 iguales entre motores. Nada de prueba ni reserva.
El informe incluye los dos Spearman y cinco mayores diferencias de trades por llenado.
Los tiempos varían entre corridas; métricas y conteos deben reproducirse.

## Verificación de esta partición

Leer este archivo, las tres filas F09 y `bd show` de los hijos.
No hay tests nuevos ni cambios de comportamiento: `scripts/loop/smoke.sh` debe pasar.
Los tests del script se agregan con el primer hijo, no en esta entrega documental.
