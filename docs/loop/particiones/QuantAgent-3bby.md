# Partición QuantAgent-3bby: comando de evaluación de reserva

El ticket padre X20b pide elegibilidad, simulacro, apertura única con pre-registro, evaluación C0 + M1, trazas y soporte de pesos objetivo. La estimación conjunta supera 100 líneas; se divide en cuatro entregas secuenciales. No se abre la reserva real en esta partición.

1. **QuantAgent-3bby.1 — elegibilidad y simulacro (≤100 líneas).** Implementar el CLI `backtest hipotesis reserva` para validar lista cerrada, último estado etapa 2 PASA y candado; el modo por defecto calcula fechas sin abrir ni leer el período reservado. Binario: fixtures sintéticos prueban rango y rechazos, candado intacto. Revisión: probar.
2. **QuantAgent-3bby.2 — pre-registro y apertura única (≤100 líneas; depende de .1 en Beads).** Bajo `--abrir`, registrar elegibilidad antes de invocar `candado.abrir(id, 'reserva')` y escribir `reserva-accesos.log`. Binario: el orden queda demostrado y la segunda apertura falla en fixture. Revisión: probar.
3. **QuantAgent-3bby.3 — evaluación y resultado (≤100 líneas; depende de .2 en Beads).** Ejecutar etapa 3 sobre la reserva abierta, C0 y M1 con Sharpe en exceso de DGS3MO; imprimir veredicto y registrar intento. Binario: snapshot sintético produce curva, benchmark, veredicto y registro coherentes. Revisión: probar.
4. **QuantAgent-3bby.4 — pesos objetivo (≤100 líneas; depende de .3 en Beads).** Conectar posiciones objetivo al camino con panel completo. Binario: snapshot sintético obtiene trades > 0 y curva/pesos esperados sin lookahead. Revisión: probar.

Cada hijo debe mantenerse en ≤100 líneas, usar solo snapshots sintéticos y correr los tests listados en su ticket más `scripts/loop/smoke.sh`. El padre queda `in_progress` hasta que los cuatro hijos se revisen e integren.
