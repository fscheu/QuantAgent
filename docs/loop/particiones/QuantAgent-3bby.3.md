# Partición QuantAgent-3bby.3: evaluación de reserva

La entrega supera 100 líneas al incluir ejecución del motor 2023+, curva y benchmark 1/N, C0, diagnóstico M1 con DGS3MO, registro e integración sintética. Se divide para preservar el flujo de prueba 2019–2022 y cubrir cada resultado con una prueba reproducible.

1. **QuantAgent-3bby.3.1 — curva, C0 y benchmark.** Ejecutar con motor y CLI existentes la reserva 2023+ ya abierta. Binario: snapshot sintético produce curva y métricas C0 más benchmark 1/N coherentes, sin modificar C0 ni leer datos reales. Revisión: probar.
2. **QuantAgent-3bby.3.2 — M1, veredicto e intento.** Depende de .3.1. En la CLI existente, agregar el diagnóstico de Sharpe en exceso de DGS3MO, salida del veredicto de etapa 3 con C0 intacto y fila de intento. Binario: fixture sintético produce el M1, veredicto e intento coherentes para la reserva abierta, sin leer datos reales. Revisión: probar.

Las filas reemplazan X20b.3 en `PLAN-CONTINUACION.md` §4.
