# Aportes, retiros y capital por estrategia (PERF-05 y PERF-06)

Decidido por Fede el 2026-11-01, al revisar las sugerencias del hilo de próximo hito.

## Decisiones

1. **Una cuenta por estrategia, sin caja virtual (PERF-06).** En una cuenta real corre una estrategia a la vez. El rendimiento de una estrategia en ejecución es el de su cuenta; no se reparte capital ni caja dentro de una cuenta. Cierra la pregunta #38 del diseño de Rendimiento.
2. **Flujos manuales por CLI, por cuenta (PERF-05).** Cada aporte o retiro se registra con fecha, monto y tipo. El retorno de la cuenta se calcula ponderado por tiempo, sin contar los flujos como ganancia. Si hay movimientos sin conciliar, el informe queda "no calculable" con el motivo. No se automatiza desde Alpaca.
3. **Escalera de incubación:** solo el mínimo fijo de 3 meses; el resto de los escalones se fija después del primer mes de paper.
4. **CC-06 espera:** no es solo un tema de números, depende de que Fede decida invertir de verdad.

## Consecuencias

- `runPerformance` deja de depender de "capital por estrategia": un sujeto `run` es el informe de su cuenta, con el nombre de la estrategia que corre.
- `accountPerformanceTab` se enciende cuando exista el registro de flujos (PERF-05).
- Cada cuenta ya tiene su propia base, así que no hace falta id de cuenta en los snapshots.
