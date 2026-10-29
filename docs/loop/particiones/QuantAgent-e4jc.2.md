# Partición de QuantAgent-e4jc.2

Estimación previa: más de 100 líneas. El cambio combina cálculo de pesos por fecha,
una estrategia registrada y una nueva semántica opcional de ejecución en el motor.
No se implementó código.

| Hijo | Objetivo | Aceptación binaria | Estimación | Revisión |
|---|---|---|---:|---|
| QuantAgent-e4jc.2.1 | Pesos objetivo F1 por fecha sin lookahead | En fixture sintético de SPY/IEF, una cartera de referencia 60/40 da SPY 100 % si queda bajo 60 % al cierre previo, IEF 100 % si queda sobre 60 %, y 50/50 si queda exactamente en 60 %; fuera de los últimos cinco retornos, el objetivo es 50/50. No consulta cierres posteriores a la fecha de decisión | 40 líneas | leer |
| QuantAgent-e4jc.2.2 | Registrar la estrategia `rebalanceo-fin-de-mes` y emitir órdenes | El registro instancia el alias; sobre la misma serie sintética, las posiciones objetivo coinciden con .2.1 y la curva fuera de ventana coincide día a día con el benchmark 50/50 | 35 líneas | leer |
| QuantAgent-e4jc.2.3 | Agregar llenado opcional al cierre en backtesting | Con el modo activo, una orden decidida en t se llena al close de t; sin flag, se llena al open de t+1 como hoy; backtests, goldens y oráculos no cambian | 45 líneas | probar |

Orden: .2.1 → .2.2 → .2.3. Cada hijo tiene un alcance estimado menor a 100 líneas y conserva
el alcance y los archivos relevantes del padre. El padre queda `in_progress` hasta
que se integren los tres hijos. La fila X09 de §4 pasa a depender de .2.3.
