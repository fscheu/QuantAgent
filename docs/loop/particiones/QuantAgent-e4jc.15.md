# Partición de QuantAgent-e4jc.15

El ticket conecta consultas DB paper, cálculo de pesos con equity, lectura de sombra, detección de sesión faltante y tres chequeos al digest; además la DB actual no declara un modelo de dividendos. La implementación y pruebas de integración exceden el objetivo de 100 líneas, por lo que se divide en tres entregas:

1. **QuantAgent-e4jc.15.1 — consulta paper.** Exponer posiciones con precio, órdenes y dividendos disponibles de la sesión desde SQLite, aislados a PAPER. Binario: una DB sintética devuelve esos datos de sesión y los importes de posiciones son cantidad × precio. Revisión: probar.
2. **QuantAgent-e4jc.15.2 — pesos y sombra.** Integrar pesos paper y `leer_sombra` en el digest por DB y avisar si falta la fecha de la última sesión cerrada. Binario: coincidencia da código 0, divergencia de pesos da código 1 y fecha faltante muestra aviso. Depende de `.1`. Revisión: probar.
3. **QuantAgent-e4jc.15.3 — rechazos y dividendos.** Ejecutar los chequeos de órdenes y dividendos desde el digest DB. Binario: rechazo sin comentario o dividendo sin tratamiento aparece en salida y da código 1; casos atendidos dan código 0. Depende de `.1`. Revisión: probar.

El padre queda `in_progress` hasta que las tres entregas se revisen e integren. No se cambia código en esta partición.
