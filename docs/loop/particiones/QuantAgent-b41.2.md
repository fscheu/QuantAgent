# Partición de QuantAgent-b41.2

La implementación completa supera el objetivo de 100 líneas: combina consultas a varias bases SQLite,
filtros de fecha y cartera, heartbeat y prueba CLI, además de documentación. Se divide en entregas:

1. **QuantAgent-b41.2.1:** definir y probar la lectura de posiciones, trades y heartbeats desde el esquema SQLAlchemy.
   Aceptación: la consulta entrega datos compatibles con `build_digest`, aislados por fecha y ambiente paper.
   Revisión: leer.
2. **QuantAgent-b41.2.2:** añadir el comando y la prueba CLI con SQLite temporal.
   Aceptación: fecha, DB y cartera llegan al armado; la salida no está vacía y mide menos de 4000 caracteres.
   Revisión: probar.
3. **QuantAgent-b41.2.3:** documentar el comando y ejemplo reproducible.
   Aceptación: el manual documenta opciones, salida y alerta cuando no hay actividad.
   Revisión: leer.
4. **QuantAgent-b41.2.4:** limitar warnings extensos en el digest y reforzar el caso de 200 trades.
   Aceptación: warnings mayores a 4000 producen digest menor a 4000 con contador de omitidos; test afirma
   `len(text) < 4000` para 200 trades. Revisión: probar.

Secuencia: 2.1 → 2.2 → 2.3 y 2.4 en paralelo con 2.1. Mantener el ticket padre abierto hasta cerrar los hijos.
