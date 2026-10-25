# Partición de QuantAgent-b41

El digest reúne lectura de datos de paper, composición de varias secciones y una entrada de CLI documentada;
la estimación supera el objetivo de 100 líneas. Se divide en dos entregas secuenciales:

1. **QuantAgent-b41.1:** armar el texto en una función pura y cubrir las secciones con datos seed en SQLite.
   Aceptación: tests verifican posiciones por cartera, trades/P&L, errores y warnings, heartbeat con alerta
   tras más de un día hábil, reconciliación, truncamiento explícito bajo 4000 caracteres y día sin actividad.
   Revisión: probar.
2. **QuantAgent-b41.2:** exponer la función con `quantagent trading digest --date YYYY-MM-DD` y documentar el
   comando. Depende de b41.1. Aceptación: CLI funcional, salida de texto plano bajo 4000 caracteres y
   reproducción mediante `try.sh` con smoke en verde. Revisión: probar.

Ambas entregas respetan los archivos relevantes y el alcance del ticket padre.
