# QuantAgent-gfs1.1.2: separar CLI, referencias y default

Estimación previa: 160–220 líneas entre propagación del CLI, tests de
ambos comandos, adaptación de referencias y documentación. Supera 100.
El motor ya acepta fill, pero el CLI no lo expone. Hay numerosos tests que
construyen Backtest sin fill y esperan ejecución al cierre; cambiar el default
requiere revisar esas expectativas además de los golden y oráculos.

Entregas en orden (criterios completos y comandos en BEADS):

1. QuantAgent-gfs1.1.2.1, 85 líneas, revisión probar: exponer ambos valores
   en run/compare conservando cierre por defecto. PASS si se propagan al
   motor y snapshot y un valor inválido sale 2.
2. QuantAgent-gfs1.1.2.2, 95 líneas, revisión probar: fijar cierre en las
   referencias y tests dependientes. PASS si oráculos pasan sin cambiar
   números y golden RSI/SPY 2007–2018 exige 187 / -9141.21 en cierre.
   Si el inventario excede 100 líneas, particionar antes de editar.
3. QuantAgent-gfs1.1.2.3, 80 líneas, revisión probar: activar apertura por
   defecto. PASS si run/compare sin flag usan apertura, cierre conserva
   resultados, última señal no abre posición y replay no cambia.

Cada hijo depende del anterior. gfs1.2 sigue dependiendo del padre;
el padre se cierra cuando se hayan integrado todos sus hijos.
La reversión en apertura sale al open y señala la nueva entrada al cierre
de esa vela: demora una vela extra. Se documenta, no se modifica.
No se cambia código ni se regeneran referencias en esta partición.
