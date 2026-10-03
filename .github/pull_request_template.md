Cambió: <qué hace ahora el sistema que antes no, en una frase>
Decidir: <pregunta con opciones cerradas, o "nada: merge si el check está verde">
Riesgo: <qué se puede romper y cómo se notaría, o "nada fuera de <archivo>">

── Bloque 1 · leer ──
Ticket: QuantAgent-xxx · T0N · Revisión: leer | decidir | probar
Tamaño: <N> líneas / <M> archivos (límite 150 / 6)
Leé en este orden:
1. <archivo>:<función> — <qué mirar, en una línea>

── Bloque 2 · probar y decidir ──
Probar (copiar en la VM):
~/repos/projects/QuantAgent/scripts/loop/try.sh loop/<ID> -- '<comando>'
Esperado: <salida en ≤5 líneas>
Obtenido por el loop: <salida real, ≤15 líneas>
Verificador independiente: PASS | FAIL por criterio
CI: <verde | rojo | pendiente>
Qué NO se hizo: <lista corta>

Registro — comentá:
R: <ID> vi: <dato concreto de esta entrega> decido: merge | cambio <qué> | descarto <por qué>
