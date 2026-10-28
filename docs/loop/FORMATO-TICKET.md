# Formato de ticket del loop

Formato de la descripción de un ticket de Beads que va a ejecutar un agente implementador. Lo escribe el PM
(`.claude/agents/pm-revisor.md`). El contrato completo del loop (gate, tamaño, resumen de entrega, revisión)
está en `PLAN-CONTINUACION.md` §3 y el prompt del implementador en `loop/PROMPT.md`.

```text
**Contexto:** <qué hay hoy y por qué hace falta el cambio, 1-2 frases>
**Cambio requerido:** <qué tiene que hacer el sistema después de la entrega>
**Criterio de aceptación:**
- [ ] <condición binaria, verificable con un test o un comando>
- [ ] ...
**Archivos relevantes:** <rutas que el agente tiene que leer o tocar>
**Fuera de scope:** <lo que no se toca aunque parezca relacionado>
Tests a correr: <archivos de tests/ de los módulos que toca>
Revisión: leer | decidir | probar
```

Reglas:

- Tamaño: un ticket se estima en ≤100 líneas agregadas+borradas. Si da más, se parte antes de lanzarlo
  (`PLAN-CONTINUACION.md` §3.2).
- Criterio binario: cada ítem se puede marcar PASS/FAIL sin interpretación. "Funciona bien" no es criterio.
- Tipo de revisión: **leer** (el diff alcanza), **decidir** (hay una pregunta cerrada para Fede, que no la
  aprueba el PM) o **probar** (hay que correr algo para ver el resultado).
- Dependencias entre tickets van en Beads (`bd dep add`), no en el texto.
- Ticket de lote: label `lote-<Lnn>` y padre el epic del lote.
- Guías posteriores del PM (aclaraciones, cambios de alcance chicos) van como comentario del ticket. Si una
  guía cambia la línea "Tests a correr:", manda el comentario más nuevo.
