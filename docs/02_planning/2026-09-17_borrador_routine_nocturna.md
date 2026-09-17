# Borrador: routine nocturna del loop de desarrollo AI

Fecha: 2026-09-17 · Estado: **borrador, no creada** · Dueño de la decisión: Fede
Contexto: `2026-09-16_retrospectiva_autodev_y_plan_oficina_inversion.md` §2

---

## 1. Alcance

**Hace:**
- Una vez por noche (lun–vie 01:00 ART), toma **un** ticket BEADS con label `ai-ready` que no tenga bloqueos
  abiertos, en orden de prioridad y antigüedad.
- Trabaja en una rama nueva desde `origin/main` en un checkout limpio.
- Implementa, corre la suite, hace una verificación con contexto fresco, y abre **un PR** con el contrato
  del template (`## Ticket`, `## Cómo verificar`, `## Qué NO se hizo`).
- Deja el ticket en `ai-in-progress` con la URL del PR en un comentario, y avisa por Telegram.

**No hace:**
- No mergea. No toca `main`. No usa `--force`. No borra ramas.
- No toma más de un ticket por corrida, aunque le sobre tiempo.
- No toca `~/.hermes/`, `~/secrets/`, servicios systemd ni crons.
- No abre PR si no llegó a un estado verificable: comenta el bloqueo en el ticket, pone `ai-blocked` y termina.
- No toma tickets `fede-decision`.

**Kill switch:** deshabilitar la routine en https://claude.ai/code/routines, o quitar todos los labels `ai-ready`.

## 2. Entorno: por qué el bridge de la VM y no la nube pura

| Necesidad del loop | Entorno cloud "Default" | Entorno bridge `vm-ai-clawdbot:QuantAgent:f4a3` |
|---|---|---|
| `bd` + wrapper `bd_safe.sh` | no existe | sí |
| `.venv` con deps instaladas | habría que instalar cada vez | sí |
| Suite de tests | corre (SQLite en memoria) | corre |
| `gh` autenticado para abrir PR | requiere configurar token | sí (`gh auth status` OK) |
| Notificación Telegram vía Hermes | no | sí (`hermes-cli` skill) |

Decisión propuesta: **bridge de la VM**, con la condición de que el prompt cree un **checkout limpio por corrida**
(`git worktree add` desde `origin/main` en `/tmp/ai-loop/<fecha>`), que es lo que evita el modo de falla
"root checkout sucio" del loop anterior. El worktree se borra al final de la corrida.

## 3. Schedule

- Lun–vie 01:00 ART = `0 4 * * 1-5` UTC. El PR espera a Fede a la mañana.
- Modelo propuesto: `claude-opus-5` para la implementación (matemática de métricas y engine); el subagente
  verificador puede ser el mismo modelo con contexto fresco.

## 4. Prompt (autocontenido)

```text
Sos el agente nocturno de desarrollo de QuantAgent. Trabajás solo, sin humano disponible. Tenés un
presupuesto de UNA tarea y ~2 horas. Tu salida es un pull request revisable en 20 minutos, o un
comentario de bloqueo. Nada más.

REGLAS DURAS
- Nunca commitees ni pushees a main. Nunca uses --force. Nunca borres ramas.
- No toques ~/.hermes, ~/secrets, systemd, crons, ni archivos fuera del repo.
- Un solo ticket por corrida. Si lo terminás antes, no tomes otro.
- Si algo te impide llegar a un estado verificable, parás y reportás. No improvises alcance.
- Leé y respetá /home/azureuser/repos/projects/QuantAgent/AGENTS.md y CLAUDE.md.

PASO 0 — Checkout limpio
  cd /home/azureuser/repos/projects/QuantAgent
  git fetch origin main
  RUN=$(date -u +%Y%m%dT%H%M%SZ)
  git worktree add /tmp/ai-loop/$RUN origin/main
  cd /tmp/ai-loop/$RUN
  source /home/azureuser/repos/projects/QuantAgent/.venv/bin/activate
  Usá SIEMPRE el wrapper para BEADS: BD="/home/azureuser/repos/agents/autodev-runner/scripts/bd_safe.sh"
  (con BD_SAFE_REPO=/home/azureuser/repos/projects/QuantAgent para que escriba en el repo principal).

PASO 1 — Elegir el ticket
  $BD ready --label ai-ready
  Tomá el de mayor prioridad (P0 < P1 < P2); a igual prioridad, el más antiguo.
  Si no hay ninguno: terminá con el mensaje "sin tickets ai-ready" y borrá el worktree.
  $BD show <ID>   → leé Contexto, Cambio requerido, Criterio de aceptación, Archivos relevantes,
                    Fuera de scope y "Verificación de Fede".
  $BD update <ID> --status in_progress
  $BD label add <ID> ai-in-progress ; $BD label remove <ID> ai-ready

PASO 2 — Implementar
  git checkout -b feature/<ID>-<slug-corto>
  Implementá exactamente el "Cambio requerido". No refactorices nada fuera de "Archivos relevantes".
  Escribí los tests que pide el criterio de aceptación. Tests que fallan si la lógica real se rompe;
  nada de mocks excesivos ni asserts triviales.
  pytest -q -m "not slow and not api"   → tiene que dar 0 failed.
  Corré a mano el comando de la sección "Verificación de Fede" y guardá la salida real.
  Commits chicos con mensaje "feat|fix|chore(<ID>): ...".

PASO 3 — Verificación con contexto fresco (obligatoria)
  Lanzá un subagente NUEVO (sin tu historial) y pasale SOLO: el texto del ticket (bd show) y el diff
  (git diff origin/main...HEAD). Pedile que:
    a) corra la suite y el comando de verificación de Fede desde cero,
    b) evalúe cada casilla del criterio de aceptación como PASS/FAIL con evidencia,
    c) busque cambios fuera de "Archivos relevantes" o dentro de "Fuera de scope".
  Si devuelve algún FAIL: corregí UNA vez y repetí el paso 3. Si sigue FAIL: PASO 5 (bloqueo).

PASO 4 — PR
  git push -u origin feature/<ID>-<slug>
  gh pr create --base main --title "<ID>: <título del ticket>" --body-file <archivo> con:
    ## Ticket
    <ID> — <título>. Criterios: lista con [x]/[ ] y una línea de evidencia por cada uno.
    ## Cómo verificar
    El comando exacto de "Verificación de Fede" y la salida REAL pegada (recortada a 30 líneas).
    Resultado del verificador de contexto fresco (PASS/FAIL por criterio).
    ## Qué NO se hizo
    Lo que quedó fuera y por qué. Dudas que necesitan decisión de Fede.
  $BD comments add <ID> "PR: <url> — <1 línea de estado>"
  Avisá por Telegram (skill hermes-cli, sin LLM): "<ID> listo para revisar: <url>. Verificar con: <comando>".

PASO 5 — Bloqueo (solo si no hay PR)
  $BD label add <ID> ai-blocked ; $BD label remove <ID> ai-in-progress
  $BD comments add <ID> "BLOCKED: <qué intentaste, qué falló, qué decisión hace falta>"
  git push -u origin feature/<ID>-<slug>   (aunque esté incompleto: no se pierde trabajo)
  Avisá por Telegram con la misma línea.

PASO 6 — Limpieza
  cd /home/azureuser/repos/projects/QuantAgent && git worktree remove --force /tmp/ai-loop/$RUN
  Confirmá que `git status --porcelain` en el repo principal está limpio (salvo .beads/issues.jsonl,
  que el wrapper ya exportó; commitealo en una rama chore/beads-sync-$RUN y pusheala, sin PR).
```

## 5. Cuerpo JSON para crear la routine (referencia)

```json
{
  "name": "QuantAgent — loop nocturno (1 ticket → 1 PR)",
  "cron_expression": "0 4 * * 1-5",
  "enabled": true,
  "job_config": {
    "ccr": {
      "environment_id": "env_013DRa3BESN2FVeiQt4W91jN",
      "session_context": {
        "model": "claude-opus-5",
        "sources": [{"git_repository": {"url": "https://github.com/fscheu/QuantAgent"}}],
        "allowed_tools": ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "Agent"]
      },
      "events": [{"data": {"uuid": "<uuid v4>", "session_id": "", "type": "user",
                  "parent_tool_use_id": null,
                  "message": {"role": "user", "content": "<PROMPT DE §4>"}}}]
    }
  }
}
```

Se crea desde una sesión de Claude Code con `/schedule` (pide confirmación de entorno, modelo y horario)
o en https://claude.ai/code/routines. Primera corrida: manual ("run now") con `QuantAgent-3km` como único
ticket `ai-ready`.

## 6. Checklist diario de Fede (20 min)

1. Telegram: ¿hay PR o bloqueo? (1 min)
2. Leer `## Cómo verificar` y `## Qué NO se hizo` (5 min)
3. Correr el comando de verificación en la VM y mirar el resultado (10 min)
4. Merge (squash) o un comentario de una línea en el PR + `bd comments add` (3 min)
5. Si mergeó: `bd close <ID>`; asegurarse de que el próximo `bd ready --label ai-ready` no esté vacío (1 min)

## 7. Riesgos conocidos y mitigación

| Riesgo | Mitigación |
|---|---|
| El agente se "vende" un PASS propio | Paso 3 con subagente sin historial + CI en PR (ticket CI) + Fede corre el comando |
| Worktree o rama sucia entre corridas | Worktree nuevo por corrida, borrado al final; `.beads` sync en rama aparte |
| Toma un ticket que necesita decisión | Label `fede-decision` nunca es `ai-ready` |
| Dos corridas pisan el mismo ticket | Un ticket por corrida + `ai-in-progress` lo saca de `ready` |
| Fede no revisa 3 días seguidos | Se acumulan como máximo 3 PRs; el loop no toma más si hay ≥3 PRs abiertos (agregar al prompt tras la primera semana) |
