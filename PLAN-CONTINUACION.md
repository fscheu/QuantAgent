# PLAN-CONTINUACION

Generado: 2026-09-26. Reemplaza a `docs/02_planning/2026-09-17_plan_mes_2_loop_ai.md` y a
`docs/02_planning/2026-09-17_borrador_routine_nocturna.md`.
Tickets en BEADS con label `plan-continuacion`.

Modo de trabajo (desde 2026-10-06, `QuantAgent-9vn`): el loop hace entregas chicas, de a una. La sesión
`pm-revisor` revisa cada entrega y la integra en un **lote** de 5-10 tickets relacionados. Fede revisa y mergea
a `main` un PR por lote, y decide lo que el PM le escala. Detalle en §3.7.

Modo anterior (2026-09-26 a 2026-10-06): una entrega por noche y Fede revisaba cada PR. Las secciones §1 y §2
describen ese período y quedan como registro.

---

## 1. Cierre del ciclo anterior

### 1.1 Estado real (verificado 2026-09-26 sobre `feature/plan30`)

**Funcionando**

| Qué | Evidencia |
|---|---|
| `python -m quantagent.cli backtest run --strategy rsi --fixture spy-90d --out run.csv` | exit 0; `Trades: 227`, win rate 49.78%, PF 0.61, Sharpe 0.40, PnL -4717.33 |
| Trade log CSV con fechas de vela | 228 líneas; primera fila `2026-01-02T05:00:00` |
| `trades.backtest_run_id` + migración | tests de aislamiento en la suite |
| Fixtures versionados + loader | `tests/fixtures/spy-90d.csv`, `spy-smoke.csv` |
| Suite rápida | 781 passed, 0 failed, 22 skipped |
| README Quick Start y sección del CLI | reproducible copiando y pegando |
| Tickets cerrados | `iip`, `gg6`, `83e` |

**A medias**

| Qué | Estado |
|---|---|
| Merge a `main` | No hecho. `feature/plan30` mergea sin conflictos contra `origin/main` |
| `fifty-two-week-high` y `triple-screen` | exit 0 pero 0 trades sobre `spy-90d` |
| Salida del CLI | Ruido de "Insufficient data" y `SAWarning` antes de las 5 líneas |
| Métricas | Sin auditar contra un cálculo independiente |

**Abandonado o diferido**

| Qué | Destino |
|---|---|
| D19, botón de Streamlit | Diferido sin fecha |
| Plan de mes 2 (21/09) | No arrancó. Reemplazado por este plan |
| Routine nocturna | Nunca creada. Reemplazada por §2 y §3 |
| Skill de revisión macro (`wwi`) | Fuera del loop, pendiente de sesión de escritorio |

### 1.2 Trabajo mínimo para declararlo CERRADO

| # | Tarea | Quién | Dónde | Hecho cuando |
|---|---|---|---|---|
| C1 | `CHANGELOG.md` de cierre, totales del checklist, tracker apuntando a este plan | Claude | repo | Commit `e8132343` en `feature/plan30` ✅ |
| C2 | Abrir PR `feature/plan30` → `main` con resumen de 3 líneas | Claude | GitHub | PR #1 abierto ✅ |
| C3 | Correr la demo desde la rama del PR en la VM | Fede | celular + VM | Salida con `Trades: 227` ✅ 2026-09-29 |
| C4 | Escribir la línea de revisión (§5) en el PR y mergear | Fede | GitHub mobile | PR #1 mergeado 2026-09-29 ✅ (la línea `R:` quedó con `vi:` vacío) |

El ciclo quedó **CERRADO** el 2026-09-29 con el merge del PR #1. No se agrega ningún cambio de código al PR de cierre.

El diff del PR de cierre supera el límite de §3.2. Es la única excepción: el código ya se revisó en las
sesiones del 15 al 17/09. La revisión de C3 es **probar**, no leer el diff.

Comando de C3, para copiar en la VM:

```bash
cd ~/repos/projects/QuantAgent && git fetch -q origin && git checkout -q feature/plan30 && git pull -q && source .venv/bin/activate && python -m quantagent.cli backtest run --strategy rsi --fixture spy-90d --out /tmp/run.csv 2>/dev/null | tail -6 && wc -l /tmp/run.csv
```

Esperado: las 5 métricas, `Trade log written to /tmp/run.csv` y `228 /tmp/run.csv`.

### 1.3 Artefacto de cierre

`CHANGELOG.md`, entrada 2026-09-26: qué hace el sistema, cómo se corre, salida verificada,
limitaciones conocidas y qué quedó fuera.

---

## 2. Fase 0 — montaje del loop

Dos sesiones de escritorio, con notebook. Estimación total: **4 h**. Requiere C4 cerrado.

### 2.1 Decisiones de alcance (recortes para que entre en 2 sesiones)

| Se incluye | Se deja afuera |
|---|---|
| Cron de Hermes en modo script (`--no-agent`) que lanza Claude Code headless en la VM | Routine cloud de Claude Code: el entorno bridge no está verificado |
| Gate determinista en un script, antes de invocar al modelo | Gate solo en el prompt |
| Aviso por Telegram con la salida del script | Notificaciones enriquecidas o por PR |
| Un ticket por noche, domingo a jueves | Varios tickets, fines de semana |
| Modos: ticket nuevo, pedido de cambio sobre el PR abierto | Auto-merge, reintentos automáticos |
| Límite de tamaño chequeado por script del loop | Chequeo de tamaño en CI |

Hermes funciona solo como reloj y canal de Telegram. No orquesta ni usa Kimi en este flujo.

### 2.2 Sesión 1 — construir (2,5 h)

| # | Tarea | Tiempo | Listo cuando |
|---|---|---:|---|
| 0.1 | CI en `pull_request` hacia `main`, sin deploy (`QuantAgent-zui`) | 20 min | Un PR de prueba muestra el check `CI (Lint + Tests)` y no ejecuta `Deploy to QA` |
| 0.2 | Crear `loop/PROMPT.md`, `loop/config.env`, `.github/pull_request_template.md`, `docs/loop/REVIEW-LOG.md` con el texto de §3 (`QuantAgent-hak`) | 25 min | Archivos en el repo |
| 0.3 | `scripts/loop/gate.py` según §3.1 | 45 min | Tres escenarios manuales dan el resultado esperado: sin PR previo, PR sin `R:`, PR mergeado con `R:` válida |
| 0.4 | `scripts/loop/diff_size.sh` según §3.2 y `scripts/loop/try.sh <rama> -- <cmd>` (worktree temporal, corre, borra) | 20 min | `diff_size.sh` sobre `feature/plan30` sale con código 1; `try.sh` corre la demo sin tocar el checkout principal |
| 0.5 | Verificar que `bd show`, `bd update` y `bd export` funcionan dentro de un worktree limpio | 15 min | `bd export` escribe `.beads/issues.jsonl` dentro del worktree |
| 0.6 | Wrapper `scripts/loop/run_nightly.sh` según §3.5, y `scripts/loop/smoke.sh` | 25 min | Primera corrida manual abre el PR de T01 con el formato de §3.3 |

**Criterio de listo de la sesión 1:** una corrida manual del wrapper abre el PR de T01. Una segunda
corrida inmediata imprime `⏸ esperando revisión` en menos de 30 segundos y no invoca a Claude.

### 2.3 Sesión 2 — ciclo completo y cron (1,5 h)

| # | Tarea | Tiempo | Listo cuando |
|---|---|---:|---|
| 0.7 | Revisar el PR de T01 desde el celular, como ensayo. Escribir `R:` y mergear | 15 min | PR mergeado con `R:` válida |
| 0.8 | Probar una `R:` inválida en un PR de prueba (`R: x vi: ok decido: merge`) | 5 min | El gate la rechaza con el motivo |
| 0.9 | Tercera corrida manual del wrapper | 40 min | Cierra `e35` en BEADS, toma T02, el PR incluye la línea de T01 en `docs/loop/REVIEW-LOG.md` |
| 0.10 | Instalar los dos shims y crear los dos jobs de Hermes siguiendo `docs/loop/SETUP-HERMES.md` (confirmación explícita de Fede); probar `loop/PAUSE` | 20 min | `hermes cron list` muestra el job; con `loop/PAUSE` en `main` el wrapper imprime `⏸ pausado` |

**Criterio de listo de la fase 0:** job visible en `hermes cron list`, un ciclo completo cerrado
(PR → `R:` → merge → siguiente PR), y el gate rechazó al menos una vez por falta de `R:` y una vez por `R:` inválida.

Instalación (0.10): `docs/loop/SETUP-HERMES.md`. Son dos jobs porque Hermes corta los scripts de cron
a los 120 segundos y una corrida dura hasta 2 horas:

- **Lanzador**, domingo a jueves 23:00 ART: corre el gate en segundos. Si toca esperar, avisa por Telegram
  sin invocar al modelo. Si pasa, lanza el loop desacoplado.
- **Reporte**, lunes a viernes 02:30 ART: manda las 3 líneas de la entrega y la URL del PR.

Ninguno de los dos cambia la config de Hermes. El timeout de 2 h del loop termina antes del housekeeping de
worktrees de las 03:20.

---

## 3. Contrato del loop

Las reglas viven en dos lugares. El wrapper y sus scripts las **hacen cumplir** de forma determinista.
El prompt las **repite** para que el agente no intente violarlas.

### 3.1 Regla 1 — Gate de revisión

Entrega = el PR más reciente con label `loop`, sin importar su estado.
Evento de entrega = el commit más reciente de ese PR.

Antes de mirar la entrega, el gate busca el lote activo: el único PR abierto con label `lote` (§3.7).

| Lote | Resultado |
|---|---|
| Ningún PR `lote` abierto | WAIT `⏸ sin lote abierto` |
| El PR del lote ya no es borrador (está entregado a Fede) | WAIT `⏸ lote entregado, falta que Fede lo mergee` |
| Más de un PR `lote` abierto | WAIT |
| Un PR `lote` abierto y en borrador | Sigue con la tabla de abajo; la rama del lote es la base de la entrega |

`scripts/loop/gate.py` busca el último comentario de `fscheu` posterior al evento de entrega que cumpla
el formato de §5. La línea puede ser `R:` (Fede) o `PM:` (sesión `pm-revisor`). **Una `PM:` solo vale si la
base del PR es una rama de lote**: en un PR hacia `main` el gate la rechaza. Resultado:

| Situación | Resultado | El loop |
|---|---|---|
| No existe ningún PR `loop` | PASS `nuevo` | Toma el primer ticket de la cola |
| No hay `R:` posterior a la entrega | WAIT | Imprime `⏸ esperando revisión: <url> (desde <fecha>)` y termina |
| `R:` inválida (§5.3) | WAIT | Imprime `⏸ R inválida: <motivo>` y termina |
| `decido: merge` y PR mergeado | PASS `nuevo` | Cierra el ticket en BEADS y toma el siguiente |
| `decido: merge` y PR abierto | WAIT | Imprime `⏸ R dice merge, falta mergear` |
| `decido: descarto` y PR cerrado sin merge | PASS `nuevo` | Pone label `loop-descartado` al ticket y toma el siguiente |
| `decido: cambio` y PR abierto | PASS `cambio` | Aplica el pedido sobre la misma rama y hace una nueva entrega |
| `PM: ... decido: escalo` y PR abierto | PASS `nuevo` | Deja el ticket `in_progress` y el PR abierto para Fede; toma el siguiente del lote |
| Cualquier otra combinación | WAIT | Imprime la combinación encontrada |

WAIT no invoca al modelo. No hay excepciones, no hay acumulación: nunca existen dos entregas sin revisar
por el PM. Las escaladas a Fede sí pueden quedar abiertas mientras el lote sigue con otros tickets.

Pausa manual: si existe `loop/PAUSE` en `origin/main`, el wrapper imprime `⏸ pausado` y termina.
Se crea o borra desde GitHub mobile.

### 3.2 Regla 2 — Tamaño de entrega

`loop/config.env`:

```bash
LOOP_LABEL=loop
LOOP_REVIEWER=fscheu
LOOP_QUEUE_FILE=PLAN-CONTINUACION.md
LOOP_LOT_LABEL=lote
LOOP_LOT_PREFIX=lote/
LOOP_MAX_LINES=400          # agregadas + borradas, límite duro
LOOP_TARGET_LINES=100       # objetivo por entrega: no cambió
LOOP_MAX_FILES=10
LOOP_EXCLUDE=".beads/** tests/fixtures/** docs/loop/REVIEW-LOG.md"
LOOP_TIMEOUT_MIN=120
```

- Un archivo borrado completo no suma líneas, pero se lista en el resumen.
- `scripts/loop/diff_size.sh <base>` cuenta con `git diff --numstat <base>...HEAD` aplicando las exclusiones.
  La base es la rama del lote.
- El límite duro subió de 150 a 400 el 2026-10-06 para no tirar una entrega que salió algo más grande que lo
  estimado. **El tamaño de los tickets no cambia**: si el agente estima más de 100 líneas, no escribe código
  y entrega una partición (§3.4). Las entregas chicas son lo que le permite al PM revisar de verdad.
- Si al terminar el diff excede el límite, el agente descarta la implementación y entrega la partición.
  El wrapper vuelve a medir y, si excede, pone label `excede-limite` al PR.

### 3.3 Regla 3 — Resumen de entrega

Cuerpo del PR (`.github/pull_request_template.md`). Las 3 primeras líneas son obligatorias y se leen sin
abrir nada más:

```text
Cambió: <qué hace ahora el sistema que antes no, en una frase>
Decidir: <pregunta con opciones cerradas, o "nada: merge si el check está verde">
Riesgo: <qué se puede romper y cómo se notaría, o "nada fuera de <archivo>">

── Bloque 1 · leer ──
Ticket: QuantAgent-xxx · T0N · Revisión: leer | decidir | probar
Tamaño: <N> líneas / <M> archivos (objetivo 100, límite 400 / 10) · Lote: <rama base>
Leé en este orden:
1. <archivo>:<función> — <qué mirar, en una línea>
2. ...

── Bloque 2 · probar y decidir ──
Probar (copiar en la VM):
~/repos/projects/QuantAgent/scripts/loop/try.sh loop/<ID> -- '<comando>'
Esperado: <salida en ≤5 líneas>
Obtenido por el loop: <salida real, ≤15 líneas>
Verificador independiente: PASS | FAIL por criterio
Qué NO se hizo: <lista corta>

Registro — lo comenta quien revisa (PM: la sesión pm-revisor, R: Fede):
PM: <ID> vi: <dato concreto de esta entrega> decido: merge | cambio <qué> | descarto <por qué> | escalo <pregunta>
```

Reglas de redacción: frases cortas, sin jerga interna, sin enlaces obligatorios. Todo lo necesario para
decidir está en el cuerpo del PR.

### 3.4 Regla 4 — Estado consistente

Todo PR `loop` cumple, antes de abrirse:

1. Tests dirigidos → 0 failed: la línea "Tests a correr:" que el PM escribe en cada ticket (los archivos de
   `tests/` de los módulos que toca), más los tests que la entrega crea o modifica. Desde el 2026-10-14 (pedido
   de Fede, QuantAgent-1vj7) el agente ya no corre la suite completa: `-m "not slow and not api and not oraculo"`
   la corre el CI de cada PR, y el CI verde (punto 3) es condición de merge. Si el CI rompe en algo que la lista
   no cubría, la entrega vuelve como `cambio` y el PM amplía la lista de ese módulo.
   Los oráculos corren en el CI de cada PR y el PM los corre en cada revisión que toque `quantagent/backtesting/`, `quantagent/trading/`, `quantagent/portfolio/` o `quantagent/strategy/`, y en el punto de control de cada lote, con `python -m pytest -q -m oraculo`.
2. `scripts/loop/smoke.sh` → exit 0 (SQLite temporal, sin tocar la base de desarrollo).
3. CI verde en el PR (el workflow corre en PRs hacia `main` y hacia `lote/**`; lo verifica el PM).
4. Si cambia un comportamiento documentado, el README o el doc afectado se actualiza en el mismo PR.

`main` solo cambia por merges de Fede: el PM integra entregas en la rama del lote, nunca en `main`. Si el agente no llega a un estado verde, la entrega es de uno de
estos dos tipos, que solo tocan documentación y dejan `main` funcionando:

| Tipo | Label | Diff | "Decidir:" |
|---|---|---|---|
| Bloqueo | `loop`, `bloqueo` | `docs/loop/bloqueos/<ID>.md`, ≤40 líneas: qué intentó, qué falló | Qué decisión destraba el ticket |
| Partición | `loop`, `particion` | `docs/loop/particiones/<ID>.md` + hijos en `.beads/issues.jsonl` | Aprobar la partición |

El código incompleto se pushea a `loop-wip/<ID>` sin PR, para no perderlo.

### 3.5 Wrapper `scripts/loop/run_nightly.sh`

Hermes ejecuta dos shims en `~/.hermes/scripts/` que corren `scripts/loop/hermes_launch.sh` y
`scripts/loop/hermes_report.sh` tal como están en `origin/main`. El lanzador copia el wrapper a un
directorio temporal antes de ejecutarlo, porque el deploy resetea el checkout principal en cada merge.

Desde el 2026-10-07 el cron no lanza la corrida: le escribe un aviso a la sesión del PM (tmux `LOOP_PM_TMUX`) y
es el PM quien revisa lo pendiente y lanza con `hermes_launch.sh --run`, una corrida tras otra. Antes el cron
lanzaba una sola corrida por noche y el lote quedaba esperando al PM hasta que Fede le escribía. Si la sesión
del PM no existe, el cron lanza una corrida como antes. El aviso dice que es automático: no es una aprobación
de Fede ni una `R:`.

1. `git fetch origin`. Si existe `loop/PAUSE` en `origin/main`: imprimir y salir.
2. `scripts/loop/gate.py` desde `origin/main`. WAIT: imprimir la línea y salir.
3. Crear worktree `/tmp/ai-loop/<RUN>` desde la rama del lote que informa el gate (`base`), o desde la rama del
   PR en modo `cambio`. Si `base` no empieza con `lote/`, no se lanza nada.
4. Agente según `LOOP_AGENT` (`claude` | `agy` | `codex`; el archivo `next-agent` del directorio de estado lo cambia
   por una corrida), con `loop/PROMPT.md` más el contexto del gate (modo, PR previo, texto de la `R:`) y
   timeout `LOOP_TIMEOUT_MIN`. `claude -p` usa `--permission-mode acceptEdits` y
   `--disallowedTools "Bash(git push origin main*)" "Bash(git push -f*)" "Bash(git push --force*)"`;
   `agy -p` usa `--mode accept-edits` y sus reglas allow/deny (`docs/loop/SETUP-HERMES.md`); `codex exec` usa
   el aislamiento `workspace-write` con red y con el `.git` y el `.beads` del checkout principal como
   escribibles. Para los tres, el hook `scripts/loop/push_guard.sh` rechaza push a main, borrado de ramas y push forzado.
   El PR entregado recibe el label `agent:<agente>` y, con `agy` o `codex`, `model:<modelo>`.
   El modelo de `agy` es `LOOP_AGY_MODEL` y el de `codex`, `LOOP_CODEX_MODEL`; el archivo `next-model` del directorio de estado lo cambia por una
   corrida. Lo escribe el PM según la tarea: el modelo por defecto para tickets acotados, uno más fuerte para
   diagnóstico o cambios en semántica de plata, mirando el cupo (medido el 2026-10-07: Sonnet 5.5 alto gasta
   cerca de 15% del cupo semanal por corrida; Gemini 3.8 Flash alto, cerca de 3%).
5. `scripts/loop/diff_size.sh origin/<base>` sobre la rama entregada. Si excede, label `excede-limite`.
6. Borrar el worktree.
7. Imprimir para Telegram: las 3 líneas del resumen y la URL del PR.

### 3.6 Prompt del agente

El texto vigente es `loop/PROMPT.md`. Hasta el 2026-10-06 esta sección tenía una copia, que quedaba vieja en
cada ajuste. El prompt repite las reglas de §3; si choca con este archivo, manda este archivo.

### 3.7 Lotes y revisión del PM

**Qué es un lote.** Entre 5 y 10 tickets relacionados que juntos dejan un punto de control: algo que el sistema
hace y que se comprueba con un comando. El PM decide la agrupación y la escribe en §4. Un lote no tiene fecha:
termina cuando cierra su punto de control.

**Mecánica**

1. El PM crea la rama `lote/<nombre>` desde `origin/main` y abre un PR **borrador** hacia `main` con label `lote`.
2. El loop toma solo filas de ese lote, parte de esa rama y abre cada PR `loop` contra ella.
3. El PM revisa cada entrega (protocolo abajo), comenta la línea `PM:` y, si decide merge, mergea el PR en la
   rama del lote. La Action `lote-automerge` puede hacer ese merge cuando la línea `PM:` dice `decido: merge`, el CI está verde y la base es `lote/*`; el pase lote → main sigue manual con la `R:` de Fede. Después lanza la siguiente corrida. Tope: 10 entregas por día. La tanda la arranca el aviso del
   cron (§3.5) o un pedido de Fede; el PM frena si la cuota del agente baja del 20% de la ventana de 5 horas.
4. Cuando no quedan filas elegibles, el PM corre el punto de control sobre la rama del lote, completa el cuerpo
   del PR del lote y lo marca listo para revisión. Desde ahí el gate espera.
5. Fede prueba el punto de control, comenta `R: lote-<nombre> vi: <...> decido: merge` y mergea a `main`.
6. El PM abre el lote siguiente.

Si dos entregas seguidas terminan en `cambio`, `bloqueo` o `descarto`, el PM frena el lote y avisa a Fede:
es señal de que los tickets están mal definidos, no de que falte velocidad.

**Protocolo de revisión del PM, por entrega**

1. Leer el ticket (`bd show`) antes que el cuerpo del PR: el criterio de aceptación es del ticket, no del
   resumen del implementador.
2. Leer el diff completo. Marcar todo lo que esté fuera de "Archivos relevantes" o del cambio pedido.
3. Correr él mismo el comando de "Probar" con `scripts/loop/try.sh` y los tests dirigidos del ticket, y confirmar
   el CI verde (la suite completa). No vale la salida pegada en el PR.
4. Buscar el hueco con al menos una comprobación que el PR no trae: otro fixture u otra estrategia, un caso
   borde, un valor recalculado por otro camino, un test que debería fallar si se rompe la lógica y no falla.
5. Decidir y comentar la línea `PM:`. En `vi:` va el dato de la comprobación propia del paso 4.
6. Al cerrar la revisión, agregar la línea a `docs/loop/REVIEW-LOG.md` en la rama del lote (§5.4).

**Revisor mecánico (Codex), desde el 2026-10-11.** Para no gastar el cupo del PM en correr comandos, los pasos 3
y 4 los puede hacer Codex con `scripts/loop/codex_review.sh` (instalación en `docs/loop/SETUP-HERMES.md`). Codex
deja un comentario `CODEX:` con las salidas reales y una sugerencia; no decide ni escribe la línea `PM:`. El PM
lee el diff y ese reporte, y decide. Tres resguardos:

- El PM sigue corriendo él mismo los pasos 3 y 4 en las entregas de tipo "decidir" y en las que tocan PnL,
  comisiones, slippage, tamaño de posición o límites.
- Del resto, el PM repite completa una de cada cinco revisiones de Codex. Si encuentra algo que Codex dejó
  pasar, ese tipo de entrega vuelve al PM y se anota en `docs/loop/REVIEW-LOG.md`.
- Las tres primeras entregas con Codex llevan doble revisión completa, para medir cuánto coinciden.

**Qué no aprueba el PM (siempre `decido: escalo`)**

- Entregas de tipo "decidir".
- Un número de referencia que cambia de una forma que el ticket no anticipaba.
- Cambios en la semántica de plata o riesgo (PnL, comisiones, slippage, tamaño de posición, límites) que el
  ticket no traiga ya decididos por Fede.
- Cambios de alcance, tickets nuevos que alteran el plan, o algo fuera del repo.
- Un segundo `cambio` sobre el mismo ticket.

Una escalada no frena el lote: el ticket queda `in_progress`, el PR abierto, y el loop sigue con las filas que
no dependen de esa decisión. Cuando Fede contesta con su `R:`, el PM la aplica (merge al lote, descarte, o
relanzar el loop en modo `cambio`).

**Cuerpo del PR del lote** (lo que lee Fede):

```text
Punto de control: <qué hace ahora el sistema, en una frase>
Probar (copiar en la VM): ~/repos/projects/QuantAgent/scripts/loop/try.sh lote/<nombre> -- '<comando>'
Esperado: <salida en ≤5 líneas>      Obtenido por el PM: <salida real>
Decidir: <preguntas abiertas con recomendación, o "nada">

Entregas: <ticket> · #<pr> · <una línea: qué cambió y qué comprobó el PM>
Encontrado en revisión: <lo que el PM rechazó, pidió cambiar o descubrió>
Qué NO se hizo: <lista corta>
```

**Reporte diario a Fede** (Telegram o la sesión, ≤10 líneas): qué sabe hacer el sistema que ayer no, avance del
lote (n de m) y de M1, lo encontrado en revisión, y las decisiones pendientes con recomendación.

**Operar el loop.** Para que Codex escriba registros reales en `local-data` por una corrida, creá
`~/.local/state/quantagent-loop/next-local-data` (vacío alcanza). Tras pasar el gate se consume y Codex recibe
`--add-dir "$QUANTAGENT_SNAPSHOT_DIR"`, registrado en el log; otros agentes lo consumen con un aviso y lo
ignoran. No cambia los permisos de los snapshots. Detalles en `docs/loop/SETUP-HERMES.md`.

**Límite conocido.** El PM comenta con la misma cuenta de GitHub que Fede, así que el gate distingue `PM:` de
`R:` por el prefijo y no por el autor. La barrera real es que una `PM:` no vale en PRs hacia `main` y que el
PM no mergea a `main`.

**El primer lote es la prueba del modo.** Fede anota en la `R:` del lote lo que encontró y el PM había dejado
pasar. Con cero o un escape, se sigue. Con más, las entregas del tipo donde falló el PM vuelven a la `R:` por PR.

---

## 4. Backlog de tickets ordenado

Esta tabla es la **cola** del loop: toma la primera fila elegible **del lote activo**. El estado vive en BEADS.
Reordenar la cola = mover filas en este archivo.

Lotes (definidos por el PM el 2026-10-06; cada uno se abre cuando Fede mergea el anterior):

| Lote | Rama | Filas | Punto de control | Decisiones de Fede previstas |
|---|---|---|---|---|
| L1 | `lote/metricas-auditadas` | T08c, T13, T09–T12, T14, T15, T15b, T10b, T20 | Las 5 métricas de rsi/spy-90d coinciden con un recálculo que no importa `quantagent`, la corrida es reproducible (`backtest verify`) y un test golden las congela | T13 (anualización del Sharpe), T10 (trades con pnl = 0) |
| L2 | `lote/tres-estrategias` | T16–T19, T19b | RSI, 52-week-high y Triple Screen operan (≥5 trades cada una) sobre fixtures versionados y deterministas | ninguna prevista |
| L3 | `lote/engine-y-cierre-m1` | T21–T25 | Comparación trade por trade contra `backtesting.py`, ADR del engine decidido (cierra M1), inventario de ramas y borrador del plan siguiente | T22, T23, T24, T25 |
| L4 | `lote/datos-reales-diarios` | V00, V00a–V00e, V01–V10, V03b, V10b | Las 3 estrategias de M1 corren sobre SPY diario real 2007–2018 desde un snapshot con manifiesto; el engine avanza por las velas del dato y `backtesting.py` coincide con RSI | V00 (performance de la suite), V10 |
| L5 | `lote/costos-y-referencia` | V11–V20, más V12b, V16b–V16g, V17b, V17c, V19a–V19g | La corrida informa comisión, slippage y tamaño, y se compara contra comprar y mantener con los mismos costos; un recálculo independiente coincide; motor y `backtesting.py` coinciden trade por trade con los costos reales | V13, V17, V18, V16e (motor oficial), V12b, V17b, V17c, V19a, V19c |
| L6 | `lote/comparacion` | V20b, V20c, V21a, V21b, V22–V26, V23a, V23b, V23c, V08b, V23d | Un comando corre cuatro estrategias diarias sobre los 12 ETFs y escribe una tabla con el resultado, la referencia de comprar y mantener, la marca "coinciden N de N" contra `backtesting.py` y la sensibilidad de apertura siguiente | V20b (carpeta de datos locales), V21a (límite diario) |
| L7 | `lote/fuera-de-muestra` | V27a–V27d, V30–V36, V36a | Hay veredicto PASA / NO PASA por estrategia y activo sobre la prueba 2019–2022, con umbrales fijados antes de abrirla, intentos contados y la reserva cerrada; las 48 celdas de la tabla validan contra `backtesting.py` o tienen causa escrita | V30 (umbrales y supuesto de llenado, antes de abrir la prueba); pivot a LEAN: se decide en M3 (Fede, 2026-10-13, con el resultado del spike `QuantAgent-ryaw`) |
| L8 | `lote/motor-del-filtro` | F01–F03, F03a, F03a.2a–k, F03b.1–2, F05–F08, F08b, F09 (F04 salió el 2026-10-15) | Plan M3 (`docs/02_planning/QuantAgent-d4cq-PL-m3-fabrica-de-estrategias.md`): `vectorbt` corre las 4 estrategias de M2 sobre los 12 ETFs 2007–2018 con llenado en la apertura siguiente, stops dentro de la vela y prueba de truncamiento; medidos el tiempo de las 48 celdas y el Spearman contra el motor propio | F01 (llenado por defecto: Fede eligió A el 2026-10-14), F03a y F03b (semántica de llenado y números de referencia), F09 (motor del filtro: B o A) |
| L9 | `lote/hipotesis` | F09.4, F03b.3–5, H01–H07 | Dos hipótesis escritas como archivo YAML corren de punta a punta en la etapa 1 (`backtest hipotesis filtrar`), con admisión, intentos por hipótesis y candado | ninguna prevista: umbrales provisorios decididos por Fede el 2026-10-15 (máx. 6 parámetros libres; corta con Sharpe fuera de muestra < 0,10 en más de 8 de 12 activos) |
| L10 | `lote/embudo` | E01–E09 | `backtest hipotesis correr` lleva una hipótesis por las etapas 0 a 3 y se detiene en la primera que corta, con estado en `hipotesis.csv`, control contra `backtesting.py` en la etapa 2 y Sharpe deflactado contra el total de intentos en la 3 | umbrales de la etapa 1 decididos por Fede el 2026-10-16 (más de 2/3 de los activos bajo 0,10; menos de 30 trades cuenta como bajo; Sharpe vacío cuenta como bajo); etapa 3 "al menos un activo pasa" y OK para abrir la prueba de las dos hipótesis de L9 (E09) |
| L11 | `lote/informe-y-tanda` | I01–I10 | `backtest hipotesis tanda hipotesis/` procesa la carpeta entera sobre datos reales y deja un informe HTML por hipótesis y un resumen; la alarma filtro vs motor frena antes de abrir la prueba, un error deja estado y una hipótesis con la prueba ya abierta no suma intentos | decididas por Fede el 2026-10-16: regla de la alarma de I01 (filtro ≥ 0,10 y motor ≤ 0 en al menos la mitad) y punto de control con las 4 hipótesis disponibles en vez de 10 (el catálogo es de L12); falta el OK para abrir la prueba de las que pasen (I10) |
| L12 | `lote/cartera` | C01–C11 | `backtest hipotesis correr` da veredicto a la cartera tendencia + momentum sobre unos 30 ETFs que cotizaban en 2007, con pesos por igual riesgo, control por celda contra `backtesting.py` y DSR de la cartera | decididas por Fede el 2026-10-17: cartera de primas (DC `2026-10-17-DC-cartera-de-primas`, D1–D6) y Sharpe a tasa 0 (C01) y D7 (una fila por cartera y etapa en `intentos.csv`); pendiente el OK para abrir la prueba en C10 |
| L13 | `lote/catalogo-de-primas` | G04a, G04b, G05, G06a, G06b, G07 (se reescriben) | Catálogo de primas aplicables con ETFs corrido como carteras; el LLM propone primas con cita y universo, el crítico rechaza duplicadas | propuesto; se arma al mergear L12 |
| L14 | `lote/capa-macro` | G03 y tickets nuevos | FRED/ALFRED punto en el tiempo y régimen de mercado como hipótesis aparte, aceptada solo si mejora el DSR del núcleo | propuesto; puede ir en paralelo con M4 |
| LB1 | `lote/datos-externos` (carril B, label `carril-b`, PR #257) | B-F1–B-F4, B-T1, B-T2, B-T9, B-T8 | Snapshot macro `macro-2026-10` punto en el tiempo (10 series FRED/ALFRED) y `AlpacaSource` bajando SPY con claves reales | OK de Fede el 2026-10-17 para paralelizar la capa de datos de M4 y L14 (nube y notebook); fuera: broker y ejecución en paper hasta cerrar L12. Necesita `FRED_API_KEY` y claves de Alpaca paper en `~/secrets` (las pone Fede) |
| L15 | `lote/gate-por-lote` | K01 | El gate del loop con `LOOP_LOTE` evalúa solo la entrega previa del lote activo: un PR loop de una máquina no frena ni activa a la otra | PM, 2026-10-18; sale de origin/main para llegar a main sin esperar a L13 ni L14 |
| L16 | `lote/criterio-etapa3` | E01–E08 | `quantagent backtest hipotesis informe <id>` de una cartera validada muestra DSR con N crudo y N efectivo, Sharpe por año del holdout, recargo de paper y veredicto con el criterio del DC; recalculado sobre las tres carteras que cortaron | decidido por Fede el 2026-10-19 (DC #322, plan #323); sale de origin/main; todo corre en la VM |
| L17 | `lote/loop-investigador` | I01–I08 | Un ciclo completo sin correr nada en el embudo: fichas de postmortem de las tres carteras de L16, el pedido al investigador, hasta 3 propuestas en `catalogo/propuestas/` con chequeo de hipótesis y de originalidad, y un comando que las lista para que Fede las revise; nada pasa a `hipotesis/` sin su OK | plan `QuantAgent-vbnh-PL-l16-l17-criterio-etapa3-y-loop.md` §4; reglas decididas por Fede el 2026-10-19 (sin Sharpe del holdout, ideas de literatura, máx. 2 variantes por idea, todo intento suma a N); sale de origin/main |
| L18 | `lote/m4-prep-curva-puv2` | Q01–Q13 | Veredicto de la cartera núcleo + curva contra el criterio de paper (recalculado de las curvas guardadas, sin abrir 2019–2022); `cambio-de-mes` con su señal corrida hasta la etapa 2; los datos de Alpaca por fuente y los tres diagnósticos de paper (`say`, `oc0h`, `beo`) listos para decidir | plan `QuantAgent-ro3w-PL-l18-curva-puv2-prep-m4.md`; Fede aprobó los tres frentes el 2026-10-20; decisiones previstas: Q01 (si pasa, entra a M4), Q04 (OK para la etapa 3), Q09, Q10–Q12 |
| L19 | `lote/medicion-y-generador` | S01–S09 | `quantagent backtest hipotesis informe <id>` de las cuatro carteras muestra Sharpe en exceso de la tasa corta, benchmark con los mismos pesos y error estándar del ΔSharpe; hay una tabla que separa la brecha contra 1/N en pesos, señal y tasa; el pedido al investigador sale con criterio, inventario y lista negra de efectos decaídos, y las propuestas pasan por chequeo de DOI contra Crossref | decidido por Fede el 2026-10-21 (revisión del PR #352; plan por S01–S09 sin cambios); no corre nada en el embudo ni abre ventanas; sin sesiones cloud por el cupo semanal: S01–S03 y S05–S08 en la notebook con Codex, S04 y S09 en la VM después de que cierre L18; sale de origin/main |
| L20 | `lote/investigacion-edge` | X06–X09, X10a–X10c, X11–X18, X20; X01–X05 y X19 movidos a L21 | Riesgo y sizing por trade; F1/F2 hasta etapa 2; datos AQR/French, ETFs ampliados y pysystemtrade; generador con lección/refutador; protocolo de reserva. X10 y X14 se desglosan. | R01 (`docs/04_decisions/QuantAgent-R01-DC-decisiones-investigacion-edge.md`); PR #416 desde `origin/main`; ninguna apertura de reserva sin OK de Fede |
| L21 | `lote/paper-alpaca` | X01–X05, X19a–c y W02–W10, W15–W16 | Dos carteras operan en Alpaca paper desde el scheduler; rebalanceo, sombra, reconciliación y chequeos; digest operativo antes del piloto. | `QuantAgent-av1p` y `docs/02_planning/QuantAgent-L21-PL-paper-alpaca-m4.md`; aprobado por Fede el 2026-10-22; PR #412. La puesta en marcha por dos semanas requiere autorización operativa explícita. |

T13 se adelantó dentro de L1: es un documento de decisión que no depende del resto, así la respuesta de Fede
llega mientras el lote avanza. T20 pasó a L1 porque congela los números que ese lote audita.

Tipos de revisión: **leer** (el diff alcanza), **decidir** (hay una pregunta cerrada en "Decidir:"),
**probar** (un comando en la VM con `scripts/loop/try.sh`). Ningún ticket toca la UI, así que ninguno
requiere el deploy de QA.

Los primeros 5 son los más fáciles de revisar.

| # | Ticket | Objetivo | Aceptación binaria | Revisión | Lote |
|---:|---|---|---|---|---|
| T01 | `QuantAgent-e35` | Borrar `tests/test_backtest_apscheduler_9wz.py`, que no prueba código propio | El archivo no existe y CI está verde | leer | — |
| T02 | `QuantAgent-bv8` | Salida del CLI limpia: solo las 5 líneas de métricas | `backtest run ... 2>&1 \| wc -l` devuelve 5 | probar | — |
| T02b | `QuantAgent-fdi` | El backtest determinista deja de exigir `OPENAI_API_KEY` | `env -u OPENAI_API_KEY scripts/loop/smoke.sh` termina con exit 0 | probar | — |
| T03 | `QuantAgent-fiu` | `test_parallel_execution` sobre `spy-smoke.csv`, sin skip | El test pasa sin `@pytest.mark.skip` | leer | — |
| T04 | `QuantAgent-3km` | Guardrail de 730 días para 1h/4h en el provider Yahoo | Test de rango fuera de ventana da error explícito o recorte con log | leer | — |
| T05 | `QuantAgent-11v` | `--equity-out` exporta la equity curve a CSV | Mínimo de `drawdown_pct` del CSV = drawdown impreso | probar | — |
| T06 | `QuantAgent-89e.1` | Test `xfail` que reproduce la fila duplicada en reversiones + diagnóstico | Test `xfail(strict=True)` presente; resumen responde "¿round-trip real o error?" | decidir | — |
| T07 | `QuantAgent-89e` | Corregir la duplicación | `Trades: N` = filas de datos del CSV; el `xfail` pasa a test normal | probar | — |
| T08 | `QuantAgent-hx0.5` | Script de recálculo independiente: trades, PnL, win rate, PF | No importa `quantagent`; test con 5 trades a mano | leer | — |
| T08b | `QuantAgent-hx0.8` | CSV con `exit_price` ejecutado; `recalc_metrics.py` verifica el PnL de cada trade | Imprime `PnL por trade: 227/227 filas coinciden`; métricas sin cambio | probar | — |
| T08c | `QuantAgent-hx0.9` | Slippage por defecto 0,05% por lado; el CLI imprime el slippage usado | 6 líneas, la última `Slippage: 0.05% por lado`; con `TRADING_SLIPPAGE_PCT=0.01` vuelve a −4717.33 | probar | L1 |
| T13 | `QuantAgent-hx0.7` | Decisión: periodos por año para Sharpe en fixtures 24/7 | Doc ≤40 líneas con Sharpe por opción | decidir | L1 |
| T09 | `QuantAgent-hx0.1` | Auditar PnL y total return; documentar fórmula | `recalc_metrics.py` = CLI en trades y PnL | probar | L1 |
| T10 | `QuantAgent-hx0.2` | Auditar win rate y profit factor; sin `inf` en DB | `recalc_metrics.py` = CLI en win rate y PF | decidir | L1 |
| T11 | `QuantAgent-hx0.6` | Recálculo de max drawdown y Sharpe desde la equity | Test con curva conocida (max DD 25%) | leer | L1 |
| T12 | `QuantAgent-hx0.3` | Auditar max drawdown; documentar la equity curve | `recalc_metrics.py --equity` = CLI en drawdown | probar | L1 |
| T14 | `QuantAgent-hx0.4` | Implementar la anualización elegida | `recalc_metrics.py --equity` = CLI en Sharpe (±0.01) | probar | L1 |
| T15 | `QuantAgent-y8z` | `backtest verify`: doble corrida y diff | Imprime `OK reproducible` con exit 0 para RSI | probar | L1 |
| T15b | `QuantAgent-hx0.11` | `backtest run` no depende de corridas anteriores en la misma base | Dos `backtest run` seguidos sobre la misma base SQLite imprimen las mismas 6 líneas | probar | L1 |
| T10b | `QuantAgent-hx0.10` | El CLI imprime `Profit factor: n/a` cuando no hay trades perdedores | Test del CLI y de `recalc_metrics.py` con solo ganadores: la línea es `Profit factor: n/a` | leer | L1 |
| T20 | `QuantAgent-piv` | Test golden RSI/spy-90d con los valores ya verificados | El test falla si cambia un trade o una métrica | leer | L1 |
| T16 | `QuantAgent-46h.1` | Generador determinista del fixture diario de 2 años | Dos corridas dan el mismo `sha256sum` | leer | L2 |
| T17 | `QuantAgent-46h` | 52-week-high corre sobre ese fixture | `Trades:` ≥ 5 | probar | L2 |
| T18 | `QuantAgent-8u8.1` | Generador determinista del fixture 4h de 1 año | Dos corridas dan el mismo `sha256sum` | leer | L2 |
| T19 | `QuantAgent-8u8` | Triple Screen corre sobre ese fixture | `Trades:` ≥ 5 | probar | L2 |
| T19b | `QuantAgent-7c6` | CSV de equity redondeado a 4 decimales como máximo | Ningún valor del CSV tiene más de 4 decimales; el golden de T20 sigue verde | leer | L2 |
| T21 | `QuantAgent-832.1` | RSI portado a `backtesting.py` | Script exit 0; parámetros iguales lado a lado | leer | L3 |
| T22 | `QuantAgent-832` | Comparar ambos engines trade por trade | Doc con cada diff explicado o ticket de bug abierto | decidir | L3 |
| T23 | `QuantAgent-lcv` | ADR: engine propio vs `backtesting.py` | ADR con 3 opciones; decisión en la `R:` | decidir | L3 |
| T24 | `QuantAgent-cv1` | Inventario de worktrees y ramas muertas | Tabla + comando exacto; el loop no borra nada | decidir | L3 |
| T23b | `QuantAgent-al2.1` | Engine: modo `intrabar_stops` apagado por defecto (regla + tests) | Con el modo apagado el golden de T20 sigue verde; con `--intrabar-stops` rsi/spy-90d da `Trades:` distinto de 227 | probar | L3 |
| T23c | `QuantAgent-al2.2` | Comparar el engine con `intrabar_stops` contra el port intravela | `crossval_compare.py --intrabar` imprime `TODO DENTRO DE TOLERANCIA` o lista cada trade que difiere con su causa | probar | L3 |
| T23d | `QuantAgent-al2.3` | `intrabar_stops` por defecto y números de referencia nuevos | `backtest verify` da `OK reproducible` en las tres estrategias; tabla antes / después en el PR | decidir | L3 |
| T23e | `QuantAgent-dj0` | La suite corre limpia sobre la base temporal de `try.sh` (15 tests solo-Postgres) | `try.sh <rama> -- 'python -m pytest -q -m "not slow and not api"'` da 0 failed y 0 errors; CI verde sin saltear esos tests | leer | L3 |
| T25 | `QuantAgent-lp1` | Borrador del plan siguiente con la evidencia de T01–T24 | Doc con cola nueva en el mismo formato | decidir | L3 |
| V00 | `QuantAgent-vfd` | Diagnóstico de performance de la suite: tests más lentos, causa y mejoras con ahorro estimado | Doc con la tabla de `--durations` de una corrida real y un objetivo de tiempo; nada fuera de `docs/` | decidir | L4 |
| V00a | `QuantAgent-pcz` | Silenciar los 80.000 warnings de `datetime.utcnow` en la configuración de pytest | Suite con 0 failed y menos de 100 warnings; el filtro no es global | leer | L4 |
| V00b | `QuantAgent-ron` | Marcador `oraculo` para golden y crossval: el loop lo saltea, el CI lo corre | `-m oraculo` corre solo esos dos archivos; el filtro del CI los sigue incluyendo; `.github/` sin cambios | leer | L4 |
| V00c | `QuantAgent-ykx` | Datos reducidos en `test_backtest_run_isolation` y `test_triple_screen_strategy` | Los dos archivos en <=20 s, 0 failed, >=2 trades por corrida y la mutación sigue fallando | probar | L4 |
| V00d | `QuantAgent-iky` | `test_backtest_cli.py` comparte corridas en vez de repetir backtests | Mismo número de tests, 0 failed, <=120 s | probar | L4 |
| V00e | `QuantAgent-ap7` | Suite en paralelo con `pytest-xdist` en la VM | `-n 4`: 0 failed en 3 corridas seguidas y <=180 s; sin `-n` no cambia | probar | L4 |
| V01 | `QuantAgent-dga` | Snapshot: Parquet por símbolo con manifiesto (hashes, rango, filas, versión de `yfinance`) | `pytest -q tests/test_snapshot.py`: un byte cambiado hace fallar `verify_snapshot` | leer | L4 |
| V02 | `QuantAgent-36g` | `data snapshot create` y `verify` desde Yahoo; no pisa un nombre existente | (VM) crea `etf-1d-2026-10` y `verify` da `OK 12 símbolos`; repetir `create` sale con 1 | probar | L4 |
| V03 | `QuantAgent-jcl` | Precios ajustados por splits y dividendos; guarda el cierre sin ajustar | Test con split 2:1 a mano: el ajustado no salta y el sin ajustar sí | leer | L4 |
| V04 | `QuantAgent-r3q` | `data snapshot check`: sesiones contra NYSE, OHLC incoherente, volumen 0, saltos | Test que borra un día y lo nombra; (VM) `faltan 0` en los 12 | probar | L4 |
| V03b | `QuantAgent-5rc` | Nombre correcto del cierre guardado (Yahoo ya ajusta por splits) y `--end` obligatorio en `create` | `git grep -i "sin ajustar\|sin_ajustar" -- quantagent tests` vacío; `create` con `--end` de hoy sale con 1 | leer | L4 |
| V05 | `QuantAgent-1xd` | Fixture `spy-90d-habiles` (sin fines de semana) y `--fixture` en `crossval_*` | `crossval_compare.py --fixture spy-90d-habiles --intrabar` imprime resultado y `Entradas en fin de semana: N` | probar | L4 |
| V06 | `QuantAgent-48n` | El engine avanza por las velas del dato, no por una grilla de calendario | Sobre `spy-90d-habiles`: `TODO DENTRO DE TOLERANCIA` y 0 en fin de semana; referencias de M1 sin cambio | probar | L4 |
| V07 | `QuantAgent-kwi` | `backtest run --snapshot --symbol --from --to` | (VM) rsi/SPY 2007–2018 exit 0 con 6 líneas; golden de M1 verde | probar | L4 |
| V08 | `QuantAgent-43b` | Candado de la reserva (desde 2023-01-01) con registro de aperturas | `--to 2024-01-01` sin `--abrir-reserva` sale con 1; con el flag, el registro suma una línea | probar | L4 |
| V09 | `QuantAgent-5ti` | `backtest verify --snapshot`; las 3 estrategias sobre SPY 2007–2018 | (VM) tres `OK reproducible`; el PR informa trades y tiempo | probar | L4 |
| V10 | `QuantAgent-a03` | Control con backtesting.py sobre SPY real; cuenta velas que tocan stop y take profit | (VM) `TODO DENTRO DE TOLERANCIA` o cada diferencia con su causa | decidir | L4 |
| V10b | `QuantAgent-8to` | El engine pide la historia por velas del dato, no por días de calendario (salteaba 180 de 3020 velas en SPY diario) | (VM) `crossval_compare.py --snapshot etf-1d-2026-10 --symbol SPY --from 2007-01-03 --to 2018-12-31 --intrabar`: misma cantidad de trades en los dos lados; como mucho 30 `Insufficient data`; referencias de M1 sin cambio | decidir | L4 |
| V11 | `QuantAgent-de5` | Comisiones del `PaperBroker` en backtest y CLI (`--commission-pct`, default 0) | Default: referencias sin cambio; con 0.001, PnL menor y `Comisión: 0.10% por lado` | probar | L5 |
| V12 | `QuantAgent-40o` | `recalc_metrics.py --commission-pct` (entrada y salida) | `N/N filas coinciden`, o evidencia de que falta la comisión de entrada (escala) | probar | L5 |
| V13 | `QuantAgent-ld3` | Decisión: perfiles de costos ETF y cripto, con fuente | Doc ≤40 líneas con efecto medido por perfil; el PM reproduce una fila | decidir | L5 |
| V14 | `QuantAgent-dgi` | `--costos <perfil>` | Diferencia de PnL entre perfiles = la de `recalc_metrics.py` (±0.01); sin flag, 340 / 4571.27 | probar | L5 |
| V15 | `QuantAgent-rnl` | Referencia comprar y mantener en el CLI, con los mismos costos y ventana | Test con 3 velas a mano; (VM) aparece `Comprar y mantener:` | probar | L5 |
| V16 | `QuantAgent-bzt` | `recalc_metrics.py --comprar-y-mantener` sin importar `quantagent` | Igual a la línea del CLI sobre spy-90d (±0.01) | probar | L5 |
| V17 | `QuantAgent-cub` | ¿El límite de pérdida diaria actúa en backtest? (`date.today()`) | `xfail(strict=True)` si lo confirma; respuesta con evidencia en el PR | decidir | L5 |
| V18 | `QuantAgent-cbp` | Decisión: tamaño de posición contra la referencia (hoy 5%) y regla del límite diario | Doc ≤40 líneas con una corrida por opción; el PM reproduce una fila | decidir | L5 |
| V12b | `QuantAgent-j62` | Toda orden paga su comisión en efectivo y el PnL del trade neta entrada y salida, en la cartera (PRs #96 y #99) | rsi/spy-90d con 0.001: 2150.48; sin comisión, referencias sin cambio | decidir | L5 |
| V16b | `QuantAgent-cok` | Estrategia diaria `sma-cross` (cruce de medias 50/200) | (VM) SPY 2007–2018: 5 trades; fechas iguales a un cálculo independiente | probar | L5 |
| V16c | `QuantAgent-jgn` | Estrategia diaria `momentum-12m` (momentum absoluto de 12 meses) | (VM) SPY 2007–2018: 3 trades; fechas iguales a un cálculo independiente | probar | L5 |
| V16d | `QuantAgent-i6n` | Adaptador genérico: cualquier `TradingStrategy` corre dentro de `backtesting.py` (`scripts/bt_adapter.py`) | A slippage 0 coincide con el motor trade por trade | probar | L5 |
| V16e | `QuantAgent-80x` | Decisión: qué motor da el número oficial de M2. **Decidido: C**, motor propio con coincidencia obligatoria por celda, más sensibilidad de apertura siguiente (PR #102) | Doc con las cinco estrategias en los dos motores; el PM reproduce una fila de cada uno | decidir | L5 |
| V16f | `QuantAgent-4x6` | Diagnóstico: qué estrategias dependen del largo de la ventana de historia | Script y tabla por estrategia; solo Triple Screen cambia la señal | leer | L5 |
| V16g | `QuantAgent-824` | Tolerancias de fila de `crossval` para SPY real y `--from/--to` en comprar y mantener | (VM) `TODO DENTRO DE TOLERANCIA` con 187 = 187 | probar | L5 |
| V17b | `QuantAgent-oat` | Los cierres no se rechazan por pérdida diaria ni circuit breaker; un cierre no ejecutado corta el backtest | Test que reproduce el rechazo; referencias sin cambio | decidir | L5 |
| V17c | `QuantAgent-8wxb` | Los controles de capital y de tamaño máximo no rechazan la orden que reduce o cierra | (VM) momentum-12m sobre XLK termina con 2 trades | decidir | L5 |
| V19a | `QuantAgent-e18` | `PositionSizer`: regla de tamaño elegible (`confianza` o `fijo`) y hasta 100% (V18 decidió B como convención de medición, PR #103) | Test: 40 y 100 acciones a 5%, 800 y 2.000 a 100%; referencias sin cambio | decidir | L5 |
| V19b | `QuantAgent-3has` | `--tamano-posicion` y `--regla-tamano` en `backtest run` y `verify` | Test: qty en 3 trades; (VM) rsi/SPY a 100% y slippage 0: -4908.13 | probar | L5 |
| V19c | `QuantAgent-oh70` | La entrada que pide todo el efectivo se recorta a lo que alcanza, no se rechaza | (VM) sma-cross/SPY a 100% con `--costos etf`: 5 trades y efectivo >= 0 | decidir | L5 |
| V19d | `QuantAgent-xyf6` | Slippage y `--costos` en el adaptador de `backtesting.py`, con la regla del `PaperBroker` | (VM) rsi/SPY con `--costos etf`: -447.29 y 187 de 187 con el motor | probar | L5 |
| V19e | `QuantAgent-zd9n` | Control motor vs librería: tolerancia de cantidad relativa (1.5e-6, provisoria), no depende del tamaño | (VM) rsi/SPY con `--costos etf`: 187 de 187 al 5% y al 100% | decidir | L5 |
| V19f | `QuantAgent-gti6` | Adaptador de `backtesting.py`: el mismo recorte de la entrada que compra con todo el efectivo | (VM) sma-cross/SPY a 100% con `--costos etf`: 5 trades en los dos y coinciden 5 de 5 | probar | L5 |
| V19g | `QuantAgent-bwqb` | Adaptador con precios a 8 decimales como el motor; bajar la tolerancia de cantidad y la de PnL por trade del control (R: de Fede en #109) | (VM) `compare_trades` da 187 de 187 en rsi/SPY al 5% y al 100%; totales sin cambio | decidir | L5 |
| V20 | `QuantAgent-688` | Golden sobre el snapshot real, solo en la VM: número del motor con `--costos etf` y 100%, junto con la marca "coinciden N de N" contra la librería (opción C) | (VM) pasa y falla con un dígito cambiado; en CI, skipped con motivo | leer | L5 |
| V20b | `QuantAgent-78ja` | Datos locales en `data/local/`, carpeta del repo ignorada por git, con README versionado (pedido de Fede) | `git status` limpio con archivos adentro; el PR trae los comandos para mover, que corre el PM | decidir | L6 |
| V20c | `QuantAgent-im48` | La carpeta de datos locales pasa a `local-data/`: `data/` es de root y la monta el contenedor de QA (corrige V20b) | `git grep "data/local"` sin usos; ningún montaje de `quantagent_qa` incluye `local-data` | decidir | L6 |
| V22 | `QuantAgent-ol11` | Cinco métricas de comparación al final del resumen (PR #107, adelantado; falta traer el lote y cubrir la curva vacía) | `recalc_metrics.py` = CLI en las cinco; referencias sin cambio | probar | L6 |
| V21a | `QuantAgent-j422` | El día del límite de pérdida diaria es el de la vela; apagado en corridas `1d` (decisión del PR #103) | El `xfail` de cub pasa a test normal; (VM) rsi/SPY sigue 187 / -9141.21 | decidir | L6 |
| V21b | `QuantAgent-6zam` | `backtest run` informa el peor día de la corrida | Línea del CLI = `recalc_metrics.py` sobre el `--equity-out` | probar | L6 |
| V23a | `QuantAgent-5nx0` | Los números del resumen de `backtest run` salen de una función reutilizable | Salida de las 3 referencias idéntica byte a byte | leer | L6 |
| V23 | `QuantAgent-opfu` | `backtest compare`: una corrida por par estrategia/activo y tabla en CSV | (VM) 4 estrategias × 12 ETFs: 49 líneas; rsi/SPY 187 / -9141.21 | probar | L6 |
| V23b | `QuantAgent-k6n9` | Columna de control "coinciden N de N" contra `backtesting.py` (opción C) | (VM) rsi/SPY 187 de 187; el PR cuenta cuántas de las 48 celdas coinciden | probar | L6 |
| V23c | `QuantAgent-00cn` | Sensibilidad: PnL con llenado en la apertura siguiente (R: de Fede en #102) | Test con gap a mano; (VM) el PR informa celdas donde cambia la cantidad de trades | probar | L6 |
| V08b | `QuantAgent-bjc1` | Tope del tramo de ajuste: sin `--to` no se lee después de 2018-12-31 y el CLI avisa la fecha asumida; el tramo de prueba exige `--abrir-prueba` (decisión de Fede al revisar #127) | Sin `--to`, ninguna vela desde 2019-01-01 y aviso en stderr; (VM) rsi/SPY sigue 187 / -9141.21 | probar | L6 |
| V23d | `QuantAgent-5f1f` | Sensibilidad: con `next-open` el adaptador recorta la cantidad a lo que alcanza en la apertura (decisión de Fede al revisar #127) | Test con gap a mano donde hoy se cancela la orden; (VM) el PR informa celdas con distinta cantidad de trades, antes (21) y después | probar | L6 |
| V24 | `QuantAgent-skrn` | `compare` escribe markdown con rango, hash del snapshot, costos y parámetros | Dos corridas dan el mismo archivo, con esos 4 datos | leer | L6 |
| V25 | `QuantAgent-bz3z` | Parámetros de estrategia desde un archivo JSON | Dos archivos dan dos filas distintas en `compare` | leer | L6 |
| V26 | `QuantAgent-j2rt` | Agregar una estrategia = un archivo + una línea de registro | Test que registra una sin tocar `quantagent/backtesting/` ni el CLI | leer | L6 |
| V27d | `QuantAgent-w5n9` | Diagnóstico: por qué tardan la suite, los oráculos y cada corrida de `compare` (adelantado por Fede antes del barrido) | Doc en `docs/loop/` con duraciones, perfil de una corrida y tickets propuestos con ahorro estimado | leer | L7 |
| V27e | `QuantAgent-06yj` | La base SQLite aislada de `compare` y `verify` no espera al disco (primer ticket del diagnóstico w5n9) | (VM) CSV de `compare` idéntico byte a byte; el PR informa el tiempo antes y después | probar | L7 |
| V27a | `QuantAgent-ws3d` | El adaptador de `backtesting.py` replica la regla de salto en la apertura del motor (decisión de Fede en #113) | (VM) rsi/EFA, rsi/XLE y rsi/XLF pasan a `coinciden N de N`; ninguna celda deja de coincidir | probar | L7 |
| V27b | `QuantAgent-24qm` | Piso absoluto de 5e-9 en la tolerancia de cantidad del control (decisión de Fede en #113) | (VM) rsi/IWM `coinciden 236 de 236`; rsi/SPY sigue 187 de 187 | probar | L7 |
| V27c | `QuantAgent-u8lf` | Diagnóstico: rsi/GLD da 243 trades al cierre y 242 con apertura siguiente | Doc con el primer trade que diverge y la regla del código que lo explica | leer | L7 |
| V30 | `QuantAgent-yo4s` | Decisión: umbrales del veredicto, supuesto de llenado, tamaño mínimo de orden y costo de cortos, antes de abrir 2019–2022 | Doc con las cuatro decisiones: opciones, dato medido sobre 2007–2018 y recomendación | decidir | L7 |
| V30a | `QuantAgent-wquk` | Motor: tamaño mínimo de orden opcional (`--orden-minima`), decisión de Fede en #140 | Oráculos sin cambios; (VM) rsi/SPY con `--orden-minima 50` sigue 187 / -9141.21 | probar | L7 |
| V30b | `QuantAgent-p3yg` | El adaptador de `backtesting.py` replica el tamaño mínimo de orden | (VM) control de rsi sobre los 12 ETFs con `--orden-minima 50`: 12 celdas `coinciden N de N` | probar | L7 |
| V31a.1 | `QuantAgent-av3i.1` | Cargar y validar grilla de sweep (padre `QuantAgent-av3i`) | RSI 2x2: cuatro combinaciones; rechaza parámetro desconocido y más de 50 | leer | L7 |
| V31a.2 | `QuantAgent-av3i.2` | Comando sweep con base aislada y candado; depende de V31a.1 | CLI inválido sale 1 sin CSV; sin flags para abrir prueba ni reserva | probar | L7 |
| V31a.3 | `QuantAgent-av3i.3` | Integración real de sweep; depende de V31a.2 | RSI 2x2 coincide con defaults; (VM) periodos 14 y 7: 187 / -9141.21 y 325 / -14560.44 | probar | L7 |
| V31b.1 | `QuantAgent-8cry.1` | Registro de cada combinación de sweep en `intentos.csv`, sin reescribir (padre `QuantAgent-8cry`) | Dos sweeps de 4 dejan 8 filas; prefijo intacto; grilla inválida no modifica el registro | probar | L7 |
| V31b.2 | `QuantAgent-8cry.2` | `backtest attempts`: contar corridas y combinaciones; depende de V31b.1 | 8 corridas / 4 combinaciones distintas; filtros por estrategia y activo; lectura sin cambios | probar | L7 |
| V32.1 | `QuantAgent-36y1.1` | Ventanas de ajuste/prueba y selección por Sharpe (padre `QuantAgent-36y1`) | 4 ventanas 4/2 años; sin solapamiento; empate elige primero | leer | L7 |
| V32.2 | `QuantAgent-36y1.2` | Extraer el registro append de sweep para reutilizarlo; depende de V32.1 | Dos sweeps mantienen 8 filas y prefijo intacto | probar | L7 |
| V32.3a | `QuantAgent-36y1.3.1` | Preparar grilla, rango protegido y ventanas; depende de V32.2 | Validación antes de ejecutar; fin de prueba recortado a --to | leer | L7 |
| V32.3b | `QuantAgent-36y1.3.2` | Comando walkforward aislado y CSV; depende de V32.3a | Rango insuficiente sale 1 sin CSV ni intentos; sin flags de apertura; fila por ventana | probar | L7 |
| V32.4 | `QuantAgent-36y1.4` | Selección real por ventana y evidencia VM; depende de V32.3b | Una combinación ganadora por prueba; (VM) tabla de 4 ventanas | probar | L7 |
| V33.1 | `QuantAgent-8sn3` | Regla del Juego 2 (V30) como función pura `verdict.evaluar`; depende de V32.4 | Bordes 0.10 / 0.35 / 2 trades pasan; cada falla da su motivo; valores ausentes no rompen | leer | L7 |
| V33.2 | `QuantAgent-hnlg` | `compare_control.py` agrega Sharpe y drawdown con apertura siguiente = motor + (adaptador next-open − adaptador close) (opción D de Fede, 2026-10-14); depende de V33.1 | Curva a mano con Sharpe y drawdown exactos; columnas vacías si el control no coincide | probar | L7 |
| V33.3 | `QuantAgent-0z29` | `backtest verdict --tabla`: PASA / NO PASA por celda, al cierre y con apertura siguiente; depende de V33.2 | Tabla a mano de 4 filas con veredictos y motivos exactos; (VM) rsi 12 ETFs 2007–2018 `PASA: N de 12` | probar | L7 |
| V34.1 | `QuantAgent-8xcd` | Robustez en el veredicto: concentración del PnL y PnL por año desde los trades; depende de V33.3 | Un trade que explica todo da 1.0; sin `--trades-dir` la salida no cambia | leer | L7 |
| V34.2 | `QuantAgent-ghdf` | `sweep --vecinos-20`: default y ±20% por parámetro, dentro del ajuste; depende de V34.1 | rsi_period 11 y 17; (VM) la fila default da 187 / -9141.21 | probar | L7 |
| V35 | `QuantAgent-750o` | Sharpe deflactado (Bailey y López de Prado) como función pura; depende de V33.3 | Valor a mano con tolerancia 1e-6; más intentos, menor DSR | leer | L7 |
| V36 | `QuantAgent-0oqm` | `scripts/recalc_verdict.py` sin importar `quantagent`; depende de V33.3 | `coinciden 4 de 4`; un veredicto cambiado a mano da exit 1 | probar | L7 |
| V36a.1 | `QuantAgent-7ssw.1` | `compare --equity-dir`: curva de equity del motor por celda | Filas = curva del motor; última equity = capital + pnl; sin opción, salida idéntica | leer | L7 |
| V36a.2 | `QuantAgent-7ssw.2` | `verdict --equity-dir`: p-valor del t-test de retornos diarios (informativo); depende de V36a.1 | p-valor a mano 1e-9; None con 2 retornos o curva constante | leer | L7 |
| V36a.3 | `QuantAgent-7ssw.3` | `verdict --equity-dir`: Monte Carlo por bloques, percentil final y DD p95 (informativo); depende de V36a.1 | Semilla determinista; retornos constantes dan 1.0 / 0.0 | leer | L7 |
| V36a.4 | `QuantAgent-7ssw.4` | `compare --crisis`: 2008, 2020 y 2022 contra comprar y mantener (informativo); depende de V36a.1 | Valor a mano 1e-6; año incompleto vacío; (VM) rsi/SPY crisis_2008 | leer | L7 |
| V33.2d | `QuantAgent-jzb6` | Diagnóstico de la diferencia entre la curva de equity del motor y la del adaptador con los mismos trades; no bloquea el veredicto | Primera fecha de diferencia y causa, o lo descartado; sin cambios de código | leer | L7 |
| V33.2e | `QuantAgent-mxy7` | `bt_adapter.equity_curve(stats)`: con llenado al cierre, el costo del llenado en su vela; `compare_control` la usa (fix de jzb6) | La curva del adaptador iguala a la del motor (1e-4 USD) con entradas y salidas por regla; (VM) ajustes de rsi/SPY antes y después | probar | L7 |
| V33.2f | `QuantAgent-lba3` | Adaptador acepta entradas en corto al 100% con comisión > slippage, como el motor (fix de n51l) | rsi spy-2y-1d `--costos cripto` coinciden 46 de 46; largos sin cambio | leer | L7 |
| F01 | `QuantAgent-gfs1` | Decisión: llenado por defecto del motor en la apertura siguiente; tabla de las 48 celdas con los dos llenados | Doc ≤60 líneas con la tabla y "Decidir: A o B"; el PM reproduce una fila | decidir | L8 |
| F02 | `QuantAgent-d4cq.1` | Perfil de las 4 estrategias en el motor propio, SPY 2007–2018 (diagnóstico) | Top 10 por estrategia; causa en el top 10; sin cambios en `quantagent/` | leer | L8 |
| F03 | `QuantAgent-d4cq.2` | Corregir el cuello de botella del perfil; depende de F02 | Oráculos y métricas sin cambio; la más lenta 3 veces más rápida | probar | L8 |
| F03a | `QuantAgent-gfs1.1` | Motor: llenado por defecto en la apertura siguiente, `--llenado cierre` como opción (decisión A de Fede en F01) | Entrada y salida al `open` de t+1 a mano; con `cierre`, 187 / -9141.21 | probar | L8 |
| F03a.2a | `QuantAgent-gfs1.1.2.1` | CLI: `--llenado apertura\|cierre` explícito; todavía cierre por defecto | Ambos comandos propagan fill; inválido sale 2; snapshot conserva fill | probar | L8 |
| F03a.2b | `QuantAgent-gfs1.1.2.2` | `--llenado` de run y compare pasa a apertura por defecto; `Backtest` y `correr_par` quedan en cierre; tests del CLI que dependen del cierre lo piden explícito | Run sin flag llena al open siguiente; oráculos sin números nuevos | probar | L8 |
| F03a.2c | `QuantAgent-gfs1.1.2.3` | Diagnóstico: por qué difiere la cantidad del control con apertura (#188) | Causa y corrección propuesta reproducibles | leer | L8 |
| F03a.2d | `QuantAgent-gfs1.1.2.4` | Adaptador: con apertura siguiente dimensiona con el capital y el precio de la apertura, como el motor | Fila 1 de rsi/SPY con la misma qty; al cierre sin cambio | leer | L8 |
| F03a.2e | `QuantAgent-gfs1.1.2.5` | `compare_control --llenado`, apertura por defecto; depende de F03a.2d | rsi/SPY 2007–2018 coinciden 187 de 187 con apertura y con cierre | probar | L8 |
| F03a.2f | `QuantAgent-gfs1.1.2.6` | Diagnóstico: control con apertura 32 de 48 (11 en la última fila, 5 en salidas intermedias) | Causa por grupo y corrección propuesta; reproducción en la VM | leer | L8 |
| F03a.2g | `QuantAgent-gfs1.1.2.7` | Adaptador con apertura: liquidación final al cierre con `backtest_end` | 8 celdas del grupo última fila pasan a coinciden; cierre 48 de 48 | leer | L8 |
| F03a.2h | `QuantAgent-gfs1.1.2.8` | Adaptador con apertura: gap en la vela de entrada; depende de F03a.2g | 5 celdas de salidas intermedias pasan a coinciden; cierre 48 de 48 | leer | L8 |
| F03a.2i | `QuantAgent-gfs1.1.2.9` | Motor con apertura: stop y take profit desde el precio llenado (decisión A de Fede, 2026-10-15) | Niveles reescalados a mano; cierre sin cambio; oráculos pasan | probar | L8 |
| F03a.2j | `QuantAgent-gfs1.1.2.10` | Adaptador con apertura: stop y take profit reescalados como el motor; depende de F03a.2i | Mismo caso a mano; 48 celdas sin retroceso | leer | L8 |
| F03a.2k | `QuantAgent-gfs1.1.2.11` | Control: tolerancia de pnl escalada por qty y `QTY_ATOL` 1e-8 por el redondeo a 8 decimales con apertura; depende de F03a.2j | 48 celdas: apertura 48 de 48 y cierre 48 de 48 | leer | L8 |
| F03b.1 | `QuantAgent-gfs1.2.1` | Golden spy-90d en apertura (padre gfs1.2) | CSV exacto y métricas regeneradas; comando documentado | probar | L8 |
| F03b.2 | `QuantAgent-gfs1.2.2` | Golden SPY 2007–2018, cierre explícito y recálculo; depende de F03b.1 | Apertura control 48/48; cierre 187 / -9141.21; recálculo coincide; tabla viejo → nuevo | probar | L8 |
| F09.4 | `QuantAgent-d4cq.7.4` | `bar_signals` pasa solo `required_history_bars` velas, como el motor (condición de la decisión B de Fede, 2026-10-15) | Estrategia de prueba nunca ve más de n velas; truncamiento verde; filtro total < 764 s y Spearman apertura >= 0,8 | leer | L9 |
| F03b.3 | `QuantAgent-gfs1.2.3` | Resolver alcance de oráculos legacy; depende de F03b.2 | Inventario completo, oráculos verdes y decisión explícita sobre referencias al cierre | decidir | L9 |
| F03b.4 | `QuantAgent-gfs1.2.4` | Golden SPY 2007–2018 congelado fuera del repo para apertura y cierre, byte a byte (el PM genera el de apertura) | Mutación de una fila del golden detectada | leer | L9 |
| F03b.5 | `QuantAgent-gfs1.2.5` | `sweep` y `walkforward` con `--llenado`, apertura por defecto | Default apertura; cierre usado en ajuste y prueba | leer | L9 |
| F05 | `QuantAgent-d4cq.3` | `bar_signals`: señales por vela desde una estrategia, solo con velas hasta la actual | La estrategia recibe i+1 velas en la vela i; cruce de sma-cross a mano | leer | L8 |
| F06 | `QuantAgent-d4cq.4` | `check_no_lookahead`: prueba de truncamiento; depende de F05 | 4 estrategias sin diferencias; `shift(-1)` y media global detectados | leer | L8 |
| F07 | `QuantAgent-d4cq.5` | `run_filter` en `vectorbt`: apertura siguiente fija y costos; depende de F05 | Llenado al `open` de la vela siguiente, PnL a mano 1e-9; sin modo al cierre | leer | L8 |
| F08 | `QuantAgent-d4cq.6` | Stops dentro de la vela en `run_filter`; gana el stop; depende de F07 | Stop, gap, stop+TP en la misma vela y trailing a mano | leer | L8 |
| F08b | `QuantAgent-d4cq.8` | Trailing stop de `run_filter` con la regla del motor: máximo de cierres, salida en la apertura siguiente; depende de F08 | Trailing y stop fijo a mano; rsi/SPY contra el motor en el resumen | leer | L8 |
| F09.1 | `QuantAgent-d4cq.7.1` | Script de medición del filtro, 48 celdas; partición de F09 | CSV con métricas y tiempos de señales/simulación; test excluye fechas posteriores a 2018 | leer | L8 |
| F09.2 | `QuantAgent-d4cq.7.2` | Spearman contra compare apertura y cierre; depende de F09.1 | Spearman +1/-1 y empates; 48 claves iguales; top cinco por llenado | leer | L8 |
| F09.3 | `QuantAgent-d4cq.7.3` | Medición VM y decisión B o A; depende de F09.2 | Pared motor apertura y filtro, ambos Spearman y top cinco reproducibles; regla vigente de Fede | decidir | L8 |
| H01 | `QuantAgent-d4cq.9` | Archivo de hipótesis YAML y cargador (nube) | YAML válido carga; campo faltante o id inválido falla nombrando el campo | leer | L9 |
| H02 | `QuantAgent-d4cq.10` | Admisión etapa 0: justificación, máx. 6 parámetros libres, universo, truncamiento (nube); depende de H01 | Motivos juntos; estrategia que mira el futuro rechazada | leer | L9 |
| H03 | `QuantAgent-d4cq.11` | `intentos.csv` con columnas `hipotesis` y `etapa`, compatible con el encabezado viejo (nube) | Filas viejas intactas; conteo por hipótesis | leer | L9 |
| H04 | `QuantAgent-d4cq.12` | Candado por hipótesis para prueba y reserva (nube) | Segunda apertura del mismo par falla | leer | L9 |
| H05 | `QuantAgent-d4cq.13` | `filtrar_activo`: walk-forward por activo con `run_filter` (nube); depende de H01 y F09.4 | Ganador por ventana elegido en el ajuste; la prueba no ve la ventana siguiente | leer | L9 |
| H06.1 | `QuantAgent-d4cq.14.1` | CLI de hipótesis: carga, rango 2007–2018 y admisión; padre H06, depende de H02, H03, H05 | Rechazo sale 1 con motivos e intento etapa 0; no admite velas desde 2019 | leer | L9 |
| H06.2a | `QuantAgent-d4cq.14.2.1` | Filtro por activo, intentos etapa 1 y tabla; depende de H06.1 | Integración real de dos activos: PASA, tabla e intentos; solo 2007–2018 | probar | L9 |
| H06.2b | `QuantAgent-d4cq.14.2.2` | Bordes del veredicto etapa 1; depende de H06.2a | Nueve de doce bajo 0,10 cortan; ocho pasan; 0,10 no cuenta como bajo | leer | L9 |
| H06.3 | `QuantAgent-d4cq.14.3` | Evidencia VM del CLI RSI con dos combinaciones; depende de H06.2 | Tabla de doce ETFs, veredicto, tiempo e incremento de doce intentos | probar | L9 |
| H07 | `QuantAgent-d4cq.15` | Punto de control L9: dos hipótesis de punta a punta (loop, VM); depende de H06 | Doc con veredicto, tiempos y conteo de intentos | leer | L9 |
| E01 | `QuantAgent-d4cq.16` | Etapa 1: corte si más de 2/3 bajo 0,10; menos de 30 trades cuenta como bajo (nube) | 12 activos: 8 pasa, 9 corta; 6: 4 pasa, 5 corta; 29 trades es bajo | leer | L10 |
| E02 | `QuantAgent-d4cq.17` | `parametros_elegidos` y `filtrar --out` (nube); depende de E01 | Más elegido gana, empate el más reciente; CSV por activo | leer | L10 |
| E03 | `QuantAgent-d4cq.18` | Estado por hipótesis en `hipotesis.csv` (nube) | Dos registros, último estado; valores inválidos no escriben | leer | L10 |
| E04 | `QuantAgent-d4cq.19` | `correr_motor`: compare por activo con sus parámetros e intentos por etapa (nube); depende de E02 | Cada activo con sus parámetros; 2 intentos con hipótesis y etapa | leer | L10 |
| E05 | `QuantAgent-d4cq.20` | Etapa 2: control contra `backtesting.py` y V30 por activo (nube); depende de E04 | `sin validar` corta; todos NO PASA corta | leer | L10 |
| E06 | `QuantAgent-d4cq.21` | `dsr_contra_intentos` (nube) | Igual al cálculo a mano; más intentos, menor DSR | leer | L10 |
| E07a | `QuantAgent-d4cq.22.1` | Extraer resultados reutilizables de admisión y filtro; partición de E07 | `filtrar` conserva salida, intentos y CSV; resultados reutilizables comprobados | leer | L10 |
| E07b | `QuantAgent-d4cq.22.2` | `correr`, etapas 0 y 1 con estado; depende de E07a | Rechazo deja una fila; corte del filtro deja dos; sin compare | leer | L10 |
| E07c | `QuantAgent-d4cq.22.3` | Encadenar etapa 2 con parámetros elegidos; depende de E07b | Control real, tres filas de estado y salida según veredicto | leer | L10 |
| E08a | `QuantAgent-d4cq.23.1` | Validar etapa 3: motor 2019–2022, V30 y DSR por activo; partición de E08, depende de E06, E07 | V30 PASA y DSR ≥ 0,95; basta un activo; intentos etapa 3 | leer | L10 |
| E08b | `QuantAgent-d4cq.23.2` | Conectar etapa 3 y candado al CLI; depende de E08a | Candado antes de leer prueba; tabla y cuarto estado; corte previo no abre | leer | L10 |
| E08c | `QuantAgent-d4cq.23.3` | Integración real y segunda apertura; depende de E08b | Cuatro estados y candado; segunda corrida sin compare de prueba; nada desde 2023 | leer | L10 |
| E09 | `QuantAgent-d4cq.24` | Punto de control L10: las dos hipótesis de L9 por el embudo (VM); depende de E08 | Doc con etapa, motivo, tiempos y conteos | probar | L10 |
| I01 | `QuantAgent-d4cq.25` | Alarma filtro vs motor: regla pura | 8 con filtro ≥ 0,10: 4 con motor ≤ 0 alarma, 3 no | leer | L11 |
| I02 | `QuantAgent-d4cq.26` | La alarma frena `correr` antes de la etapa 3; depende de I01 | Etapa 2 ALARMA, candado sin fila | leer | L11 |
| I03 | `QuantAgent-d4cq.27` | ERROR registrado si falla la etapa 2 o 3 | Fila ERROR; candado usado en etapa 3; la excepción sale | leer | L11 |
| I04 | `QuantAgent-d4cq.28` | `correr` no repite etapas con la prueba ya abierta | 0 intentos y 0 estados nuevos | leer | L11 |
| I05 | `QuantAgent-d4cq.29` | `next-local-data`: Codex escribe en `local-data` una corrida (VM) | Con archivo, `--add-dir`; sin él, comando igual | leer | L11 |
| I06 | `QuantAgent-d4cq.30` | Informe HTML por hipótesis | Etapa, motivo y activos; sin precios | leer | L11 |
| I07 | `QuantAgent-d4cq.31` | Resumen de la tanda | Conteos por etapa y veredicto; causas agrupadas | leer | L11 |
| I08 | `QuantAgent-d4cq.32` | `backtest hipotesis tanda`; depende de I04, I06, I07 | 3 YAML, uno corta y uno lanza: 3 informes o logs y resumen | leer | L11 |
| I08b | `QuantAgent-d4cq.35` | `tanda` saltea hipótesis con veredicto final; depende de I08, agregado por el PM el 2026-10-17 al revisar #248 | CORTA/ALARMA/PASA etapa 3 se saltean; ERROR y sin estado corren | leer | L11 |
| I09 | `QuantAgent-d4cq.33` | Hipótesis máximo de 52 semanas y momentum 12m | La admisión acepta las dos | leer | L11 |
| I10 | `QuantAgent-d4cq.34` | Punto de control L11: tanda de `hipotesis/` (VM); depende de I01–I09 | Informe por hipótesis; rsi y sma sin intentos nuevos | probar | L11 |
| C01 | `QuantAgent-d4cq.37` | `sharpe_base` en `intentos.csv`; etapas 2 y 3 a tasa 0; varianza del DSR sobre filas tasa 0 (decisión de Fede) | Filas viejas intactas; N = total | leer | L12 |
| C02 | `QuantAgent-d4cq.46` | YAML de cartera: piernas con clase, parámetros, activos y universo; formato viejo = 1 pierna | 2 piernas cargan; 8 parámetros libres cortan | leer | L12 |
| C03 | `QuantAgent-d4cq.47` | Universo amplio: ETFs con primera vela ≤ 2007-01-03 (VM; el PM baja el snapshot antes) | `snapshot verify` del amplio pasa | leer | L12 |
| C04 | `QuantAgent-d4cq.48` | Pesos por igual riesgo (1/σ de 60 sesiones, mensual) y retorno de cartera (nube) | 1% y 2% → 2/3 y 1/3; no mira el futuro | leer | L12 |
| C05a | `QuantAgent-d4cq.49` | `filtrar_activo` devuelve los retornos fuera de muestra (nube) | Sharpe desde la serie = `sharpe_tasa_0` | leer | L12 |
| C05b.1 | `QuantAgent-d4cq.50.1` | Cálculo de etapa 1 para cartera (`evaluar_cartera` y veredicto); depende de C01, C02, C04, C05a | 24 trades corta por trades; Sharpe individual coincide | leer | L12 |
| C05b.2 | `QuantAgent-d4cq.50.2` | Integración de etapa 1 de cartera en CLI y registro D7; depende de C05b.1 | 12 × 5 trades pasa; 2 piernas × 3 activos agrega 1 fila | leer | L12 |
| C06.1 | `QuantAgent-d4cq.51.1` | Combinación de etapa 2 y V30 de cartera; depende de C05b | V30 sobre curva combinada; celda que no coincide corta | leer | L12 |
| C06.2 | `QuantAgent-d4cq.51.2` | Alarma de cartera, CLI etapa 2 y registro D7; depende de C06.1 | Alarma dispara y no dispara; 1 fila en intentos.csv | leer | L12 |
| C07 | `QuantAgent-d4cq.52` | Etapa 3: V30 y DSR de cartera (VM); depende de C06 | DSR 0,96 pasa, 0,94 corta | leer | L12 |
| C08 | `QuantAgent-d4cq.53` | Informe y resumen de cartera; causas por hipótesis; depende de C07 | Universos y fila de cartera; sin precios | leer | L12 |
| C09 | `QuantAgent-d4cq.54` | Primera cartera: momentum-12m + tendencia de Faber sobre el universo amplio; depende de C02, C03 | Admisión acepta; 0 parámetros libres | leer | L12 |
| C10 | `QuantAgent-d4cq.55` | Punto de control L12 (VM, lo corre el PM; etapa 3 con OK de Fede) | Veredicto e informe de la cartera | probar | L12 |
| C11.1 | `QuantAgent-d4cq.76.1` | Loop: lote y ticket por variable o argumento (`LOOP_LOTE`, `LOOP_TICKET`) | Dos lotes borrador: con `LOOP_LOTE` toma el indicado; ticket no elegible dice por qué | leer | L12 |
| C11.2 | `QuantAgent-d4cq.76.2` | Loop: reclamo por ticket con CLAIM y RELEASE; depende de C11.1 | Dos CLAIM del mismo ticket en <6h el segundo no lanza; con RELEASE sí | leer | L12 |
| M01.1 | `QuantAgent-d4cq.69.1` | Módulo puro `fabrica/regimen.py`: `clasificar_regimen`, `sharpe_por_regimen` y `permutar_regimenes_bloques` | Régimen con datos hasta t-1; Sharpe por categoría; permuta bloques de un mes | leer | L14 |
| M01.2 | `QuantAgent-d4cq.69.2` | Medición de la cartera núcleo por régimen 2007–2018 y contraste contra azar; depende de M01.1 | Tabla de Sharpe diario por régimen 2007–2018; permutación en bloques de un mes; conclusión se sigue o se archiva | decidir | L14 |
| M07 | `QuantAgent-d4cq.75` | Punto de control de la capa macro (VM, lo corre el PM; `--margen` lo fija Fede antes de correr) | Veredicto de cada capa con ΔDSR contra el margen | probar | L14 |
| G04a | `QuantAgent-d4cq.39` | Catálogo semilla: 15 de momentum, reversión y tendencia (nube) | Ids únicos, fuente verificable | leer | L13 |
| G04b | `QuantAgent-d4cq.40` | Catálogo semilla: 15 de estacionalidad, volatilidad y régimen (nube); depende de G04a | Ídem; 30 en total | leer | L13 |
| G05 | `QuantAgent-d4cq.41` | Crítico determinista y familia en `hipotesis.csv`; depende de G03 | Duplicada corta en etapa 0 sin intentos | leer | L13 |
| G06a | `QuantAgent-d4cq.42` | Generador: pedido y lectura de respuesta, sin LLM | 3 válidos de 5 bloques; no pisa | leer | L13 |
| G06b | `QuantAgent-d4cq.43` | `backtest hipotesis proponer` vía `QUANTAGENT_LLM_CMD` (VM); depende de G06a | Comando falso: 2 archivos y log | leer | L13 |
| G07 | `QuantAgent-d4cq.44` | Resumen cuenta causas por hipótesis; ERROR con la última línea | 3 activos de una hipótesis cuentan 1 | leer | L13 |
| P05.4 | `QuantAgent-d4cq.63` | Etapas 2 y 3 con piernas transversales: `SenalExterna` y celdas transversales en `validar_etapa2`/`validar_etapa3`; depende de P05.3 (hecho) | Una pierna transversal alimenta a los dos motores con la misma serie 0/1; `senal-externa` no es válida en un YAML | leer | L13 |
| P09 | `QuantAgent-d4cq.67` | Motivo de ERROR con la última línea no vacía de la excepción, `<Tipo>: `, truncada a 200 | ERROR con excepción de 3 líneas muestra la última | leer | L13 |
| P10 | `QuantAgent-d4cq.68` | Punto de control L13 (VM, lo corre el PM sobre el snapshot amplio `etf-1d-2026-10-amplio`; Fede dio OK por cartera el 2026-10-19 para abrir 2019–2022 en las carteras que pasen la etapa 2); depende de P05.4 y P09 | Cada entrada `destino: L13` con veredicto e informe o `motivo`; el informe muestra N de intentos y DSR | probar | L13 |
| K01 | `QuantAgent-kw84` | Loop: el gate toma la entrega previa solo del lote activo (`LOOP_LOTE`) | Dos PRs loop de lotes distintos: con `LOOP_LOTE=lote/a` evalúa solo el de lote/a; un cambio sobre lote/b no activa modo cambio en lote/a | leer | L15 |
| E01 | `QuantAgent-w12l` | Limpiar `intentos.csv`: copia fechada y retiro de las 10 filas `sintetico`; IM con hashes (corrida con `next-local-data`) | `grep -c sintetico local-data/intentos.csv` da 0; copia fechada con el sha256 original; el resto de las filas igual; DSR de la etapa 2 de `cartera-transversal` ya no da 0,00 por el SR0 | leer | L16 |
| E02 | `QuantAgent-hzkq` | Las pruebas sintéticas no escriben en el registro real y `dsr_contra_intentos` ignora Sharpe no finitos o `sintetico` | `hipotesis correr --snapshot sintetico` deja `intentos.csv` con el mismo hash; fila con Sharpe 1e13 e `inf` no cambia el DSR | leer | L16 |
| E03 | `QuantAgent-gziw` | N efectivo: `n_efectivo(filas)` por `(estrategia, params, activo, desde, hasta)` y `n_intentos` en `dsr_contra_intentos` | 5 filas con 2 idénticas dan `n_efectivo = 4`; DSR con N menor es mayor o igual que con N crudo | leer | L16 |
| E05 | `QuantAgent-txun` | `sharpe_por_anio` y `anios_positivos` sobre una curva ya calculada | Curva de 4 años con 3 positivos da `3 de 4`; un año con <30 sesiones se descarta e informa | leer | L16 |
| E06 | `QuantAgent-344q` | `recargo_paper(dsr)`: (50 %, 12 sem) bajo 0,50; (75 %, 6) hasta 0,95; (100 %, 0) desde 0,95 | Test de los tres tramos y de los bordes; `None` da (50, 12) | leer | L16 |
| E04 | `QuantAgent-cjh3` | Criterio de pase de la etapa 3: Sharpe > 0, mayor al benchmark equiponderado en etapas 2 y 3, ≥ 3 de 4 años positivos y drawdown V30; el DSR se informa y no corta | Cartera con DSR 0,00 que cumple lo demás pasa; con 2 de 4 años positivos corta con motivo; menor al benchmark corta aunque el DSR sea alto | decidir | L16 |
| E07 | `QuantAgent-haua` | Informe con N crudo y efectivo, DSR con ambos, años positivos, benchmark y recargo; glosario sin 'pasa con 0,95' | El HTML de una fila de etapa 3 muestra las cuatro cifras y el recargo | probar | L16 |
| E08 | `QuantAgent-n96l` | Punto de control L16: recalcular las tres carteras que cortaron con el criterio nuevo (curvas guardadas, sin abrir 2019–2022); IM | El IM dice cuáles pasan y por qué; ninguna corrida nueva sobre 2019–2022; `pytest -m oraculo` verde | decidir | L16 |
| S01a | `QuantAgent-lwji.1` | M1: cálculo de Sharpe en exceso con DGS3MO compartida; partición de S01 (`QuantAgent-lwji`) | Curva que gana la tasa da ≈ 0; tasa variable compartida sin anticipación; cobertura faltante explícita | leer | L19 |
| S01b.1 | `QuantAgent-lwji.2.1` | M1: diagnóstico de solo lectura desde curvas guardadas; partición de S01b; depende de S01a | Dos Sharpes con S01a; registros y DSR idénticos; ausencia explícita; sin motor ni fechas nuevas | leer | L19 |
| S01b.2 | `QuantAgent-lwji.2.2` | M1: mostrar diagnóstico en HTML y conectar CLI/tanda; depende de S01b.1 | HTML con cartera y benchmark en exceso; CLI sin diagnóstico funciona; registros idénticos | probar | L19 |
| S02 | `QuantAgent-jnxy` | M2: benchmark con los pesos de la cartera, siempre invertido, junto a 1/N; no corta (notebook, Codex) | Tests sintéticos con brecha conocida por construcción; 1/N sigue siendo el corte | leer | L19 |
| S03 | `QuantAgent-057q` | M3: error estándar de ΔSharpe con largo de bloque fijo en el código (notebook, Codex) | En datos i.i.d. coincide con √((2−2ρ)/T); el informe lo muestra junto a la cifra puntual | leer | L19 |
| S04 | `QuantAgent-v2ll` | Atribución de las cuatro carteras en pesos, señal y tasa con curvas guardadas (VM, después de L18); depende de S01–S03 | Tabla por cartera; md5 de `intentos.csv`, `hipotesis.csv` e `hipotesis-candado.csv` iguales antes y después; ninguna ventana nueva | decidir | L19 |
| S05 | `QuantAgent-syml` | Catálogo con `tema` (lista cerrada), `muestra`, `fidelidad`, `decaimiento` y sección de descartadas con cita (notebook, Codex) | Carga sin errores; toda entrada tiene tema válido; las cuatro descartadas tienen DOI | leer | L19 |
| S06.1a | `QuantAgent-nb94.1.1` | Validar los campos nuevos de propuestas y sus listas cerradas; parte de nb94.1, depende de S05 | Faltantes y valores fuera de lista se rechazan; pruebas dirigidas verdes | leer | L19 |
| S06.1b | `QuantAgent-nb94.1.2` | Completar las tres propuestas existentes con el contrato nuevo; parte de nb94.1, depende de S06.1a | Las tres propuestas completas pasan validación local; escritura rechaza una incompleta | leer | L19 |
| S06.2 | `QuantAgent-nb94.2` | Validar DOI contra Crossref e integrar respuesta en la escritura; depende de S06.1b | DOI correcto pasa; cita ajena se rechaza; sin red queda "sin verificar" y no escribe propuestas | probar | L19 |
| S07.1 | `QuantAgent-whq0.1` | Mostrar advertencias en etapa 0 y pedido; no bloquean (hijo de S07) | Etapa 0 y pedido las incluyen; validación puede pasar con advertencias | leer | L19 |
| S07.2 | `QuantAgent-whq0.2` | Derivar advertencias estructurales desde fidelidad, muestra y composición de piernas (hijo de S07) | Las condiciones activas producen sus códigos; una propuesta fiel sin solapamiento ni overlay solo largo no tiene advertencias | leer | L19 |
| S07.3 | `QuantAgent-whq0.3` | Mostrar advertencias en informe y verificar las cuatro carteras conocidas (hijo de S07) | Informe las muestra; las cuatro carteras disparan lo esperado y una propuesta fiel no dispara ninguna | probar | L19 |
| S08 | `QuantAgent-55nh` | `armar_pedido` nuevo con criterio, inventario desde manifiestos y lista negra (notebook, Codex); depende de S05 y S06 | El texto contiene criterio, clases, activos y lista negra; sigue sin Sharpe, DSR ni ganancias | leer | L19 |
| S09 | `QuantAgent-cq4u` | Punto de control L19 (VM, lo corre el PM): informe de las cuatro carteras con M1–M3 y pedido en seco, sin LLM; depende de S01–S08 | IM con las cuatro filas recalculadas; ninguna ventana nueva; `pytest -m oraculo` verde | decidir | L19 |
| U1 | `QuantAgent-0flc` | Salida unificada 1/5 (beo, VM): razones de salida como constantes en un solo lugar; el scheduler emite mayúsculas | Caso 4 de `test_exit_rule_divergence.py` con igualdad; backtest sin cambio | leer | L19 |
| U2 | `QuantAgent-ln4v` | Salida unificada 2/5 (beo, VM): paper abre con política, trailing y max hold de la señal y TP 3 %; depende de U1 | Señal con trailing 5 % deja `trailing_stop_pct == 0.05` y política `TRAILING_STOP`; backtest sin cambio | leer | L19 |
| U3 | `QuantAgent-3fjf` | Salida unificada 3/5 (beo, VM): particionada en U3a y U3b; depende de U2 | Ver `docs/loop/particiones/QuantAgent-3fjf.md` | decidir | L19 |
| U3a | `QuantAgent-3fjf.1` | Scheduler delega en `strategy.should_exit`; adaptador del piloto usa regla base; depende de U2 | Casos 1 y 3 con igualdad; no existe `_check_exit_conditions` | decidir | L19 |
| U3b | `QuantAgent-3fjf.2` | Migrar tests del scheduler tras eliminar `_check_exit_conditions`; depende de U3a | Archivo de tests sin referencias al método y pruebas verdes | leer | L19 |
| U4 | `QuantAgent-dfa4` | Salida unificada 4/5 (beo, VM): particionada en U4a y U4b; depende de U3 | Ver `docs/loop/particiones/QuantAgent-dfa4.md` | decidir | L19 |
| U4a | `QuantAgent-dfa4.2` | Mover `check_intrabar_stops` a módulo neutral y reexportarla desde backtest; mantener parche por nombre de crossval | `test_intrabar_stops.py` y crossval intravela pasan sin cambiar resultados | leer | L19 |
| U4b | `QuantAgent-dfa4.1` | Paper evalúa la función en la última vela cerrada y cierra por el camino existente; depende de U4a | Caso 2 cierra a 90 en ambos caminos con cierre de vela 95 | decidir | L19 |
| U5 | `QuantAgent-5fh4` | Salida unificada 5/5 (beo, VM): particionada en U5a y U5b; depende de U4 | Ver `docs/loop/particiones/QuantAgent-5fh4.md` | decidir | L19 |
| U5a | `QuantAgent-5fh4.1` | Extraer a módulo neutral la regla de salida al cierre y mantener `should_exit` delegando | Pruebas dirigidas conservan razón y precio para entradas al cierre | leer | L19 |
| U5b | `QuantAgent-5fh4.2` | Integrar salida intravela y al cierre en función pura; backtest y paper delegan; depende de U5a | Los cuatro casos de divergencia y goldens/crossval pasan con números invariantes; oráculos verdes | decidir | L19 |
| U5b.1 | `QuantAgent-5fh4.2.1` | Unir evaluación intravela y al cierre en función pura de posición y vela | Divergencia valida razón y precio en cuatro casos | leer | L19 |
| U5b.2 | `QuantAgent-5fh4.2.2` | Delegar backtest/replay y comprobar invariantes; depende de U5b.1 | Divergencia, goldens, crossval y oráculos pasan | probar | L19 |
| I01 | `QuantAgent-ui1p` | Ficha de postmortem: `ficha_postmortem(h, filas)` clasifica la causa de corte (`pocos_trades`, `costos_comen_ganancia`, `rotacion_alta`, `una_pierna_domina`, `drawdown`) con umbrales constantes y escribe `catalogo/postmortem/<id>.yaml` (nube) | Cartera de pocos trades da `pocos_trades`; el YAML no contiene Sharpe, DSR ni pnl del holdout (por nombre y por valor) | leer | L17 |
| I02 | `QuantAgent-fqye` | Chequeo de propuestas: validador de `catalogo/propuestas/<id>.yaml` (hipótesis económica, fuente con DOI, universo con criterio, 0 parámetros libres, campo `idea`); rechaza la tercera variante de una `idea` (nube) | Sin DOI y sin hipótesis se rechazan con motivo; variantes 1 y 2 pasan, la 3 no | leer | L17 |
| I03 | `QuantAgent-xiya` | Chequeo de originalidad estructural (señal, universo, ventana) contra el catálogo y las hipótesis probadas; depende de I02 (nube) | Idéntica a una existente se marca parecida; cambiar solo la clase de activo no | leer | L17 |
| I07 | `QuantAgent-qr1n` | `scripts/retrospectiva_lote.py`: tabla de causas de corte por cartera y de intentos del lote; depende de I01 (nube) | 3 fichas sintéticas: la tabla cuenta cada causa; sin Sharpe ni DSR | leer | L17 |
| I04 | `QuantAgent-d4cq.42` | Investigador-1: arma el pedido al LLM (catálogo, reglas D1 y D2, N de intentos, fichas de I01) y escribe cada propuesta en `catalogo/propuestas/`, nunca en `hipotesis/`; depende de I01 e I02 (nube; el LLM real no se llama) | Respuesta simulada escribe N archivos; el pedido no contiene cifras de Sharpe, DSR ni pnl; YAML inválido no escribe nada | leer | L17 |
| I05 | `QuantAgent-d4cq.43` | `quantagent backtest hipotesis proponer` vía `QUANTAGENT_LLM_CMD`, pasa por I04, I02 e I03 y lista aceptadas y rechazadas con motivo; depende de I04 e I03 (VM) | Con un `QUANTAGENT_LLM_CMD` de mentira lista la propuesta aceptada; con uno que falla sale con error claro sin escribir; no ejecuta ninguna hipótesis | leer | L17 |
| I06 | `QuantAgent-z67w` | `hipotesis propuestas aprobar <id>` mueve a `hipotesis/` y deja el registro del OK; `rechazar <id> --motivo` archiva; depende de I05 | Aprobar mueve y registra; una no aprobada no la toma `tanda`; rechazar sin motivo falla | leer | L17 |
| I09 | `QuantAgent-o6y9` | Celdas de la etapa 2 por pierna y activo: `validar_etapa2` escribe `celdas.csv` (trades, pnl, max_drawdown, sin Sharpe) y `filas_postmortem(carpeta)` las lee para `ficha_postmortem`; agregado por el PM el 2026-10-20 con OK de Fede (VM); depende de I01 | `celdas.csv` de una cartera de 2 piernas × 2 activos tiene 4 filas sin Sharpe ni DSR; una pierna que aporta más del 70% da `una_pierna_domina` | leer | L17 |
| I10 | `QuantAgent-1dq9` | Drawdown de la cartera en la ficha de postmortem: `celdas.csv` trae `max_drawdown_cartera` y `ficha_postmortem` la usa en vez de la celda suelta (las 3 carteras de L16 salían marcadas por celdas de un solo ETF); agregado por el PM el 2026-10-20 (VM); depende de I09 | Celdas con drawdown 0,5 y cartera 0,2: sin `drawdown`; cartera 0,4: con `drawdown`; sin la columna, igual que antes | leer | L17 |
| I08 | `QuantAgent-j7bl` | Punto de control L17 (VM, lo corre el PM): fichas de las tres carteras de L16, hasta 3 propuestas con resultado de I02 e I03, retrospectiva del lote L16; depende de I05, I06, I07 , I09 e I10 | Fichas, propuestas con chequeos y retrospectiva; ninguna hipótesis en `hipotesis/` sin OK de Fede | decidir | L17 |
| B-F1 | `QuantAgent-pdxx` | Cliente FRED/ALFRED con todas las versiones (nube) | 2 páginas; la clave nunca aparece | leer | LB1 |
| B-F2 | `QuantAgent-vj37` | Snapshot macro congelado con manifiesto (nube); depende de B-F1 | `verify` detecta un parquet alterado | leer | LB1 |
| B-F3 | `QuantAgent-okp9` | Lectura punto en el tiempo (nube); depende de B-F2 | Revisión de mayo no se ve en abril | leer | LB1 |
| B-F4 | `QuantAgent-2oaf` | Primer snapshot `macro-2026-10` (VM o notebook; lo baja el PM con la clave de Fede) | `verify` pasa; CPI con varias versiones | leer | LB1 |
| B-T1 | `QuantAgent-qix2` | `CandleSource` y `YahooSource` en UTC (nube) | Hora de Nueva York → UTC | leer | LB1 |
| B-T2 | `QuantAgent-br4h` | `DataProvider` delega en la fuente (nube); depende de B-T1 | Pide solo el hueco | leer | LB1 |
| B-T9 | `QuantAgent-jpms` | `AlpacaSource` solo datos (nube) | 2 páginas en UTC; solo host de datos | leer | LB1 |
| B-T8 | `QuantAgent-v38v` | Medición de Alpaca con claves reales (notebook o VM); depende de B-T9 | Cifras o "sin claves" | leer | LB1 |
| Q01a | `QuantAgent-ro3w.1` | Cálculo de criterio paper desde curvas guardadas y tests sin escrituras; padre ro3w | Sintéticas PASA/CORTA con motivo; sin aperturas de candado ni escrituras de registros | leer | L18 |
| Q01b | `QuantAgent-ro3w.2` | CLI con tabla y DSR crudo/efectivo; depende de Q01a | CLI sintética muestra métricas, recargo y motivos; curva faltante se informa | probar | L18 |
| Q01c | `QuantAgent-ro3w.3` | Corrida autorizada núcleo + curva e IM; depende de Q01b (VM, Codex) | Tabla y veredicto reales o curvas faltantes; registros intactos; oráculos verdes | decidir | L18 |
| Q02 | `QuantAgent-rpna` | Señal `turn-of-the-month` registrada como `halloween` (nube) | Abre el último día hábil, cierra el 3.º del mes siguiente, feriados sin corrimiento; no mira el futuro | leer | L18 |
| Q03 | `QuantAgent-rdcf` | `cambio-de-mes.yaml` con ventana en listas, de vuelta en `hipotesis/` (nube); depende de Q02 | Carga y la admisión la acepta con 0 parámetros libres; `proponer` y `tanda` cargan la carpeta | leer | L18 |
| Q03b | `QuantAgent-lgg6` | `backtest hipotesis correr --hasta-etapa N` para frenar antes de abrir la prueba 2019–2022 (nube); hallazgo del PM: hoy `correr` abre solo el candado al pasar la etapa 2 | Con `--hasta-etapa 2` no se abre el candado ni se escribe etapa 3; sin la opción, igual que hoy | leer | L18 |
| Q04 | `QuantAgent-dtbv` | `cambio-de-mes` por las etapas 0 a 2, 2007–2018, con `--hasta-etapa 2`, intentos contados (VM); depende de Q03 y Q03b | IM con veredicto; ninguna entrada nueva en el candado; pide a Fede el OK de la etapa 3 si pasa | decidir | L18 |
| Q05 | `QuantAgent-vr3z` | T3: `QUANTAGENT_DATA_SOURCE` y `build_source` (nube) | Sin variable da `YahooSource`; valor inválido falla listando los válidos; el backtest sigue en `offline` | leer | L18 |
| Q06 | `QuantAgent-gjmf` | T6: solo velas cerradas y UTC en `DataProvider`, un `now()` inyectable (nube) | La vela abierta no se guarda; hora de Nueva York queda en UTC; tres referencias de M1 sin cambio | leer | L18 |
| Q07 | `QuantAgent-1y0i` | T4: columna `source` y clave única en `market_data`, migración (nube; Postgres) | `upgrade` y `downgrade -1` sin error con duplicados; misma clave con otro `source` entra | probar | L18 |
| Q08a | `QuantAgent-lrf8.1` | T5a: DataProvider lee y guarda por fuente; padre lrf8, depende de Q07 | Fuentes aisladas, offline sin fuente conserva legacy; tests y smoke verdes | leer | L18 |
| Q08b | `QuantAgent-lrf8.2` | T5b: fuente explícita en las dos consultas del backtest; depende de Q08a | Snapshot aislado de alpaca en calendario y precios; default legacy; oráculos verdes | leer | L18 |
| Q08c1 | `QuantAgent-lrf8.3.1` | T5c1: `load_fixture` escribe `fixture` y `load_snapshot` `snapshot:<nombre>`; CLI `correr_par` y `_run_once` pasan `source_name`; adaptar `scripts/medir_dependencia_ventana.py` y los tests que precargan `MarketData`; depende de Q08b | Loaders rotulan; `run` y `verify` siguen dando trades; referencias M1 y golden sin cambio; oráculos y smoke verdes | leer | L18 |
| Q08c2 | `QuantAgent-lrf8.3.2` | T5c2: pruebas reales de convivencia con filas `alpaca` en `run` y `verify` (fallan si leen la fuente equivocada) e IM con la salida real; depende de Q08c1 | M1 340/4571,27, 6/-249,35, 6/174,34 y rsi/SPY 187/-9141,21 intactas; cualquier cambio numérico se escala | decidir | L18 |
| Q09 | `QuantAgent-01kb` | T7: detectar ajuste retroactivo de precios y renovar la ventana (nube); depende de Q07 y del cierre de lrf8 (Q08a–Q08c), porque borra por `(symbol, timeframe, source)` | Split 2:1 simulado deja la historia ajustada; sin diferencia no borra nada | leer | L18 |
| Q10 | `QuantAgent-say` | Diagnóstico: la salida en paper manda la señal contraria; test sobre el camino real, sin tocar `quantagent/` | El test responde si queda cartera plana o posición opuesta, con evidencia | decidir | L18 |
| Q11 | `QuantAgent-oc0h` | Tests de reversión al 100 %, 99,85 % y 5 % con slippage y comisión; sin corregir | Efectivo nunca menor a -1e-6; el PR responde si al 100 % queda plana | decidir | L18 |
| Q13a | `QuantAgent-ev7d.1` | Cierre de cartera por trade o símbolo en OrderManager; padre ev7d | `close_trade` conserva comportamiento y `close_symbol` cierra la posición completa con validación y registro | leer | L18 |
| Q13b | `QuantAgent-ev7d.2` | Scheduler cierra la posición al salir; depende de Q13a | Diagnóstico sin xfail; salidas larga, corta y sin trade_id quedan planas; referencias M1 y oráculos sin cambio | probar | L18 |
| Q12 | `QuantAgent-beo` | Sale de la cola el 2026-10-21: su entrega (tabla de diferencias y propuesta de unificación) ya está en main desde #83 (`docs/03_design/QuantAgent-beo-DS-regla-de-salida-duplicada.md`); queda solo la decisión de Fede | — | decidir | L18 |
| X01 | `QuantAgent-brwj` | Paper: guard 'vela posterior a la entrada' en la salida intravela (beo/dfa4); depende de dfa4 | Entrada y ciclo en la misma vela con mínimo previo a la compra no cierra; golden y crossval sin cambio | probar | L21 |
| X02 | `QuantAgent-0gh7.1` | Stop resuelto antes de dimensionar: ±2 % por defecto en un lugar compartido, marcado 'stop por defecto'; depende de beo | Placeholder 0.01 da stop a ±2 %; stop del lado equivocado da error; backtests sin cambio | leer | L21 |
| X03 | `QuantAgent-0gh7.2` | Regla 'riesgo' en `PositionSizer` y `TRADING_RIESGO_POR_TRADE_PCT` (0,25 %); tope 10 %, piso 0,5 % de distancia; depende de X02 | Capital 100.000, riesgo 0,25 %, entrada 100, stop 98: 125 acciones recortadas a 100; reglas actuales sin cambio | leer | L21 |
| X04 | `QuantAgent-0gh7.3` | El stop llega a `execute_decision` desde los cinco llamadores, el replay y la pata de apertura de la reversión; depende de X03 | Cada llamador dimensiona con el stop de su señal; replay sin stop usa el default y lo informa | leer | L21 |
| X05 | `QuantAgent-0gh7.4` | `ResolvedConfig`, `--riesgo-por-trade` en backtest, sweep y walkforward, informe con 'stop por defecto'; depende de X04 | `--regla-tamano riesgo` corre en los tres comandos; sin el flag, igual que hoy | leer | L21 |
| X06 | `QuantAgent-0gh7.5` | `bt_adapter.py` con la regla de riesgo para la validación cruzada; prioridad baja; depende de X04 | Crossval con regla riesgo coincide con el motor en una corrida sintética | leer | L20 |
| X07 | `QuantAgent-e4jc.1` | F1-a: verificar en el texto del paper la definición de la 'cuarta semana' | Definición escrita con cita de sección o página; propuesta F1 consistente | leer | L20 |
| X08a | `QuantAgent-e4jc.2.1` | F1-b1: pesos objetivo de F1 por fecha sin lookahead; padre e4jc.2; ~40 líneas | Señal SPY/IEF sintética confirma reglas 60/40 y objetivo 50/50 fuera de los últimos cinco retornos, sin datos futuros | leer | L20 |
| X08b | `QuantAgent-e4jc.2.2` | F1-b2: estrategia `rebalanceo-fin-de-mes` registrada; depende de X08a; ~35 líneas | Alias instancia la estrategia; posiciones esperadas y curva diaria 50/50 fuera de ventana coinciden | leer | L20 |
| X08c | `QuantAgent-e4jc.2.3` | F1-b3: modo opcional de llenado al cierre; depende de X08b; ~45 líneas | Orden de t llena al close de t; default en open de t+1, backtests, goldens y oráculos sin cambio | probar | L20 |
| X09 | `QuantAgent-e4jc.3` | F1-c (R10): F1 por las etapas 0 a 2 con `--hasta-etapa 2` (VM); depende de X08c y S01 | IM con veredicto; ninguna entrada nueva en el candado; pide a Fede el OK para la reserva | decidir | L20 |
| X09b | `QuantAgent-jpwz` | F1: diagnóstico del corte de etapa 1 por menos de 30 trades (hallazgo de #526); sin repetir la corrida ni sumar a N | Operaciones esperadas vs contadas con la causa; test sintético que reproduce el conteo | probar | L20 |
| X09c | `QuantAgent-g7kg` | F1: `filtrar_activo` cuenta los rebalanceos de estrategias de pesos objetivo (hallazgo de jpwz, #528); sin repetir la corrida de F1 | El test de jpwz afirma 288 cambios / 576 órdenes; estrategias de señal con los mismos trades; oráculos sin cambio | leer | L20 |
| X09c1 | `QuantAgent-g7kg.1` | Filtro de pesos objetivo con el panel completo; partición de g7kg | Cambios de peso contados por activo; métricas de estrategias de señal sin cambios | leer | L20 |
| X09c2 | `QuantAgent-g7kg.2` | Integración del filtro objetivo y carga de cierres sincronizados en etapa 1; depende de g7kg.1 | La etapa 1 corre la estrategia de pesos objetivo y suma órdenes por activo | leer | L20 |
| X09c3 | `QuantAgent-g7kg.3` | Prueba sintética de conteo F1; depende de g7kg.1 y g7kg.2 | 288 fechas y 576 órdenes afirmadas; pruebas de señal y oráculos sin cambio | probar | L20 |
| X09d | `QuantAgent-tm5d` | F1: la etapa 2 (control de celdas) cuenta los rebalanceos de estrategias de pesos objetivo o declara 'no aplica' (hallazgo de #534); sin repetir la corrida de F1 | Test sintético: etapa 2 cuenta 288/576 órdenes; estrategias de señal sin cambio; oráculos sin cambio | leer | L20 |
| X09e | `QuantAgent-tso5` | F1: el motor de la etapa 2 ejecuta pesos objetivo (hallazgo de #538: 0 trades aun con #537); sin repetir F1 con datos reales | Test sin mockear `correr_motor`: etapa 2 con trades > 0 por activo; estrategias de señal y oráculos sin cambio | leer | L20 |
| X10a | `QuantAgent-e4jc.4a` | F2-a1: PnL de celda corta sintética en la cartera; padre e4jc.4 | Celda corta gana con baja y pierde con suba, con valores comprobados | leer | L20 |
| X10b | `QuantAgent-e4jc.4b` | F2-a2: costos simétricos y número long-only intacto; depende de X10a; padre e4jc.4 | Costo/PnL neto simétricos en largos y cortos de igual nocional/variación; referencia long-only sin cambio | leer | L20 |
| X10c | `QuantAgent-e4jc.4c` | F2-a3: control de celda corta contra `backtesting.py`; depende de X10a y X10b; padre e4jc.4 | Motor y control coinciden con costos en corto sintético y referencia long-only | probar | L20 |
| X11a | `QuantAgent-zpsc` | F2-b1 desglosado: hijos zpsc.1 (OrderManager/RiskManager) y zpsc.2 (scheduler, depende de zpsc.1); depende de X10c | Hijos: corto abre/cierra con RiskManager y cartera plana; scheduler integra el ciclo PaperBroker; long-only intacto | decidir | L20 |
| X11a1 | `QuantAgent-zpsc.1` | F2-b1a: apertura/cierre de cortos en OrderManager y validación RiskManager; hijo de zpsc | SELL abre corto, BUY deja cartera plana; tamaño/concentración validados y long-only intacto | leer | L20 |
| X11a2 | `QuantAgent-zpsc.2` | F2-b1b: ciclo de corto monitoreado por scheduler con PaperBroker; depende de zpsc.1; hijo de zpsc | Camino sintético scheduler abre, monitorea y cierra a cartera plana; long-only intacto | probar | L20 |
| X11b | `QuantAgent-lj6l` | F2-b2: cortos en AlpacaBroker y bloqueo en endpoint live qr6 (parte de e4jc.5); bloqueado hasta que #412 llegue a main y edge se rebase | Corto abre y cierra en cuenta paper de Alpaca; en endpoint live queda bloqueado | decidir | L20 |
| X12a1 | `QuantAgent-t2qp.1` | F2-c0a: estrategia `tsmom-largo-corto` y señal mensual frente a DGS3MO; partición de t2qp | Prueba sintética de señal SHORT/LONG y comparación de retorno a 252 sesiones | leer | L20 |
| X12a2 | `QuantAgent-t2qp.2` | F2-c0b: integrar señal corta al filtro y cartera; depende de t2qp.1 | Filtro y cartera respetan signo de señal y 1/sigma; long-only intacto | leer | L20 |
| X12a3 | `QuantAgent-t2qp.3` | F2-c0c: habilitar YAML oficial en CLI hasta etapa 1; depende de t2qp.2 | `hipotesis correr` hasta etapa 1 no falla por estrategia desconocida; YAML no se mueve a `hipotesis/` | probar | L20 |
| X12 | `QuantAgent-e4jc.6` | F2-c (R12): F2 por las etapas 0 a 2 (VM); depende de X10c, S01 y X14c | IM con veredicto y correlación con 1/N; ninguna entrada nueva en el candado | decidir | L20 |
| X12b | `QuantAgent-46pk` | F2: diagnóstico de errores del adaptador al validar órdenes cortas en la etapa 2 (hallazgo del IM de e4jc.6, #521); VM con datos reales, sin abrir 2019–2022 | Lista de celdas con error y mensaje exacto; test que lo reproduce; veredicto de F2 con y sin esas celdas | probar | L20 |
| X13 | `QuantAgent-e4jc.7` | D-1: factores AQR y French cortados en 2018-12; AQR no se versiona | Ninguna fecha posterior a 2018-12; archivos de AQR ignorados por git | leer | L20 |
| X14a | `QuantAgent-e4jc.8.1` | D-2a: regla con corte en 2007 antes de seleccionar ETFs; sin PTP | Regla integrada en commit propio, condiciones verificables y ninguna lista de candidatos | leer | L20 |
| X14b | `QuantAgent-e4jc.8.2` | D-2b: documentar empalme con proxies gratuitos; depende de X14a | Fuentes, unidades, fórmula, fecha del enlace, validación y límites por clase | leer | L20 |
| X14c | `QuantAgent-e4jc.8.3` | D-2c: aplicar regla integrada y congelar snapshot nuevo; depende de X14b | Evidencia de elegibilidad posterior al commit de la regla; manifiesto versionado; `snapshot verify` pasa | probar | L20 |
| X15 | `QuantAgent-e4jc.9` | D-3: CSV de pysystemtrade (futuros, hasta 2024-03) como datos de investigación | Manifiesto con fuente y hash; carga de una serie de prueba | leer | L20 |
| X16 | `QuantAgent-e4jc.10` | G-1: lección escrita por personas en el catálogo e incluida en el pedido, sin cifras de la prueba; depende de S05 y S08 | El pedido contiene la lección; el test confirma que no hay Sharpe, DSR ni ganancias | leer | L20 |
| X17 | `QuantAgent-e4jc.11` | G-2: refutador de otra familia (Codex), modelo más capaz con esfuerzo alto y búsqueda web; depende de S06 y S08. No correr `hipotesis proponer` antes | Comandos de mentira escriben propuesta y refutación; si el refutador falla no se escribe nada | leer | L20 |
| X18 | `QuantAgent-e4jc.12` | G-3 (opcional): cupo por tema y marca 'sin prueba ciega en 2019–2022' (§5.3 puntos 6 y 7); depende de S06 | Segunda propuesta del mismo tema en un lote se rechaza; muestra que termina en 2019 o después marca | leer | L20 |
| X19a | `QuantAgent-e4jc.13.1` | P-1a: pesos paper contra la sombra | Pesos iguales dentro de tolerancia; diferencia indica activo | probar | L21 |
| X19b | `QuantAgent-e4jc.13.2` | P-1b: rechazos y dividendos | Sin rechazo inexplicado; cada dividendo tiene tratamiento | probar | L21 |
| X19c | `QuantAgent-e4jc.13.3` | P-1c: CLI de chequeos; parte de e4jc.13 | CLI muestra las tres secciones y falla ante incumplimiento | probar | L21 |
| X20 | `QuantAgent-e4jc.14` | P-2 (R13): protocolo de la reserva 2023+ (lista cerrada cartera-transversal, F1, F2; C0 + M1; una apertura); depende de S04 | Documento con lista, criterio y regla de una apertura; sin escalera de incubación | decidir | L20 |
| X20b.1 | `QuantAgent-3bby.1` | Elegibilidad y simulacro del comando de reserva; parte de X20b | Snapshot sintético: rango calculado sin abrir/leer reserva; rechazos de lista, etapa 2 y candado | probar | L20 |
| X20b.2 | `QuantAgent-3bby.2` | Pre-registro y apertura única bajo `--abrir`; depende de X20b.1 | Pre-registro precede apertura, accesos registrados y segundo intento rechazado en fixture sintético | probar | L20 |
| X20b.3a | `QuantAgent-3bby.3.1` | Motor/CLI existente sobre reserva abierta; curva, C0 intacto y benchmark 1/N; parte de X20b.3 | Snapshot sintético produce curva y métricas C0 más benchmark 1/N coherentes, sin leer datos reales | probar | L20 |
| X20b.3b | `QuantAgent-3bby.3.2` | M1 con DGS3MO, veredicto C0 intacto e intento en CLI; depende de X20b.3a | Fixture sintético produce Sharpe en exceso, veredicto y fila de intento coherentes, sin leer datos reales | probar | L20 |
| X20b.4 | `QuantAgent-3bby.4` | Camino de pesos objetivo con panel completo; depende de X20b.3 | Estrategia sintética de pesos objetivo cuenta trades > 0 y reproduce pesos/curva | probar | L20 |
| X19f | `QuantAgent-bduo` | Paper: el chequeo diario tolera hasta 1 acción de diferencia entre Alpaca y la sombra (peso de 1 acción = precio / valor invertido); bloquea el paper del 2026-11-02 | Peso a menos de 1 acción: sin hallazgo; a más de 1: hallazgo y código 1; sin precio: 1e-4; sin `tolerancias` igual que hoy | probar | L20 |
| X19g | `QuantAgent-8kxc` | Paper: dividendos de Alpaca activos por defecto en el digest y validados; refina X19e (`QuantAgent-xtns`); bloquea el paper del 2026-11-02 | Cliente falso de clase real con DIV: lista y código 0; error de API o actividad malformada: aviso y código 1; `--sin-dividendos` vuelve al aviso | probar | L20 |
| X20c1 | `QuantAgent-ku33.1` | C0 cartera, DSR y benchmark 1/N desde curvas `{activo}.csv`; parte de ku33 | Test sintético comprueba C0, DSR y benchmark con lectura real de archivos; sin reserva real | probar | L20 |
| X20c2 | `QuantAgent-ku33.2` | Recuperar veredicto de reserva y M1 idempotente; depende de ku33.1 | M1 coincide con valor esperado; dos ejecuciones dejan un intento/estado y no llaman `abrir()` ni `correr_motor` | probar | L20 |
| X19d.1 | `QuantAgent-e4jc.15.1` | P-1d1: consultar posiciones, órdenes y dividendos paper para digest; parte de e4jc.15 | DB SQLite sintética devuelve datos de sesión aislados a PAPER; posiciones cantidad × precio | probar | L20 |
| X19d.2 | `QuantAgent-e4jc.15.2` | P-1d2: integrar pesos DB-sombra y aviso de sesión faltante; parte de e4jc.15 | Coincidencia da 0, divergencia da 1 y sombra sin última sesión muestra aviso; depende de e4jc.15.1 | probar | L20 |
| X19d.3 | `QuantAgent-e4jc.15.3` | P-1d3: ejecutar chequeos de órdenes y dividendos desde digest DB; parte de e4jc.15 | Rechazo sin comentario y dividendo sin tratamiento dan 1 con hallazgo; atendidos dan 0; depende de e4jc.15.1 | probar | L20 |
| X19e | `QuantAgent-xtns` | P-1e: dividendos paper desde actividades DIV de Alpaca en el digest (`--dividendos-alpaca`); decisión de Fede 2026-10-27 | DIV listado con tratamiento y código 0; error de API: aviso y código 1; sin opción sin cambios | probar | L20 |


| W02 | `QuantAgent-13a` | Contrato Broker para fills parciales, rechazos y nocional | ABC y PaperBroker cubren el contrato documentado | leer | L21 |
| W03 | `QuantAgent-8659` | Medir rechazo por exceso del poder de compra y fijar regla | Medición paper y regla única en trading | decidir | L21 |
| W03a | `QuantAgent-8659.1` | Reproducir rechazo por gap entre señal y fill next-open | Caso determinista y test del límite de efectivo | probar | L21 |
| W03b | `QuantAgent-8659.2` | Reserva de efectivo para compras | Entradas/reversiones reservan efectivo y ventas siguen operativas | leer | L21 |
| W04 | `QuantAgent-qr6` | Selector de broker y bloqueo del endpoint live | Live bloqueado sin autorización explícita | leer | L21 |
| W05a | `QuantAgent-yme.1` | Envío/cancelación Alpaca paper | Market/limit, fills, rechazo y cancelación cubiertos | probar | L21 |
| W05b | `QuantAgent-yme.2` | Consulta cuenta/posiciones y mapeo de estados | Estados, datos y errores API cubiertos | leer | L21 |
| W05c | `QuantAgent-yme.3` | Registrar AlpacaBroker en StrategyAssembler | Selector paper; live rechazado | probar | L21 |
| W05d | `QuantAgent-yme.4` | Prueba reproducible contra Alpaca paper | Orden limit remota cancelada; evidencia en entrega | probar | L21 |
| W06a | `QuantAgent-y62.1` | Auditar ActivePosition contra Alpaca | Divergencias visibles y persistidas como WARNING; no corrige estado | probar | L21 |
| W06b | `QuantAgent-y62.2` | CLI de reconciliación DB/Alpaca | Reporta diferencias sin mutar posiciones | probar | L21 |
| W06c | `QuantAgent-y62.3` | Reconciliación periódica para Alpaca paper | Job solo paper; no muta ActivePosition | probar | L21 |
| W07 | `QuantAgent-av1p.1` | Pesos objetivo de las carteras por fecha | Coinciden con curva de backtest; cerrado | leer | L21 |
| W08 | `QuantAgent-av1p.2` | Rebalanceador de pesos y posiciones a órdenes | Caso sintético; cerrado, aún sin integración al scheduler | probar | L21 |
| W09 | `QuantAgent-av1p.3` | Sombra por fecha para comparar con paper | Se guarda/lee sombra; cerrado, aún sin integración al scheduler | leer | L21 |
| W10a | `QuantAgent-av1p.4.1` | Persistir órdenes pendientes/rechazadas sin posiciones ficticias | Sin fill no cambia cartera; parcial aplica solo cantidad real | probar | L21 |
| W10b | `QuantAgent-av1p.4.2` | Persistir identidad remota de órdenes; depende W10a | ID Alpaca sobrevive reinicio; migración reversible | probar | L21 |
| W10b-T1 | `QuantAgent-7v45.1` | Transportar selección hasta curva/evaluación; depende W10b | Fixture T4 da 110000; mezcla conserva 1/S | probar | L21 |
| W10b-T2 | `QuantAgent-7v45.2` | Conservar benchmark en efectivo con pesos cero; depende T1 | Suma cero conserva equity; fórmula positiva y 1/N intactos | probar | L21 |
| W10b-T3 | `QuantAgent-7v45.3` | Selección ganadora por ventana hasta agregado etapa 1; depende T2 | Panel ganador correcto llega a curva T4 sin recalcular ranking | probar | L21 |
| W10b-T4 | `QuantAgent-7v45.4` | Lector CSV compartido, validación 2/3 y mismo_peso; depende T3 | Curva y pesos benchmark usan selección; falta de CSV no usa pesos antiguos | probar | L21 |
| W10b-T5 | `QuantAgent-7v45.5` | Informe futuro con selección y mismos pesos; depende T4 | Curva/pesos T4 o diagnóstico no disponible, sin reescribir evidencia | probar | L21 |
| W10b-T6 | `QuantAgent-7v45.6` | Criterio desde curvas de consultas nuevas; depende T5 | Ambas etapas usan selección; 1/N y tasas intactos | probar | L21 |
| W10b-T7 | `QuantAgent-7v45.7` | Objetivo→sombra nueva y adenda sintética; depende T6 | Sombra conserva objetivos T4; ninguna medición histórica nueva | probar | L21 |
| W10c | `QuantAgent-av1p.4.3` | Consultar estado Alpaca sin reenviar; depende W10b | Pending→partial→filled no reenvía ni duplica Fill | probar | L21 |
| W10d1 | `QuantAgent-av1p.4.4.1` | Aplicar fills incrementales idempotentes; depende W10c | 0→3→10→10 aplica solo 3 y 7 al precio incremental correcto, incluso al reabrir sesión | probar | L21 |
| W10d2 | `QuantAgent-av1p.4.4.2` | Consultar pendientes y evitar reenvíos; depende W10d1 | Pending/partial no duplica submit; cancelación/rechazo quedan persistidos con solo fills reales | probar | L21 |
| W10e | `QuantAgent-av1p.4.5` | Reserva de efectivo compatible con Alpaca; depende del cierre de av1p.4.4 (W10d1/2) | Compra entera respeta reserva sin costos remotos inventados | probar | L21 |
| W10f1 | `QuantAgent-av1p.4.6.1` | Calendario de sesión NYSE cerrada; depende W10e | Fines de semana/feriados conservan sesión y cierre temprano usa hora real | probar | L21 |
| W10f2 | `QuantAgent-av1p.4.6.2` | Panel y precios por sesión cerrada; depende W10f1 | ≥253 cierres completos, held con precio vigente, sin barras incompletas/futuras ni datos rellenados | probar | L21 |
| W10g1 | `QuantAgent-sy3n` | Exposición de momentum, SMA y ranking transversal con señales existentes; depende de .6 cerrado (#479, #480) | Los tres reproducen estado largo/efectivo del backtest en fixture sintético, sin mirar futuro | probar | L21 |
| W10g2 | `QuantAgent-2j54` | Pierna curva-pendiente (macro T10Y2Y); bloqueada por falta de fuente macro vigente para paper (snapshot congelado no es feed futuro); solo la usa cartera-nucleo-curva | Curva reproduce la señal del backtest; con dato macro vencido no opera y reporta bloqueo | probar | L21 |
| W10h | `QuantAgent-av1p.4.8` | Componer pesos y exposición; depende W10g y cierre QuantAgent-7v45, corrección T4 ya autorizada por Fede ([partición](docs/loop/particiones/QuantAgent-7v45.md)) | Referencia T4 prospectiva, efectivo/calendario preservados y datos anteriores intactos | probar | L21 |
| W10i | `QuantAgent-av1p.4.9` | Ciclo paper del scheduler; depende W10h | Pesos→órdenes→posiciones/sombra; live bloqueado, estrategias sueltas sin cambio | probar | L21 |
| W10j | `QuantAgent-av1p.4.10` | Registro/armado explícito de ambas; depende W10i | Una cartera/cuenta/DB/proceso, sin operación real (decisión Fede 2026-10-24) | probar | L21 |
| W10j2 | `QuantAgent-8hih` | Macro FRED explícito en `--portfolio` (curva-pendiente); hueco que dejó #490; depende W10j | Cliente FRED llega a `paper_portfolio`; sin clave y con curva el arranque falla claro | probar | L21 |
| W10k | `QuantAgent-av1p.4.11` | Smoke paper de ambas carteras; depende W10j2 | Dos ciclos independientes con fills reales/sombra y reinicio idempotente | probar | L21 |
| W15a | `QuantAgent-b41.1` | Armar digest operativo desde datos paper; depende W06 y W10 | Función pura y tests cubren posiciones, actividad/P&L, warnings, heartbeat, reconciliación, truncamiento y sin actividad | probar | L21 |
| W15b.1 | `QuantAgent-b41.2.1` | Consultar posiciones, trades y heartbeats paper para el digest | Estructuras de `build_digest` aisladas por fecha y ambiente | leer | L21 |
| W15b.2 | `QuantAgent-b41.2.4` | Limitar warnings extensos y reforzar límite de 200 trades | Digest menor a 4000 con contador de warnings omitidos; test afirma longitud | probar | L21 |
| W15b.3 | `QuantAgent-b41.2.2` | Agregar CLI digest con selección de DB/cartera y prueba SQLite | CLI pasa fecha y datos; salida no vacía y menor a 4000 caracteres | probar | L21 |
| W15b.4 | `QuantAgent-b41.2.3` | Documentar comando y ejemplo reproducible | Manual cubre opciones, salida y alerta sin actividad | leer | L21 |
| W16 | `QuantAgent-55x` | Reloj por velas en corridas online; opcional si W10 lo necesita | No reevalúa última vela en sábado/feriado | leer | L21 |

X01–X20 se agregaron el 2026-10-22; X01–X05 y X19a–c pertenecen a L21, X06–X18 y X20 a L20. D-2 se partió en X14a–c, según las decisiones de Fede registradas en `docs/04_decisions/QuantAgent-R01-DC-decisiones-investigacion-edge.md` (PR #400). Paper es prueba de sistema: ningún resultado de paper promueve una estrategia a dinero real ni cuenta como evidencia de edge. Lo que ya cubría L19 (S01–S08: M1–M3, atribución, catálogo, DOI, advertencias y pedido) no se duplica; el rol `diversificador` (C1) queda fuera del pedido y de los campos, y `nb94.1`, `syml` y `55nh` llevan un comentario para que el PM lo revise. Sin ticket por ahora, y registrado en el DC: `cambio-de-mes` y `volatilidad-gestionada` fuera, F3 `carry-futuros` en espera, subastas del Tesoro rechazadas, DBC/USO y retención del 30 % sobre dividendos (se resuelven antes de dinero real; paper sin cambios). Dato para estrategias futuras, no restricción: capital live estimado de USD 1.000–3.000.

V08b y V23d se agregaron el 2026-10-12 al revisar V23c (#127): la corrida del agente leyó el tramo de prueba 2019–2022 porque el CLI solo protege la reserva, y con llenado en la apertura siguiente el adaptador cancela órdenes al 100% del capital (21 de 48 celdas). Van antes de V24 para que el markdown salga con la sensibilidad corregida.

**Decisión de Fede (2026-10-10):** Triple Screen sale de la comparación diaria de M2 y se evalúa en el lote de 4h (`docs/06_implementation/QuantAgent-4x6-IM-dependencia-ventana.md`). La comparación diaria usa cuatro estrategias: `rsi`, `fifty-two-week-high`, `sma-cross` y `momentum-12m`.

Con el modo por lotes, el ritmo lo marcan las decisiones de Fede (T13, T10, T22, T23) y no la implementación.
M1 se cumple al cerrar T23d. **M1 cumplido el 2026-10-09** (lote L3, PR #52 mergeado a `main`).

**M2, validación con datos reales.** Plan aprobado por Fede el 2026-10-09 (PR #62): `docs/02_planning/QuantAgent-lp1-PL-m2-validacion-datos-reales.md`. Son 7 lotes; acá están las filas de los dos primeros
(L4 y L5 = lotes 1 y 2 del plan), ya cargadas en BEADS. Las de los lotes 3 a 7 están en §11 de ese doc y el PM las copia
y las carga al abrir cada lote. El texto completo de cada ticket está en BEADS; los `Vnn` son el número de fila.
V00a–V00e son las mejoras de performance de la suite que Fede aprobó en la `R:` del PR #64 (objetivo: 180 s); V00b cambia el
comando de la suite del loop y saca los dos oráculos de cada entrega, que pasan a correr en el CI y en la revisión del PM.
V01–V04 las hace una sesión de Claude Code cloud lanzada por Fede el 2026-10-09 (tickets `in_progress`, PRs sin label `loop`).
V00 (`QuantAgent-vfd`) y la fuente de SPY 4h (Alpaca, solo datos) se agregaron por decisión de Fede del 2026-10-09.
Filas marcadas (VM): necesitan `QUANTAGENT_SNAPSHOT_DIR` y el snapshot `etf-1d-2026-10`, que crea el PM al mergear V02.

T23b, T23c y T23d se agregaron el 2026-10-08: T22 mostró que el engine evalúa stop loss y take profit solo al
cierre de la vela (227 trades / 13084.98 contra 340 / 5803.02 evaluando dentro de la vela). Fede decidió en la
revisión de #55 seguir con el engine propio y corregirlo dentro de L3. T23d cambia los números de referencia de
las tres estrategias.

T23e se agregó el 2026-10-09 a pedido de Fede: 15 tests usan SQL solo de Postgres y fallan sobre la base temporal
de `try.sh`, lo que obliga al PM a comparar la suite del lote contra la del PR en cada revisión (unos 30 minutos).

T08b y T08c se agregaron el 2026-10-06: al revisar T08, Fede detectó que el script daba por bueno el `pnl` del CSV.
El diagnóstico mostró 1% de slippage por lado y un CSV que mezcla precio ejecutado y teórico. Cambian los números
de referencia de rsi/spy-90d (227 / −4717.33), que T09 y T20 tienen que tomar de T08c.

T15b y T10b se agregaron el 2026-10-07 por decisión de Fede. T15b salió de la revisión: dos corridas sobre la misma
base dan 227 y 229 trades, y `backtest verify` no lo cubre porque usa una base limpia en cada pasada. T10b es la
`R:` de Fede en el PR #30. Los dos van antes de T20 para que el test golden congele números que ya no dependan
del estado de la base.

**Fuera del loop** (sesión de escritorio, no entran en la cola):
- `QuantAgent-wwi`, skill de revisión macro semanal: instala en `~/.hermes`.
- `QuantAgent-zui` y `QuantAgent-hak`: son parte de la fase 0 (PR de la sesión 1).
- `QuantAgent-hyt`: reevaluar el validador funcional de QA, comentado en el workflow el 2026-09-29.
- `QuantAgent-u0w`, todo M2, la UI de Streamlit.

---

## 5. Formato de log de revisión

### 5.1 La línea

Desde el 2026-10-06 hay dos líneas con el mismo formato y la misma validación. `PM:` la escribe la sesión
`pm-revisor` en cada entrega del loop; admite además `decido: escalo <pregunta>` y solo vale en PRs hacia una
rama de lote. `R:` la escribe Fede en el PR del lote y en lo que el PM le escala.

Un comentario en el PR, escrito desde GitHub mobile:

```text
R: <ticket> vi: <qué miré, con un dato concreto de esta entrega> decido: <merge | cambio <qué> | descarto <por qué>>
```

### 5.2 Ejemplos

Válidas:

```text
R: e35 vi: CI verde, 22 skipped igual que antes decido: merge
R: bv8 vi: en la VM salieron 5 lineas, sharpe 0.40 decido: merge
R: 89e.1 vi: filas 14 y 15 mismo pnl, son un solo round trip decido: merge
R: hx0.7 vi: tabla con sharpe 0.40 vs 0.83 decido: cambio usar 24x365
R: 46h.1 vi: rupturas solo en el primer año decido: descarto quiero 2 por año
```

Rechazadas por el gate:

```text
R: e35 ok
R: e35 vi: todo bien decido: merge
revisado
```

### 5.3 Validación automática (en `scripts/loop/gate.py`)

1. Autor `fscheu`, posterior al último commit del PR.
2. Cumple la expresión regular:
   `^R:\s*(\S+)\s+vi:\s*(.+?)\s+decido:\s*(merge|cambio|descarto)\b\s*(.*)$`
3. `<ticket>` coincide con el ticket del PR (con o sin el prefijo `QuantAgent-`).
4. El texto de `vi:` tiene al menos 3 palabras y al menos una palabra o número que aparece en el
   cuerpo o en el diff del PR. Palabras como "ok", "bien", "todo", "listo" no cuentan.
5. `cambio` y `descarto` requieren texto después de la palabra.

### 5.4 Dónde queda

- El comentario del PR es la señal que lee el gate.
- Cuando una revisión queda cerrada (merge al lote, descarte, o la `R:` de Fede sobre una escalada), el PM agrega
  la línea a `docs/loop/REVIEW-LOG.md` con un commit directo en la rama del lote, con fecha y número de PR:

```text
2026-10-01 · #57 · R: bv8 vi: en la VM salieron 5 lineas, sharpe 0.40 decido: merge
```

- `docs/loop/REVIEW-LOG.md` es el historial: una línea por entrega revisada, con su prefijo `PM:` o `R:`.
- Hasta el 2026-10-06 la copiaba el agente en la entrega siguiente. Con un PR escalado abierto y otro en curso,
  los dos agregaban una línea en el mismo lugar y el segundo merge chocaba (#24 y #25). Ahora escribe uno solo.

## 6. Reconciliación y temas fuera de lote

El inventario de Beads sin cerrar, los tickets integrados y el destino de cada pendiente se mantiene en [QuantAgent-pjwb-PL-reconciliacion-plan-beads.md](docs/02_planning/QuantAgent-pjwb-PL-reconciliacion-plan-beads.md). Los tickets ejecutables de L20 y L21 están en la cola de esta sección; una etiqueta histórica no incorpora automáticamente un ticket a un lote.

## 7. Ideas registradas para M3 y siguientes

| Idea | Ticket | Documento | Cuándo |
|---|---|---|---|
| Elegir el tipo de estrategia técnica según el régimen de mercado (y más adelante la macro), en lugar de probar todas en todos los activos. Primer paso: medir en 2007–2018 si el resultado de cada estrategia cambia con la volatilidad y la tendencia de SPY | `QuantAgent-ljoj` (decidir) | [`docs/02_planning/QuantAgent-ljoj-PL-regimenes-y-macro.md`](docs/02_planning/QuantAgent-ljoj-PL-regimenes-y-macro.md): análisis, riesgos y referencias comerciales y académicas (JPMaQS, Hedgeye, Fidelity, Two Sigma, meta-labeling, ALFRED) | M3, después de cerrar L7 |
| Motor top-down después de M4 (M5): lectura macro + momentum elige sectores, screener fundamental elige acciones, arquetipos técnicos con stop en el broker y seguimiento que solo achica el riesgo, en una cuenta paper aparte. Alcance completo decidido por Fede el 2026-10-25; la macro se mide como hipótesis contra el momentum solo | provisorios `M5-01`–`M5-54` (sin Beads) | Propuesta [`docs/02_planning/2026-10-25_propuesta_motor_jerarquico_fede.md`](docs/02_planning/2026-10-25_propuesta_motor_jerarquico_fede.md), diseño [`docs/03_design/QuantAgent-M5-DS-motor-top-down.md`](docs/03_design/QuantAgent-M5-DS-motor-top-down.md), plan [`docs/02_planning/QuantAgent-M5-PL-motor-top-down.md`](docs/02_planning/QuantAgent-M5-PL-motor-top-down.md) (lotes L22–L27, decisiones D1–D7) | después de cerrar L20 y L21; `M5-05` puede adelantarse |
| Que cada hipótesis defina sobre qué activos trabaja, en lugar de evaluar siempre los 12 ETFs. El corte proporcional de E01 ya lo admite; falta la regla de admisión (cuántos activos como mínimo y con qué justificación) | sin ticket | Fede, 2026-10-16, al mergear L9 | después de L10 |
