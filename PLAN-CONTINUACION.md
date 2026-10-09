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

1. `pytest -q -m "not slow and not api"` → 0 failed.
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
4. Agente según `LOOP_AGENT` (`claude` | `agy`; el archivo `next-agent` del directorio de estado lo cambia
   por una corrida), con `loop/PROMPT.md` más el contexto del gate (modo, PR previo, texto de la `R:`) y
   timeout `LOOP_TIMEOUT_MIN`. `claude -p` usa `--permission-mode acceptEdits` y
   `--disallowedTools "Bash(git push origin main*)" "Bash(git push -f*)" "Bash(git push --force*)"`;
   `agy -p` usa `--mode accept-edits` y sus reglas allow/deny (`docs/loop/SETUP-HERMES.md`). Para los dos,
   el hook `scripts/loop/push_guard.sh` rechaza push a main, borrado de ramas y push forzado.
   El PR entregado recibe el label `agent:<agente>` y, con `agy`, `model:<modelo>`.
   El modelo de `agy` es `LOOP_AGY_MODEL`; el archivo `next-model` del directorio de estado lo cambia por una
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
   rama del lote. Después lanza la siguiente corrida. Tope: 10 entregas por día. La tanda la arranca el aviso del
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
3. Correr él mismo el comando de "Probar" con `scripts/loop/try.sh` y la suite. No vale la salida pegada en el PR.
4. Buscar el hueco con al menos una comprobación que el PR no trae: otro fixture u otra estrategia, un caso
   borde, un valor recalculado por otro camino, un test que debería fallar si se rompe la lógica y no falla.
5. Decidir y comentar la línea `PM:`. En `vi:` va el dato de la comprobación propia del paso 4.
6. Al cerrar la revisión, agregar la línea a `docs/loop/REVIEW-LOG.md` en la rama del lote (§5.4).

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

Con el modo por lotes, el ritmo lo marcan las decisiones de Fede (T13, T10, T22, T23) y no la implementación.
M1 se cumple al cerrar T23d.

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
