# PLAN-CONTINUACION

Generado: 2026-09-26. Reemplaza a `docs/02_planning/2026-09-17_plan_mes_2_loop_ai.md` y a
`docs/02_planning/2026-09-17_borrador_routine_nocturna.md`.
Tickets en BEADS con label `plan-continuacion`.

Modo de trabajo: un loop corre de noche y hace **una** entrega. Fede revisa al día siguiente desde el
celular, en dos bloques de ~15 minutos. Sin entorno local, con poco tipeo.

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

`scripts/loop/gate.py` busca el último comentario de `fscheu` posterior al evento de entrega que cumpla
el formato de §5. Resultado:

| Situación | Resultado | El loop |
|---|---|---|
| No existe ningún PR `loop` | PASS `nuevo` | Toma el primer ticket de la cola |
| No hay `R:` posterior a la entrega | WAIT | Imprime `⏸ esperando revisión: <url> (desde <fecha>)` y termina |
| `R:` inválida (§5.3) | WAIT | Imprime `⏸ R inválida: <motivo>` y termina |
| `decido: merge` y PR mergeado | PASS `nuevo` | Cierra el ticket en BEADS y toma el siguiente |
| `decido: merge` y PR abierto | WAIT | Imprime `⏸ R dice merge, falta mergear` |
| `decido: descarto` y PR cerrado sin merge | PASS `nuevo` | Pone label `loop-descartado` al ticket y toma el siguiente |
| `decido: cambio` y PR abierto | PASS `cambio` | Aplica el pedido sobre la misma rama y hace una nueva entrega |
| Cualquier otra combinación | WAIT | Imprime la combinación encontrada |

WAIT no invoca al modelo. No hay excepciones, no hay acumulación: nunca existen dos entregas sin revisar.

Pausa manual: si existe `loop/PAUSE` en `origin/main`, el wrapper imprime `⏸ pausado` y termina.
Se crea o borra desde GitHub mobile.

### 3.2 Regla 2 — Tamaño de entrega

`loop/config.env`:

```bash
LOOP_LABEL=loop
LOOP_REVIEWER=fscheu
LOOP_QUEUE_FILE=PLAN-CONTINUACION.md
LOOP_MAX_LINES=150          # agregadas + borradas, límite duro
LOOP_TARGET_LINES=100
LOOP_MAX_FILES=6
LOOP_EXCLUDE=".beads/** tests/fixtures/** docs/loop/REVIEW-LOG.md"
LOOP_TIMEOUT_MIN=120
```

- Un archivo borrado completo no suma líneas, pero se lista en el resumen.
- `scripts/loop/diff_size.sh` cuenta con `git diff --numstat origin/main...HEAD` aplicando las exclusiones.
- Si el ticket no entra, el agente **no escribe código**: la entrega es una partición (§3.4).
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
Tamaño: <N> líneas / <M> archivos (límite 150 / 6)
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

Registro — comentá:
R: <ID> vi: <dato concreto de esta entrega> decido: merge | cambio <qué> | descarto <por qué>
```

Reglas de redacción: frases cortas, sin jerga interna, sin enlaces obligatorios. Todo lo necesario para
decidir está en el cuerpo del PR.

### 3.4 Regla 4 — Estado consistente

Todo PR `loop` cumple, antes de abrirse:

1. `pytest -q -m "not slow and not api"` → 0 failed.
2. `scripts/loop/smoke.sh` → exit 0 (SQLite temporal, sin tocar la base de desarrollo).
3. CI verde en el PR (lo verifica Fede en el bloque 1; el agente espera el check hasta 15 min).
4. Si cambia un comportamiento documentado, el README o el doc afectado se actualiza en el mismo PR.

`main` solo cambia por merges de Fede. Si el agente no llega a un estado verde, la entrega es de uno de
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

1. `git fetch origin`. Si existe `loop/PAUSE` en `origin/main`: imprimir y salir.
2. `scripts/loop/gate.py` desde `origin/main`. WAIT: imprimir la línea y salir.
3. Crear worktree `/tmp/ai-loop/<RUN>` desde `origin/main`, o desde la rama del PR en modo `cambio`.
4. `claude -p` con `loop/PROMPT.md` más el contexto del gate (modo, PR previo, texto de la `R:`),
   `--permission-mode acceptEdits`, `--disallowedTools "Bash(git push origin main*)" "Bash(git push -f*)" "Bash(git push --force*)"`,
   timeout `LOOP_TIMEOUT_MIN`.
5. `scripts/loop/diff_size.sh` sobre la rama entregada. Si excede, label `excede-limite`.
6. Borrar el worktree.
7. Imprimir para Telegram: las 3 líneas del resumen y la URL del PR.

### 3.6 Prompt del agente (`loop/PROMPT.md`)

```text
Sos el agente nocturno de QuantAgent. Trabajás sin humano disponible. Hacés UNA entrega por corrida.
El revisor lee tu entrega desde un celular, en 15 minutos, sin entorno local.

CONTEXTO DEL GATE (lo agrega el wrapper): modo = nuevo | cambio; PR previo; línea R: de Fede.

REGLAS DURAS
- Nunca pushees a main. Nunca uses --force. Nunca borres ramas ni worktrees ajenos.
- No toques ~/.hermes, ~/secrets, systemd, crons ni nada fuera del worktree.
- Una sola entrega. Si terminás antes, no tomes otro ticket.
- Límite duro: 150 líneas agregadas+borradas y 6 archivos, excluyendo .beads/**, tests/fixtures/**
  y docs/loop/REVIEW-LOG.md. Medí con scripts/loop/diff_size.sh.
- Cada entrega deja el proyecto funcionando: suite en 0 failed y `scripts/loop/smoke.sh` con exit 0.
- Respetá AGENTS.md y CLAUDE.md del repo.

PASO 1 — Registro y BEADS (solo si el modo trae una R: nueva)
- Agregá la línea R: a docs/loop/REVIEW-LOG.md con fecha y URL del PR previo.
- decido merge → bd close <ID> con la R: como motivo. Si todos los hijos de su epic están cerrados, cerrá el epic.
- decido descarto → bd label add <ID> loop-descartado y bd comments add <ID> con la R:.

PASO 2 — Elegir trabajo
- Modo cambio: trabajá sobre la rama del PR abierto. Aplicá SOLO lo que pide la R:.
- Modo nuevo: recorré la tabla de §4 de PLAN-CONTINUACION.md en orden. Tomá la primera fila cuyo
  ticket esté open en BEADS, sin blockers abiertos y sin label loop-descartado ni fuera-loop.
  Si no hay ninguna: entrega de bloqueo con "Decidir: la cola está vacía, ¿qué sigue?".
- Leé el ticket completo (bd show) incluidos los comentarios: el último comentario manda.
- bd update <ID> --status in_progress. Rama: loop/<ID>.

PASO 3 — Estimar antes de escribir código
- Si estimás que el cambio supera 100 líneas, no escribas código: hacé una entrega de partición
  (docs/loop/particiones/<ID>.md + hijos en BEADS, cada uno ≤100 líneas, con objetivo, criterio
  binario y tipo de revisión).

PASO 4 — Implementar
- Solo lo que pide "Cambio requerido". Nada fuera de "Archivos relevantes".
- Tests que fallan si la lógica real se rompe. Sin mocks excesivos ni asserts triviales.
- Suite y `scripts/loop/smoke.sh` en verde. Corré el comando de verificación del ticket y guardá la salida real.
- Si el diff supera 150 líneas: descartá y hacé una entrega de partición.
- Si no llegás a verde: pusheá a loop-wip/<ID> y hacé una entrega de bloqueo.

PASO 5 — Verificación independiente
- Lanzá un subagente sin tu historial. Pasale solo el ticket y el diff. Pedile PASS/FAIL por cada
  criterio de aceptación con evidencia, y que marque cambios fuera de alcance.
- FAIL: corregí una vez y repetí. Si sigue FAIL: entrega de bloqueo.

PASO 6 — Entregar
- Commit, push de loop/<ID>, gh pr create --label loop con el cuerpo de .github/pull_request_template.md.
- Las 3 primeras líneas (Cambió / Decidir / Riesgo) se entienden sin abrir el repo.
- El comando de "Probar" es una sola línea que usa scripts/loop/try.sh.
- Esperá el check de CI hasta 15 minutos y anotá el resultado en el cuerpo.
- Terminá imprimiendo solo las 3 líneas del resumen y la URL.
```

---

## 4. Backlog de tickets ordenado

Esta tabla es la **cola** del loop: toma la primera fila elegible. El estado vive en BEADS.
Reordenar la cola = mover filas en este archivo.

Tipos de revisión: **leer** (el diff alcanza), **decidir** (hay una pregunta cerrada en "Decidir:"),
**probar** (un comando en la VM con `scripts/loop/try.sh`). Ningún ticket toca la UI, así que ninguno
requiere el deploy de QA.

Los primeros 5 son los más fáciles de revisar.

| # | Ticket | Objetivo | Aceptación binaria | Revisión |
|---:|---|---|---|---|
| T01 | `QuantAgent-e35` | Borrar `tests/test_backtest_apscheduler_9wz.py`, que no prueba código propio | El archivo no existe y CI está verde | leer |
| T02 | `QuantAgent-bv8` | Salida del CLI limpia: solo las 5 líneas de métricas | `backtest run ... 2>&1 \| wc -l` devuelve 5 | probar |
| T02b | `QuantAgent-fdi` | El backtest determinista deja de exigir `OPENAI_API_KEY` | `env -u OPENAI_API_KEY scripts/loop/smoke.sh` termina con exit 0 | probar |
| T03 | `QuantAgent-fiu` | `test_parallel_execution` sobre `spy-smoke.csv`, sin skip | El test pasa sin `@pytest.mark.skip` | leer |
| T04 | `QuantAgent-3km` | Guardrail de 730 días para 1h/4h en el provider Yahoo | Test de rango fuera de ventana da error explícito o recorte con log | leer |
| T05 | `QuantAgent-11v` | `--equity-out` exporta la equity curve a CSV | Mínimo de `drawdown_pct` del CSV = drawdown impreso | probar |
| T06 | `QuantAgent-89e.1` | Test `xfail` que reproduce la fila duplicada en reversiones + diagnóstico | Test `xfail(strict=True)` presente; resumen responde "¿round-trip real o error?" | decidir |
| T07 | `QuantAgent-89e` | Corregir la duplicación | `Trades: N` = filas de datos del CSV; el `xfail` pasa a test normal | probar |
| T08 | `QuantAgent-hx0.5` | Script de recálculo independiente: trades, PnL, win rate, PF | No importa `quantagent`; test con 5 trades a mano | leer |
| T08b | `QuantAgent-hx0.8` | CSV con `exit_price` ejecutado; `recalc_metrics.py` verifica el PnL de cada trade | Imprime `PnL por trade: 227/227 filas coinciden`; métricas sin cambio | probar |
| T08c | `QuantAgent-hx0.9` | Slippage por defecto 0,05% por lado; el CLI imprime el slippage usado | 6 líneas, la última `Slippage: 0.05% por lado`; con `TRADING_SLIPPAGE_PCT=0.01` vuelve a −4717.33 | probar |
| T09 | `QuantAgent-hx0.1` | Auditar PnL y total return; documentar fórmula | `recalc_metrics.py` = CLI en trades y PnL | probar |
| T10 | `QuantAgent-hx0.2` | Auditar win rate y profit factor; sin `inf` en DB | `recalc_metrics.py` = CLI en win rate y PF | decidir |
| T11 | `QuantAgent-hx0.6` | Recálculo de max drawdown y Sharpe desde la equity | Test con curva conocida (max DD 25%) | leer |
| T12 | `QuantAgent-hx0.3` | Auditar max drawdown; documentar la equity curve | `recalc_metrics.py --equity` = CLI en drawdown | probar |
| T13 | `QuantAgent-hx0.7` | Decisión: periodos por año para Sharpe en fixtures 24/7 | Doc ≤40 líneas con Sharpe por opción | decidir |
| T14 | `QuantAgent-hx0.4` | Implementar la anualización elegida | `recalc_metrics.py --equity` = CLI en Sharpe (±0.01) | probar |
| T15 | `QuantAgent-y8z` | `backtest verify`: doble corrida y diff | Imprime `OK reproducible` con exit 0 para RSI | probar |
| T16 | `QuantAgent-46h.1` | Generador determinista del fixture diario de 2 años | Dos corridas dan el mismo `sha256sum` | leer |
| T17 | `QuantAgent-46h` | 52-week-high corre sobre ese fixture | `Trades:` ≥ 5 | probar |
| T18 | `QuantAgent-8u8.1` | Generador determinista del fixture 4h de 1 año | Dos corridas dan el mismo `sha256sum` | leer |
| T19 | `QuantAgent-8u8` | Triple Screen corre sobre ese fixture | `Trades:` ≥ 5 | probar |
| T20 | `QuantAgent-piv` | Test golden RSI/spy-90d con los valores ya verificados | El test falla si cambia un trade o una métrica | leer |
| T21 | `QuantAgent-832.1` | RSI portado a `backtesting.py` | Script exit 0; parámetros iguales lado a lado | leer |
| T22 | `QuantAgent-832` | Comparar ambos engines trade por trade | Doc con cada diff explicado o ticket de bug abierto | decidir |
| T23 | `QuantAgent-lcv` | ADR: engine propio vs `backtesting.py` | ADR con 3 opciones; decisión en la `R:` | decidir |
| T24 | `QuantAgent-cv1` | Inventario de worktrees y ramas muertas | Tabla + comando exacto; el loop no borra nada | decidir |
| T25 | `QuantAgent-lp1` | Borrador del plan siguiente con la evidencia de T01–T24 | Doc con cola nueva en el mismo formato | decidir |

A un ticket por día hábil, T01–T25 son 5 semanas. M1 se cumple al cerrar T23.

T08b y T08c se agregaron el 2026-10-06: al revisar T08, Fede detectó que el script daba por bueno el `pnl` del CSV.
El diagnóstico mostró 1% de slippage por lado y un CSV que mezcla precio ejecutado y teórico. Cambian los números
de referencia de rsi/spy-90d (227 / −4717.33), que T09 y T20 tienen que tomar de T08c.

**Fuera del loop** (sesión de escritorio, no entran en la cola):
- `QuantAgent-wwi`, skill de revisión macro semanal: instala en `~/.hermes`.
- `QuantAgent-zui` y `QuantAgent-hak`: son parte de la fase 0 (PR de la sesión 1).
- `QuantAgent-hyt`: reevaluar el validador funcional de QA, comentado en el workflow el 2026-09-29.
- `QuantAgent-u0w`, todo M2, la UI de Streamlit.

---

## 5. Formato de log de revisión

### 5.1 La línea

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
- En la siguiente entrega, el loop copia la línea a `docs/loop/REVIEW-LOG.md` con fecha y URL:

```text
2026-10-01 · #57 · R: bv8 vi: en la VM salieron 5 lineas, sharpe 0.40 decido: merge
```

- `docs/loop/REVIEW-LOG.md` es el historial de la racha: una línea por día hábil revisado.
