# Retrospectiva del loop autodev y plan hacia la "oficina de inversión"

Fecha: 2026-09-16
Autor: Claude (análisis a pedido de Fede)
Contexto: `PLAN-30-DIAS.md` (D01–D15 hechos), `docs/2026-02-19_repository_audit.md`, BEADS, `docs/envelopes/`
Rama: `feature/plan30`

---

## 0. Resumen ejecutivo

1. El loop autodev de mayo (planner → implementer → tester → integration, orquestado por Hermes) no falló por
   calidad de modelos. Falló por **plomería entre fases** y porque produjo **más de lo que una persona puede
   revisar**. 79 de 92 corridas fueron SUCCESS; aun así el resultado neto fue fricción.
2. El modo que sí rindió es el de septiembre: un agente trabajando directo en el repo, en lotes, con Fede
   revisando entre lotes. La unidad que limita el ritmo es la **capacidad de revisión**, no la implementación.
3. Propuesta: reemplazar el pipeline por un ciclo **un ticket → una rama/PR con evidencia → una revisión humana**,
   con verificación independiente pero **sin handoff entre workspaces**, limitado a lo que Fede revisa en 20 min/día.
4. La visión de "oficina de inversión" se ordena en 6 capas. Las capas de arriba (macro, asignación) arrancan como
   **rituales con agente que producen documentos**, no como software. Solo L4 (estrategia + backtest) es código hoy,
   y todavía no cerró M1.
5. Hallazgo urgente: el CSV de `backtest run --out` usa hora de reloj en vez de fecha de vela. Rompe el objetivo del
   entregable de 30 días y se arregla antes de documentar el comando.

---

## 1. Retrospectiva del loop autodev (abril–junio 2026)

### 1.1 Los números

Fuente: 92 archivos `result.json` en `docs/envelopes/`.

| Fase | SUCCESS | PARTIAL | BLOCKED | FAIL |
|---|---:|---:|---:|---:|
| planner | 25 | 0 | 1 | 0 |
| implementer | 17 | 2 | 0 | 2 |
| tester | 16 | 0 | 1 | 0 |
| integration | 7 | 0 | 0 | 0 |
| techlead / direct / salvage | 14 | 4 | 2 | 1 |

Executors: claude-code 29 corridas, codex 3, resto "direct" o salvage desde Hermes (orquestador: Kimi K2.6).

### 1.2 Dónde estuvo la fricción

- **Fallas de entorno, no de código.** Los BLOCKED/PARTIAL dicen: root checkout sucio, executor que no ve el
  worktree declarado en el envelope, DB local faltante (`EXECUTION_PRECONDITION_MISSING`), "falso verde"
  (router reporta éxito, artifact canónico BLOCKED). Un fallo de CI generó 6 tickets P0 hotfix duplicados
  (`QuantAgent-046` + 5 duplicados).
- **Costo fijo desproporcionado.** `kkj.5` (tooltips en una pestaña) pasó por 3 fases y dejó 12 archivos de artifacts.
  `kkj.11` sigue `in_progress` desde 2026-05-26 con 5 corridas y 3 worktrees prunables. Hoy hay 9 worktrees
  registrados (6 prunables) y decenas de ramas `feature/QuantAgent-*` muertas.
- **Construcción sobre fachada.** `kkj.2`–`kkj.9` agregaron UI arriba de una vista de backtesting que nunca ejecuta
  nada (`apps/streamlit/views/backtesting.py`). Mergeaban "con éxito" sin nada verificable para el supervisor.
- **Throughput > capacidad de revisión.** Hasta 5 tickets por corrida, 4 corridas por día. Sin revisión humana el
  loop degeneró: los tickets se aprobaban por labels, no por uso.
- **El tester nunca rechazó nada.** De 20 corridas de fase tester, 0 FAIL. Un verificador que siempre aprueba
  no agrega independencia; agrega latencia y un workspace más que puede quedar sucio.

### 1.3 Lo que sí funcionó

2026-09-15/16: Claude Code trabajando en el checkout principal, lotes de 3–5 tareas, revisión de Fede entre lotes.
Cerró D01–D15 del plan en dos sesiones. Ya está registrado en `PLAN-30-DIAS.md` §3 ("Recalibración 2026-09-16").

### 1.4 Diagnóstico

No fue Hermes, ni Kimi, ni "mal uso". Fue el diseño del loop: cuatro agentes con contratos entre sí y con
handoff por labels de BEADS y envelopes en disco, sin el humano en el único punto donde agrega valor
irremplazable: **usar el resultado**. La conclusión de Fede ("necesito estar yo ahí probando") no es una derrota;
es la restricción de diseño correcta.

Nota: `hermes cron list` al 2026-09-16 ya no muestra ningún cron autodev de QuantAgent; `~/CLAUDE.md` lo documenta
como activo y está desactualizado en ese punto.

---

## 2. El loop propuesto: un ticket, un PR, una revisión

### 2.1 Regla de diseño

> El sistema produce como máximo lo que Fede puede revisar en 20 minutos por día. Un PR por día.

### 2.2 Loop A — sin Fede (nocturno)

1. Tomar el primer ticket de BEADS con label `ai-ready`, en orden de prioridad. **Uno solo.**
2. Un solo agente (Claude Code) en una rama `feature/<ID>` desde `main` limpio: lee el contrato del ticket,
   implementa, escribe tests, corre la suite.
3. **Verificación independiente dentro de la misma corrida** (ver §2.4): un subagente con contexto fresco recibe
   solo el ticket y el diff, corre los comandos de aceptación y devuelve PASS/FAIL con evidencia. Si FAIL, el
   implementador corrige una vez; si sigue FAIL, no hay PR.
4. Abre un PR cuyo cuerpo tiene tres partes obligatorias: comando exacto de verificación, salida real que produjo,
   y qué NO hizo. Si no llega a estado verificable: comenta el ticket con el bloqueo y para.
5. CI (lint + tests) corre sobre el PR.
6. Aviso por Telegram: 5 líneas.

### 2.3 Loop B — Fede (30 min)

1. Leer el resumen del PR (5 min).
2. Correr el comando de verificación y mirar el resultado (10 min). **Esto no se delega.**
3. Merge, o un comentario de una línea con lo que está mal (5 min).
4. Marcar el siguiente ticket `ai-ready`, o pedir que se escriba en formato contrato (10 min).

### 2.4 Por qué no separar implementer y tester en fases distintas

La objeción es válida: un agente que testea su propio código tiende a confirmar lo que hizo. La independencia
de verificación **hay que conservarla**. Lo que conviene eliminar no es la verificación independiente sino la
**fase orquestada aparte**: corrida propia, workspace propio, envelope propio, handoff por labels.

| Qué da independencia | Costo de plomería | Evidencia en este repo |
|---|---|---|
| Fase tester separada (workspace + envelope + label) | Alto: es donde aparecieron los BLOCKED/falso verde | 20 corridas, 0 FAIL: no rechazó nunca |
| Subagente con contexto fresco en la misma corrida y rama | Nulo: mismo checkout, mismo branch | Es lo que hace `/code-review` de Claude Code |
| CI sobre el PR | Nulo: ya existe | Objetivo y determinista |
| Fede corriendo el comando de aceptación | 10 min/día | Lo único que detectó la fachada de UI y el bug de timestamps |

La independencia que importa es de **contexto** (el verificador no vio el razonamiento del implementador) y de
**criterio** (recibe los AC del ticket, no el diff comentado). Ambas se logran con un subagente fresco en la misma
sesión. Lo que no aporta es la independencia de **proceso** (otro cron, otro worktree), que fue justamente la fuente
de fricción. Tres capas de verificación quedan: subagente fresco, CI, y Fede como acceptance tester.

### 2.5 Implementación, de menor a mayor esfuerzo

| Opción | Qué es | Costo | Trade-off |
|---|---|---|---|
| A. Manual asistido | Fede abre Claude Code y dice "tomá el siguiente ticket" | 0 | Requiere presencia para arrancar. Es lo que rindió en septiembre |
| B. Routine en la nube | `schedule` de Claude Code corre un agente cloud a hora fija sobre el repo y abre el PR | 1 sesión | Aislado, desde `main` limpio, sin worktrees sucios ni dependencia de la VM |
| C. Hermes cron + `claude -p` | Hermes lanza Claude Code headless en la VM | 2–3 sesiones | Reusa infra, pero vuelve al entorno que dio los BLOCKED por checkout sucio |

Recomendación: **A esta semana** (cerrar el plan de 30 días), **B** cuando el CLI esté en `main` y haya 3 tickets
en formato contrato esperando. Hermes conserva el briefing PM matutino (leer BEADS + PR abierto → Telegram).

---

## 3. La visión completa, en capas

Antecedente: fase 2E de `phase2_roadmap.md` (Macro Agent → Sector Agent → Portfolio Optimizer). Bien pensada como
arquitectura, mal pensada como orden: asume software desde el día uno.

| Capa | Qué decide | Estado hoy | Primera versión realista |
|---|---|---|---|
| L1 Macro | Régimen global: tasas, monedas, índices, geopolítica, impacto por región/sector | Nada | Nota semanal en Obsidian generada por un skill de Hermes con web search. Cero código |
| L2 Asignación | Peso por región / sector / clase de activo según horizonte | Nada | Planilla con asignación objetivo + agente que la cuestiona contra L1 |
| L3 Universo | Qué activos concretos son candidatos | Existe (universe management en config) | Alimentar el universo desde L2 |
| L4 Estrategia + backtest | Qué estrategia por activo, confirmada con datos | **Alcance actual.** CLI recién cableado; métricas sin auditar | Plan de 30 días, luego `hx0` partido y `piv` |
| L5 Ejecución | Paper trading, luego real | PaperBroker mock existe; Alpaca no | M2–M4 |
| L6 Señales alternativas | Sentimiento de noticias, redes, foros | Nada | Última: la más cara y menos validada |

Dos reglas:

- **L1 y L2 no son software todavía.** Son rituales semanales con agente. Son lo que más enseña de trading (gap
  declarado) y cuestan cero código. Cuando tras ~2 meses de notas se repitan las mismas preguntas, ahí se convierten
  en software.
- **Nada arriba de L4 en código mientras L4 no diga la verdad.** 14 meses de proyecto y M1 no está cerrado. Todo lo
  demás se apoya en que el backtesting sea confiable.

---

## 4. Hallazgo urgente en L4 (2026-09-16)

Al correr `python -m quantagent.cli backtest run --strategy rsi --fixture spy-90d --out run.csv`:

- `entry_time` / `exit_time` salen de `Trade.opened_at` / `closed_at`, que por default son `datetime.utcnow`
  (hora de reloj al correr), no el timestamp de la vela. Ejemplo real: `2026-09-16T20:50:12` para datos del fixture.
  Así **no se puede revisar un trade contra el gráfico**, que es el objetivo entero del entregable (`gg6`).
- Reporta 111 trades (la demo del plan asumía ~11) y el stdout se llena de rechazos "Daily loss limit exceeded".

Se arregla **antes de D18** (documentar el comando). Archivos: `quantagent/cli/backtest.py` (armado de `TradeRow`),
`quantagent/portfolio/manager.py` (creación de `Trade`), posiblemente `quantagent/backtesting/backtest.py`
(propagar timestamp de vela al abrir/cerrar).

---

## 5. Plan accionable: 4 semanas, 30 min/día

Cada fila = "IA implementa, Fede revisa". Criterio de hecho: algo que Fede ve con sus ojos.

### Semana 1 — cerrar el plan de 30 días con el CSV arreglado

| # | Tarea | Fede verifica |
|---|---|---|
| 1 | Timestamps de vela en el trade log; bajar el ruido de rechazos en stdout | Abre el CSV: las fechas caen dentro del rango del fixture |
| 2 | D16: tabla de métricas en stdout | Corre el comando y ve las 5 métricas |
| 3 | D17 + D18: test E2E vía `CliRunner` y sección en README | Copia la sección del README y reproduce la demo |
| 4 | D20–D22: cerrar `gg6`, actualizar `milestone-tracker.md`, merge a `main` | `bd show QuantAgent-gg6` = CLOSED; demo corre desde `main` limpio |

D19 (botón Streamlit) se omite: sigue siendo UI sobre una fachada.

### Semana 2 — backlog consumible por el loop

| # | Tarea | Fede verifica |
|---|---|---|
| 5 | Partir `hx0` en 5 tickets formato contrato, uno por métrica (win rate, profit factor, Sharpe, drawdown, PnL), cada uno validado a mano contra el CSV en planilla | Lee 5 tickets de ≤1 pantalla cada uno y los marca `ai-ready` |
| 6 | Limpieza: prune de 6 worktrees muertos, archivar ramas, cerrar o reescribir `kkj.11` | `git worktree list` muestra solo el checkout principal |
| 7 | Armar la routine nocturna (opción B) y correrla sobre el primer ticket | Recibe el Telegram y encuentra el PR |

Esto además resuelve con datos la decisión abierta "engine propio vs `backtesting.py`" (§1.4-B del plan de 30 días).

### Semanas 3–4 — loop en régimen y primera capa de arriba

| # | Tarea | Fede verifica |
|---|---|---|
| 8 | Un PR por día desde la routine | 20 min de revisión diaria |
| 9 | Skill "revisión macro semanal" en Hermes que escribe una nota al vault los domingos. Prerrequisito: cerrar el gap P0 de web search de Hermes (Tavily/Exa) | Lee la nota del domingo y anota una pregunta que le dejó |
| 10 | Decidir con datos si el engine propio se queda o se cambia; escribir el siguiente plan de 30 días | Toma la decisión en una línea |

### Fuera de alcance en estas 4 semanas

Alpaca y todo M2; APScheduler (`u0w`); sentimiento de noticias/redes (L6); UI nueva en Streamlit; reactivar el
pipeline de 4 fases en Hermes.

---

## 6. Referencias

- `PLAN-30-DIAS.md` — plan diario vigente y checklist
- `docs/2026-02-19_repository_audit.md` — auditoría previa
- `docs/02_planning/phase2_roadmap.md` §2E — sketch original de agentes macro
- `docs/envelopes/` — artifacts del loop autodev (fuente de §1.1)
- `~/CLAUDE.md` (VM) — contrato de ticket para autodev (formato a reusar para los tickets `ai-ready`)
