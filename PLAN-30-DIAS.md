# PLAN-30-DIAS

Generado: 2026-09-14
Ventana: D01 = 2026-09-15 (mar) → D22 = 2026-10-14 (mié)
Presupuesto: 45 min/día, lun–vie. 22 días hábiles ≈ 16,5 h brutas, ~15 h efectivas.

---

## 1. Auditoría

### 1.1 Continuidad: el dato duro

| Métrica | Valor |
|---|---|
| Primer commit | 2025-07-09 |
| Último commit | 2026-07-01 (`d3af0373`) |
| Días sin commits al 2026-09-14 | **75** |
| Vida del proyecto | 14 meses |

Commits por mes:

```
2025-07  29   2025-11  36   2026-02  76   2026-05  223
2025-08  45   2025-12   0   2026-03   0   2026-06    3
2025-09  17   2026-01  97   2026-04  48   2026-07    3
2025-10  15                               2026-08+   0
```

El patrón no es "poco tiempo". Es **ráfaga y abandono**: 223 commits en mayo, 3 en junio,
3 en julio, cero desde entonces. Ya hubo dos huecos de un mes completo antes (2025-12, 2026-03).
El proyecto no muere por falta de horas; muere porque las horas llegan concentradas
y después nada sostiene el hilo.

Hay además un **segundo plan abandonado**: los tickets con label `plan-20260704` (7 tickets,
lanes m1/m2/m3) se crearon el 2026-07-04, tres días después del último commit.
Ninguno se tocó. Es decir: el ciclo anterior terminó exactamente acá — haciendo un plan
en lugar de trabajo. Este documento corre el riesgo de ser el tercero.

### 1.2 Estado real del código

**Funciona end-to-end (verificado hoy):**

- **Entorno**: levanta sin fricción. `docker ps` → 4 contenedores healthy (`quantagent-dev-db`,
  `quantagent_qa`, `quantagent_qa_db`). `.venv` con Python 3.12.3 operativo.
  Alembic en head (`d1e2f3a4b5c6`), 12 tablas creadas en `quantagent_dev`.
- **Suite de tests**: 873 tests colectados, **771 pasan, 1 falla, 21 skipped** en 45 segundos.
  Sin errores de colección. Esto es mucho mejor de lo que el proyecto "se siente".
- **Motor de backtesting headless**: `scripts/run_three_strategy_backtests.py` corre el engine
  contra SQLite en memoria con datos sembrados, sin red ni LLM. Salida real en
  `tmp/three_strategy_backtests.json` (1 de julio).
- **CLI**: `python -m quantagent.cli` arranca; tiene un solo grupo (`profile`), completo y testeado.
- **Estrategias deterministas**: RSI, Triple Screen y 52-week High existen, están registradas
  en `quantagent/strategy/registry.py` y tienen tests propios que pasan.

**A medio hacer:**

- **La vista de backtesting de Streamlit es una fachada.** `apps/streamlit/views/backtesting.py:191`
  crea la fila `BacktestRun` en la DB y muestra literalmente
  `"Run {id} created. Backend execution wiring pending."` No ejecuta nada. La tabla de runs
  lista filas con `status = "completed" if r.total_trades is not None else "pending"`,
  o sea que todo lo creado desde la UI queda en `pending` para siempre.
  **Esto es lo más engañoso del repo**: parece que hay backtesting en la UI y no lo hay.
- **Bug P0 `QuantAgent-iip` probablemente ya arreglado, nunca cerrado.** Tres commits
  (`411ad657`, `600ba2d0`, `ace682f4`, 30/6–1/7) atacan exactamente el data bleed entre
  timeframes. Hoy corrí los 14 tests que cubren ese modo de falla
  (`test_backtest_end_to_end_timeframe_isolation`, `test_stale_position_cleanup`,
  `test_backtest_run_isolation`): **los 14 pasan**. El ticket sigue OPEN y **bloquea 4 tickets**
  (`hx0`, `y8z`, `832`, `13a`). Es el ítem de mayor apalancamiento del backlog y cuesta ~2 h.
- **Trabajo sin commitear desde el 1 de julio**: `tests/test_strategy_validation_fixtures.py`
  (untracked, 2 tests que pasan), `tmp/` (script de inspección + log de 1,4 MB),
  `.beads/issues.jsonl` modificado. La última sesión terminó a mitad de camino y nunca cerró.
- **APScheduler**: existe en `quantagent/trading/scheduler.py` pero **solo para paper trading**.
  No hay ejecución background de backtests. Eso es lo que pide `u0w`.

**Muerto o falso:**

- `tests/test_parallel_execution.py::test_parallel_execution` — **el único test que falla**.
  Busca `benchmark/btc/BTC_4h_1.csv`, que no existe en el repo. Es un test que nunca va a
  pasar en este árbol.
- `tests/test_backtest_apscheduler_9wz.py` — **cobertura falsa**. Testea la librería APScheduler
  (`scheduler.start()`, `assert scheduler.running`), no código de QuantAgent. Pasa siempre
  y no valida nada propio.
- **README desactualizado**: el Quick Start manda `pip install -r requirements.txt`
  (**el archivo no existe**), `conda create python=3.11` (el venv real es 3.12) y
  `python examples/run_backtest.py` (existe, pero requiere `OPENAI_API_KEY` y red).
  Un lector nuevo — o vos en tres meses — se traba en el paso 1.
- **6 worktrees `prunable`** en `/tmp/autodev-worktrees/` (kkj.3, kkj.4, kkj.5, kkj.11 ×3) +
  3 en `.worktrees/`. Restos del autodev de mayo.
- **342 archivos `.md` en `docs/envelopes/`** — artifacts de corridas autodev. Ruido puro
  al buscar algo en `docs/`.
- **Las 3 estrategias deterministas no están validadas.** En la corrida del 1 de julio:
  Triple Screen → **0 trades**; 52-week High → **0 velas procesadas**; RSI → 13 trades,
  **0 ganadores, win rate 0%**. El milestone M1 ("3 estrategias sin bugs") no está
  ni cerca, y el tracker `milestone-tracker.md` dice `Progreso: 0/0 tickets (TBD)`.

### 1.3 Inventario de tickets con estimación honesta

Estado BEADS: 109 issues totales — 13 open, 1 in_progress, 10 blocked, 95 closed, 3 ready.

| ID | Título | Prio | Estimación honesta | Nota |
|---|---|:---:|---:|---|
| `iip` | Backtest 4H: counter en 0 / trades idénticos a 1H | P0 | **2 h** | Casi seguro ya arreglado; falta reproducir y cerrar. Bloquea 4. |
| `gg6` | Exportar trade log a CSV desde la UI | P1 | **3 h** *(spec)* / **9 h** *(real)* | Ver §1.4-A: falta FK y falta ejecución real. |
| `hx0` | Auditar y corregir métricas del engine | P1 | **12–15 h** | Matemática crítica + validación manual en planilla. Todo el mes en un ticket. |
| `y8z` | Comando de verificación de reproducibilidad | P1 | **6–8 h** | Bloqueado por `iip`. |
| `piv` | Test de regresión golden | P1 | **4 h** | Bloqueado por `hx0`. No arranca sin valores validados a mano. |
| `832` | Cross-validation vs `backtesting.py` | P1 | **12–20 h** | Alinear semántica entre engines. Alto riesgo de falso positivo. |
| `u0w` | APScheduler para ejecución background de backtests | P1 | **25–40 h** | Scheduler + worker + servicio docker + polling UI. |
| `13a` | Diseño: interfaz Broker + plan Alpaca | P2 | **6 h** | Bloqueado por `iip`. |
| `yme` | Implementar AlpacaBroker (paper) | P2 | **20–30 h** | M2. |
| `qr6` | Config y guardrails credenciales Alpaca | P2 | **8–10 h** | M2. |
| `y62` | Reconciliación posiciones DB vs Alpaca | P2 | **10–12 h** | M2. |
| `b41` | Digest operativo diario paper trading | P2 | **8 h** | M3. |
| `kkj.10` | Seedear catálogo base de configuración | P1 | **6 h** | Bloqueado por `kkj.11`. |
| `kkj.11` | Routing multi-provider por rol | P3 | **10–12 h** | In progress desde mayo. |

**Total del backlog planificado: ~130–180 h.**
A 15 h/mes son **9 a 12 meses** de trabajo sostenido. Si el objetivo mental era
"llegar a live trading" (M4–M5), eso está a **2–3 años** a este ritmo. Vale decirlo ahora
y no en marzo.

### 1.4 Deuda técnica y decisiones abiertas que bloquean

**A. `Trade` no tiene FK a `backtest_run_id`. Esto rompe `gg6` en silencio.**
`Signal` y `ActivePosition` sí la tienen (`quantagent/models.py:187`, `:367`).
`Trade` (`quantagent/models.py:210`) tiene `environment` pero **ningún vínculo con el run**.
Los trades se crean en un solo lugar (`quantagent/portfolio/manager.py:183`) sin ese dato.
Consecuencia: hoy es **imposible** responder "dame los trades del run 17" sin heurísticas
de ventana temporal. `gg6` está escrito como si el dato existiera. No existe.
Es una migración Alembic chica, pero es un prerrequisito no declarado.

**B. Decisión abierta sin dueño: ¿engine propio o engine de terceros?**
`832` la deja explícita en sus notas: *"si tras cerrar iip las divergencias persisten sin
explicación, la recomendación es swappear el engine por vectorbt/backtesting.py"*.
Mientras esa decisión no se tome, `hx0` (auditar métricas, 12–15 h) puede ser trabajo tirado
a la basura. **No se puede empezar `hx0` antes de resolver esto.**

**C. `iip` bloquea artificialmente medio backlog.**
4 tickets bloqueados por un bug que los tests dicen que ya no existe. El costo real no es
el bug: es que el grafo de dependencias está mintiendo y hace que el backlog parezca
intransitable.

**D. La UI promete algo que no cumple.**
Cualquier trabajo de UI sobre backtesting antes de resolver `u0w` construye sobre una fachada.

**E. Cobertura de tests inflada.** 873 tests suenan a mucho. Una parte testea librerías
de terceros (`test_backtest_apscheduler_9wz.py`) o archivos inexistentes
(`test_parallel_execution.py`). El número tranquiliza más de lo que debería.

### 1.5 Por qué este proyecto es difícil de retomar

Corrí el ciclo completo de reapertura hoy. La fricción **no está donde uno esperaría**:

**Lo que NO es el problema** (y conviene saberlo para no perder días ahí):
el entorno arranca, la DB está migrada, los tests corren en 45 segundos, las dependencias
están instaladas. Técnicamente podés estar produciendo en 5 minutos.

**Lo que SÍ es el problema:**

1. **El repo no dice dónde estabas.** El último commit es `docs: retain validated investigation
   artifacts`. No dice qué seguía. Para reconstruir "¿en qué andaba?" hoy tuve que cruzar
   commits + comentarios de BEADS + `tmp/three_strategy_backtests.json` + la vista de Streamlit.
   Eso son **20–30 minutos de arqueología**, es decir más de la mitad de una sesión diaria.
   A 45 min/día, si cada lunes perdés media sesión reconstruyendo contexto, el proyecto
   avanza a un tercio de velocidad.

2. **Trabajo huérfano sin commitear.** `tmp/` y un test untracked del 1 de julio.
   Al reabrir, lo primero que ves es `git status` sucio y no sabés si eso era bueno o basura.
   Duda = parálisis.

3. **El estado del backlog es falso.** `iip` figura como P0 abierto bloqueando 4 tickets.
   Abrís BEADS, ves un P0 crítico, y la reacción natural es "uf, esto está roto, no tengo
   45 minutos para esto" — cuando en realidad ya estaba arreglado. **El backlog te desalienta
   con información desactualizada.**

4. **La documentación es abundante e inútil para retomar.** 256 archivos `.md` en `docs/`
   más 342 envelopes. Hay READMEs por carpeta, pero ninguno responde
   "¿cuál es el próximo paso concreto?". `milestone-tracker.md` dice `TBD` en cada campo.
   Documentación de proceso, cero documentación de estado.

5. **Onboarding roto en el paso 1.** El README manda instalar un `requirements.txt`
   que no existe. Si alguna vez clonás esto en otra máquina, o si alguien más lo mira,
   se traba antes de empezar.

6. **El tamaño del backlog es hostil a sesiones de 45 minutos.** Casi todo lo planificado
   pesa 6–40 h. **No hay un solo ticket que entre en una sesión.** Para 45 minutos, un
   backlog sin tareas de 45 minutos es equivalente a no tener backlog.

### 1.6 Qué está sobredimensionado

Dicho sin vueltas:

- **`u0w` (APScheduler, 25–40 h)** es 2–3 meses de tu presupuesto real en un solo ticket.
  Está marcado P1 y aparece en la cola de "ready". Es una trampa: parece el próximo paso
  lógico y te come el trimestre.
- **M2 completo (`13a` + `yme` + `qr6` + `y62` ≈ 45–58 h)** son 3–4 meses. Alpaca no es
  para este mes ni para el que viene.
- **`hx0` (12–15 h)** consume el mes entero y depende de una decisión no tomada (§1.4-B).
  Peor forma posible de gastar los primeros 30 días.
- **`832` (12–20 h)** es valioso y probablemente sea lo correcto a mediano plazo, pero
  alinear semántica entre dos engines en tandas de 45 minutos es una receta para
  abandonar a mitad.
- **El plan `plan-20260704` entero** asume sesiones supervisadas largas (varios tickets dicen
  *"Sesión supervisada Claude Code (Fable)"*). Ese modo de trabajo no existe bajo la
  restricción actual. **El plan de julio fue diseñado para una persona con otro presupuesto
  de tiempo.** Por eso no arrancó nunca.

---

## 2. Entregable de 30 días

### El entregable

> **`quantagent backtest run` — un comando que corre un backtest determinista sobre datos
> fijos versionados y emite (a) una tabla de métricas en pantalla y (b) un CSV con el
> trade log completo, abrible en una planilla.**

Demo de 2 minutos:

```bash
$ python -m quantagent.cli backtest run --strategy rsi --fixture spy-90d --out run.csv

Strategy: RSIMeanReversionStrategy   Fixture: spy-90d (1h, 2160 velas)
Trades: 11   Win rate: 18.2%   Profit factor: 0.21   Sharpe: -0.26   PnL: -125.85
Trade log → run.csv (11 filas)

$ head -3 run.csv
entry_time,exit_time,symbol,side,qty,entry_price,exit_price,stop_loss,pnl,exit_reason
2026-04-02T14:00:00,2026-04-03T09:00:00,SPY,long,12,512.40,508.15,502.15,-51.00,stop_loss
```

Se abre el CSV en una planilla y se ven los trades. Fin de la demo.

### Por qué este y no otro

1. **Cierra un ciclo completo**: entrada (estrategia + datos fijos) → ejecución real del engine
   → salida verificable y legible por un humano. No es preparación ni refactor.
2. **Es la intención real de `gg6`**, que en su propio contexto dice: *"La validación manual
   de M1 requiere que Fede revise trades a mano contra el gráfico y recalcule métricas en
   planilla"*. Eso es exactamente el CSV. `gg6` lo pide vía UI; la UI **no ejecuta backtests**
   (§1.2), así que entregarlo por UI significaría exportar runs vacíos. Se entrega por el
   camino que sí ejecuta — el CLI — y el botón de UI queda como cierre opcional (D18–D19).
3. **Prioriza backlog existente**: cierra `iip` (P0, desbloquea 4 tickets) y entrega el
   contenido de `gg6`. No inventa trabajo nuevo salvo la migración del §1.4-A, que es un
   prerrequisito oculto de `gg6`, no un ticket nuevo.
4. **Desbloquea lo que viene sin comprometerse a ello**: `hx0` necesita trade log en planilla
   para su criterio de aceptación; `piv` necesita valores golden que salen de este CSV;
   `y8z` necesita comparar dos corridas — y ahora hay un artefacto comparable.
5. **Entra en el presupuesto**: cada pieza (migración, serializador, comando, test) se cierra
   en una sesión de 45 minutos. Nada obliga a mantener contexto entre días.
6. **Es honesto**: reemplaza una fachada de UI por algo que realmente corre.

### Fuera de alcance — explícito

**No se toca en estos 30 días:**

- `u0w` — APScheduler / ejecución background de backtests. **No se empieza.**
- Wiring de ejecución en la vista de Streamlit. La UI sigue diciendo "pending".
- Todo M2: `13a`, `yme`, `qr6`, `y62` — Broker, Alpaca, reconciliación.
- `b41` — digest de paper trading (M3).
- `hx0` — auditoría de métricas. **Se difiere a propósito**: depende de la decisión abierta
  del §1.4-B y no entra en 15 h.
- `832` — cross-validation contra `backtesting.py`. Es el siguiente mes, no este.
- `piv` — test golden. Necesita valores validados a mano que este mes no se generan.
- `kkj.10` / `kkj.11` — configuración y routing multi-provider.
- **No se investiga por qué Triple Screen y 52-week High dan 0 trades.** Se registra como
  ticket y nada más. El CLI se entrega con RSI.
- Estrategias LLM. El entregable es 100% determinista, sin red ni API keys.
- Refactors, limpieza de `docs/envelopes/`, poda de worktrees, reducción del `.git` de 485 MB.
- Performance, paralelización, optimización.

**Regla de corte:** si un día aparece algo interesante fuera de esta lista, se abre ticket
en BEADS y se sigue con la tarea del día. No se persigue.

---

## 3. Plan diario

### Recalibración 2026-09-16: lotes de revisión en vez de cadencia diaria

El plan original está calibrado para trabajo manual (45 min/día). Trabajando con Claude Code
la unidad real que limita el ritmo no es "tiempo de implementación" sino **capacidad de
revisión** — leer el diff, correr tests, entender qué cambió antes de seguir. D07-D09 se
hicieron en una sola sesión con una sola tanda de revisión, lo que confirma que el "día" del
plan original equivale más o menos a una tarea chica, y que agrupar 3-5 tareas relacionadas
por checkpoint es razonable.

De D10 en adelante, los días se ejecutan agrupados en **lotes**. Dentro de un lote no se para
a revisar entre tareas (se avanza igual que D07-D09); entre lotes sí hay un checkpoint
explícito antes de seguir. El criterio de "hecho" por día (tabla de abajo) no cambia — sigue
siendo binario por fila — solo cambia cuándo se para a revisar.

| Lote | Días | Qué entrega | Checkpoint al cerrar el lote |
|---|---|---|---|
| A — Datos y export | D10–D12 | Fixture versionado (`spy-90d.csv` + loader) y serializador `trades_to_csv` con su test | Revisar diff de `tests/fixtures/` + `export.py`, correr suite completa |
| B — CLI end-to-end | D13–D17 | Grupo `backtest run` con flags, wiring real, `--out`, métricas en stdout, test E2E vía `CliRunner` | Correr el comando a mano contra `spy-90d`, ver la demo real funcionando |
| C — Cierre y documentación | D18–D20 | README del comando, botón de descarga en Streamlit, cierre de `gg6` + `milestone-tracker.md` | Revisar README y UI en el navegador, confirmar `bd show QuantAgent-gg6` = `CLOSED` |
| D — Colchón y merge | D21–D22 | Tickets de lo que quedó afuera, merge a `main`, demo de punta a punta | Demo corrida desde `main` limpio, decisión sobre próximo plan de 30 días |

Convenciones:
- `bd` = `~/repos/agents/autodev-runner/scripts/bd_safe.sh`
- Todo día termina con **commit en rama** (`feature/plan30-<tema>`) y `git status` limpio.
  Nunca se commitea a `main` directo.
- "Hecho" es binario: o el comando devuelve lo que dice la fila, o el día no está hecho.

| Día | Fecha | Tarea | Criterio de "hecho" | Min |
|:---:|---|---|---|---:|
| 1 | 15/09 | Cerrar el trabajo huérfano del 1/7: commitear `tests/test_strategy_validation_fixtures.py` y `.beads/issues.jsonl`; agregar `tmp/` a `.gitignore` | `git status --porcelain` no imprime nada | 25 |
| 2 | 16/09 | Neutralizar `tests/test_parallel_execution.py` (marcar `@pytest.mark.skip` con motivo: falta `benchmark/btc/BTC_4h_1.csv`) + abrir ticket BEADS del faltante | `pytest -q -m "not slow and not api"` → `0 failed`; ticket creado con ID anotado | 30 |
| 3 | 17/09 | Arreglar el Quick Start del README: sacar `requirements.txt`, poner `pip install -e ".[dev]"`, Python 3.12, y el comando real de tests | Un lector puede ejecutar los 4 pasos sin error; no queda ninguna referencia a `requirements.txt` en el README | 30 |
| 4 | 18/09 | Crear `scripts/repro_iip_timeframe.py`: siembra OHLCV determinista y corre RSI en 1H y 4H sobre los mismos datos | El script corre sin excepción e imprime `evaluations`, `trades` y `pnl` de ambos timeframes | 45 |
| 5 | 21/09 | Ejecutar el repro y registrar el veredicto en un comentario de BEADS en `iip` con los números de ambas corridas | Comentario visible en `bd show QuantAgent-iip` con la tabla 1H vs 4H | 40 |
| 6 | 22/09 | Según D05: si 4H difiere de 1H → cerrar `iip` (`bd close`); si sigue idéntico → actualizar `iip` con la nueva hipótesis y **replanificar D07–D09** | `bd show QuantAgent-iip` muestra `CLOSED`, **o** el ticket tiene comentario nuevo con hipótesis y próximo paso | 30 |
| 7 | 23/09 | Migración Alembic: agregar `backtest_run_id` (FK nullable a `backtest_runs.id`, indexada) a `trades` | `alembic upgrade head` OK y `\d trades` muestra la columna | 45 |
| 8 | 24/09 | Propagar `backtest_run_id` en la creación de trades (`portfolio/manager.py:183` ← lo pasa `backtesting/backtest.py`) | Suite completa sigue en `0 failed` | 45 |
| 9 | 25/09 | Test: dos backtests consecutivos sobre los mismos datos → los trades de cada run tienen su propio `backtest_run_id` y no se mezclan | El test nuevo pasa y **falla** si se revierte el cambio de D08 (verificado a mano) | 45 |
| 10 | 28/09 | Versionar el fixture: guardar el OHLCV determinista de D04 como CSV en `tests/fixtures/spy-90d.csv` + loader que lo siembra en la DB | El loader corre y deja N filas en `market_data`; el CSV está commiteado | 40 |
| 11 | 29/09 | `quantagent/backtesting/export.py`: función `trades_to_csv(trades) -> str` con las 10 columnas del entregable | Función implementada; devuelve string con header + una fila por trade | 40 |
| 12 | 30/09 | Test unitario del serializador: columnas exactas, encoding UTF-8, valores formateados, lista vacía → solo header | El test pasa y falla si se cambia el orden de columnas | 40 |
| 13 | 01/10 | Registrar el grupo `backtest` en `quantagent/cli/__main__.py` con el subcomando `run` (flags `--strategy`, `--fixture`, `--out`), todavía sin lógica | `python -m quantagent.cli backtest run --help` imprime los 3 flags | 30 |
| 14 | 02/10 | Cablear la ejecución: el comando carga el fixture, corre el backtest y termina con exit code 0 | `python -m quantagent.cli backtest run --strategy rsi --fixture spy-90d` corre y sale con `echo $?` = 0 | 45 |
| 15 | 05/10 | El comando escribe el CSV en `--out` usando el serializador de D11 | El archivo existe, abre en planilla y tiene tantas filas de datos como trades reportó el run | 45 |
| 16 | 06/10 | Imprimir la tabla de métricas en stdout (trades, win rate, profit factor, Sharpe, PnL) | La salida del comando muestra las 5 métricas con valores numéricos | 35 |
| 17 | 07/10 | Test end-to-end del comando vía `CliRunner`: corre, exit code 0, CSV generado con header correcto | El test pasa en la suite completa (`0 failed`) | 45 |
| 18 | 08/10 | Documentar el comando: sección en `README.md` con el comando exacto, la salida esperada y las columnas del CSV | Copiar y pegar la sección del README reproduce la demo | 35 |
| 19 | 09/10 | Botón `st.download_button` en `apps/streamlit/views/backtesting.py` que exporta el trade log del run seleccionado usando el serializador de D11 | El botón aparece para un run con trades y descarga un CSV no vacío | 45 |
| 20 | 12/10 | Cerrar `gg6` en BEADS con evidencia (comando + captura del CSV); actualizar `milestone-tracker.md` con el estado real de M1 | `bd show QuantAgent-gg6` = `CLOSED`; el tracker ya no dice `TBD` en "Milestone actual" | 30 |
| 21 | 13/10 | **Colchón.** Si no hay deuda: abrir tickets BEADS de lo que apareció y se dejó pasar (0 trades en Triple Screen / 52-week, `test_backtest_apscheduler_9wz` falso, worktrees prunables) | Cero días marcados NO en el checklist, **o** ≥1 ticket nuevo creado | 45 |
| 22 | 14/10 | **Colchón + cierre.** Merge de `feature/plan30-*` a `main` y demo grabada o ejecutada de punta a punta | La demo corre desde `main` limpio en menos de 2 minutos | 45 |

### Versión mínima de 10 minutos (por día)

Para los días en que solo hay 10 minutos. Deja el proyecto consistente igual; el día cuenta
como hecho a medias y se completa en el colchón.

| Día | Versión de 10 min |
|:---:|---|
| 1 | Solo agregar `tmp/` a `.gitignore` y commitear eso |
| 2 | Solo agregar el `@pytest.mark.skip` con el motivo; el ticket queda para después |
| 3 | Solo corregir la línea de `requirements.txt` → `pip install -e ".[dev]"` |
| 4 | Copiar el esqueleto de siembra desde `scripts/run_three_strategy_backtests.py` a `scripts/repro_iip_timeframe.py` y commitear sin ejecutar |
| 5 | Correr el script y pegar la salida cruda en `tmp/` (sin redactar el comentario de BEADS) |
| 6 | Solo el comentario en BEADS con el veredicto en una línea; el `close` queda para después |
| 7 | Generar el archivo de migración vacío (`alembic revision`) y commitearlo |
| 8 | Agregar el parámetro `backtest_run_id=None` a la firma en `manager.py`, sin cablear el origen |
| 9 | Escribir el test con `@pytest.mark.xfail` y commitearlo |
| 10 | Solo exportar el CSV del fixture a `tests/fixtures/` y commitearlo, sin loader |
| 11 | Crear `export.py` con la firma y la lista de columnas como constante |
| 12 | Un solo assert: el header del CSV es exactamente el esperado |
| 13 | Registrar el grupo `backtest` vacío en el CLI (sin el subcomando) |
| 14 | Cablear solo la carga del fixture; el backtest queda con `TODO` |
| 15 | Escribir el CSV con datos hardcodeados para validar la ruta de `--out` |
| 16 | Imprimir solo `Trades: N` |
| 17 | Test que solo verifica exit code 0 |
| 18 | Pegar el comando y su salida en el README, sin explicar las columnas |
| 19 | Agregar el `st.download_button` con datos fijos para verificar que renderiza |
| 20 | Solo `bd close QuantAgent-gg6` |
| 21 | Crear un ticket |
| 22 | Solo el merge |

---

## 4. Checklist de tracking

Una marca por día. Sí = el criterio de "hecho" de la fila se cumplió. No = no se cumplió.
La versión de 10 minutos **no** cuenta como Sí; se marca No y se recupera en D21/D22.

```
D01  15/09 mar  [x]
D02  16/09 mié  [x]
D03  17/09 jue  [x]
D04  18/09 vie  [x]
D05  21/09 lun  [x]
D06  22/09 mar  [x]
D07  23/09 mié  [x]
D08  24/09 jue  [x]
D09  25/09 vie  [x]
D10  28/09 lun  [x]
D11  29/09 mar  [x]
D12  30/09 mié  [x]
D13  01/10 jue  [ ]
D14  02/10 vie  [ ]
D15  05/10 lun  [ ]
D16  06/10 mar  [ ]
D17  07/10 mié  [ ]
D18  08/10 jue  [ ]
D19  09/10 vie  [ ]
D20  12/10 lun  [ ]
D21  13/10 mar  [ ]
D22  14/10 mié  [ ]
```

Total de Sí: ____ / 22
Entregable demostrable desde `main` al 14/10: [ ] Sí  [ ] No
