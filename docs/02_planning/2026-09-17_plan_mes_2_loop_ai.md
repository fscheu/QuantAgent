# Plan mes 2 (2026-09-21 → 2026-10-16): loop AI + auditoría de métricas + 3 estrategias

Generado: 2026-09-17 · Label BEADS: `plan-20260921` · Modo: **IA implementa, Fede revisa ≤20 min/día**
Antecedentes: `2026-09-16_retrospectiva_autodev_y_plan_oficina_inversion.md` (loop y capas), `PLAN-30-DIAS.md` (semana 1)
Borradores asociados: `2026-09-17_borrador_routine_nocturna.md`, `2026-09-17_borrador_skill_revision_macro_semanal.md`

---

## 1. Qué entrega el mes

Al 2026-10-16, desde `main` limpio:

1. **Loop nocturno funcionando**: cada mañana hábil hay como máximo un PR con comando de verificación y salida real; CI corre en el PR; Fede lo revisa en 20 min.
2. **Métricas auditadas**: PnL, win rate / profit factor, max drawdown y Sharpe documentados en `docs/03_design/backtest_metrics.md`, con tests a mano y validación de Fede en planilla contra los CSV de `spy-90d`/RSI.
3. **Tres estrategias corriendo en el CLI** con fixtures versionados y ≥5 trades cada una (RSI, 52-week-high, Triple Screen), más `backtest verify` (reproducibilidad) y test golden.
4. **Decisión de engine** (propio vs `backtesting.py`) tomada por Fede en un ADR con la evidencia de la cross-validation.
5. **Primera nota de revisión macro semanal** en el vault (capa L1), sin código.

Fuera de alcance: Alpaca / M2, `u0w` (APScheduler), UI Streamlit, señales alternativas (L6), `kkj.10`/`kkj.11`.

## 2. Reglas del mes

- Un ticket `ai-ready` por corrida nocturna. Un PR por día. Nunca a `main` directo.
- Cada ticket tiene sección **Verificación de Fede (≤10 min)**: es lo único que Fede hace además de leer el PR.
- `fede-decision` = el loop no lo toma; es una acción de Fede con un borrador ya escrito.
- Si un día no hay revisión, el loop sigue hasta 3 PRs abiertos y después espera.
- Lo que aparezca fuera de la lista se abre como ticket y no se persigue.

## 3. Calendario por semana

Orden de cola = `bd ready --label ai-ready` (prioridad, luego antigüedad). Las fechas son la noche en que el loop lo toma; Fede revisa a la mañana siguiente.

### Semana 2 (21–25/09) — sanear la base y montar el loop

| Noche | Ticket | Entrega | Fede verifica (mañana) |
|---|---|---|---|
| Fede, manual | `QuantAgent-5z0` | Crear la routine desde el borrador; primera corrida "run now" sobre `3km` | 30 min una vez |
| lun 21 | `QuantAgent-3km` | Guardrail 730 días en el provider Yahoo (ticket de calentamiento del loop) | Leer el PR, ver test y CI verde |
| mar 22 | `QuantAgent-83e` | Trade log con fechas de vela, stdout sin spam de rechazos | Abrir el CSV: fechas de ene–mar 2026 |
| mié 23 | `QuantAgent-zui` | CI en `pull_request` hacia `main` | Ver el check en el PR |
| jue 24 | `QuantAgent-hak` | PR template + `docs/workflows/ai-dev-loop.md` + labels | Abrir "New PR" y ver el template |
| vie 25 | `QuantAgent-89e` | Filas duplicadas en reversión; `Trades: N` = filas CSV | Contar filas del CSV vs `Trades: N` |

Acción de Fede en la semana (5 min): cerrar o reescribir `QuantAgent-kkj.11` (in_progress desde mayo, bloquea `kkj.10`). Recomendación: cerrar con "superseded", reabrir cuando haya estrategias LLM en el plan.

### Semana 3 (28/09–02/10) — auditoría de métricas, una por noche

| Noche | Ticket | Entrega | Fede verifica |
|---|---|---|---|
| lun 28 | `QuantAgent-cv1` | Prune de worktrees y ramas muertas; tabla de ramas no mergeadas | `git worktree list` = 1 línea; decidir sobre la tabla |
| mar 29 | `QuantAgent-11v` | `--equity-out eq.csv` | Graficar `equity` en planilla; min de `drawdown_pct` = stdout |
| mié 30 | `QuantAgent-hx0.1` | PnL: fórmula, tests a mano, `docs/03_design/backtest_metrics.md` | `=SUMA(pnl)` = PnL impreso |
| jue 01 | `QuantAgent-hx0.2` | Win rate / profit factor; política pnl = 0; sin `inf` en DB | `CONTAR.SI(pnl>0)/CONTARA` = win rate |
| vie 02 | `QuantAgent-hx0.3` | Max drawdown; equity curve documentada | Recalcular DD en planilla sobre `eq.csv` |

### Semana 4 (05–09/10) — Sharpe, reproducibilidad y las otras dos estrategias

| Noche | Ticket | Entrega | Fede verifica |
|---|---|---|---|
| lun 05 | `QuantAgent-hx0.4` | Sharpe con anualización por calendario efectivo | Recalcular en planilla, diferencia < 0.01 |
| mar 06 | `QuantAgent-y8z` | `backtest verify` (doble corrida + diff) | Correrlo: "OK reproducible", exit 0 |
| mié 07 | `QuantAgent-46h` | Fixture diario 2 años + 52-week-high ≥5 trades | Elegir un trade y ver la ruptura en el fixture |
| jue 08 | `QuantAgent-8u8` | Fixture 4h + Triple Screen ≥5 trades | Ídem con la ruptura de la vela previa |
| vie 09 | `QuantAgent-piv` | Test golden RSI/spy-90d con los valores validados | Confirmar que los números son los de su planilla |

Al cerrar `hx0.4`, `QuantAgent-hx0` (epic) se cierra y desbloquea `piv` y `832`.

### Semana 5 (12–16/10) — decisión de engine y capa macro

| Noche | Ticket | Entrega | Fede verifica |
|---|---|---|---|
| lun 12 | `QuantAgent-832` | Cross-validation RSI vs `backtesting.py` sobre `spy-90d`; doc con diffs | Leer la tabla de diffs (1 página) |
| mar 13 | `QuantAgent-fiu` | Reescribir `test_parallel_execution` con fixture existente (relleno) | `pytest tests/test_parallel_execution.py` sin skip |
| mié 14 | `QuantAgent-lcv` (`fede-decision`) | ADR engine propio vs terceros, redactado por el agente | **Fede decide** en una línea |
| dom 11 / dom 18 | `QuantAgent-wwi` (`fede-decision`) | Instalar skill macro, primera nota, cron dominical | Leer la nota del lunes, responder una pregunta |
| jue 15–vie 16 | `QuantAgent-lp1` (`fede-decision`) | Plan del mes 3 con la evidencia del mes 2 | Aprobar o recortar (15 min) |

Colchón: si un ticket se bloquea, el loop toma el siguiente `ai-ready` y el bloqueado se discute en la revisión matutina. Hay 3 tickets de relleno (`fiu`, `cv1`, `3km`) que pueden moverse sin afectar la cadena crítica `83e → 89e → 11v → hx0.1 → hx0.2 → hx0.3 → hx0.4 → piv/832 → lcv`.

## 4. Grafo de dependencias (cadena crítica)

```
83e (timestamps) ─┬─> 11v (equity csv) ──> hx0.3 (drawdown) ──> hx0.4 (sharpe) ─┐
                  ├─> hx0.1 (pnl) ──────> hx0.2 (win rate)                      ├─> hx0 cierra ─┬─> piv (golden)
89e (dup rows) ───┘                                                               │              └─> 832 (cross-val) ──> lcv (ADR) ──> lp1 (mes 3)
83e ──> y8z (verify) ; 83e ──> 46h (52w) ; 83e ──> 8u8 (triple screen)
zui (CI PR) + hak (template) ──> 5z0 (routine)
```

## 5. Checklist de tracking (una marca por noche)

```
S2  lun21 3km [ ]  mar22 83e [ ]  mié23 zui [ ]  jue24 hak [ ]  vie25 89e [ ]   | Fede: 5z0 routine [ ]  kkj.11 [ ]
S3  lun28 cv1 [ ]  mar29 11v [ ]  mié30 hx0.1 [ ]  jue01 hx0.2 [ ]  vie02 hx0.3 [ ]
S4  lun05 hx0.4 [ ]  mar06 y8z [ ]  mié07 46h [ ]  jue08 8u8 [ ]  vie09 piv [ ]
S5  lun12 832 [ ]  mar13 fiu [ ]  mié14 lcv [ ]  dom wwi [ ]  jue15-vie16 lp1 [ ]
```

Mergeados a `main` al 16/10: ____ / 18 · Métricas validadas en planilla: ____ / 4 · Estrategias con ≥5 trades: ____ / 3

## 6. Cómo retomar si se corta el hilo

1. `bd ready --label ai-ready` dice qué sigue. `bd list --label plan-20260921` dice el estado del mes.
2. `gh pr list` dice qué está esperando revisión.
3. Este documento §5 dice dónde íbamos. No reconstruir desde commits.
