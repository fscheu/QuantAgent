# Embudo, Ficha de hipótesis y Mercado — contrato de datos (v0.4)

Canvas: páginas **Embudo v0.4**, **Ficha de hipótesis**, **Mercado** y **v0.4 · Componentes**. Tipos en `api-types.ts`; parámetros en `config.ts` (`funnelConfig`, `cycleConfig`, `marketConfig`).

Regla general: un campo `null` tiene un estado diseñado. **Nunca** se muestra 0, string vacío ni un guion sin explicar: se dice “sin datos” y el motivo, o el campo no se muestra (cuando el diseño así lo indica).

---

## 1. Cambios de contrato (v0.3.2 → v0.4)

| Tipo | Antes | Ahora | Por qué |
|---|---|---|---|
| `Funnel.cycle` | `number \| 'all'` | `CycleId \| 'all'` (`'2026-11'`); “Sin ciclo” usa `HistorySummary` | Los ciclos son mensuales con id `YYYY-MM`; el historial son eventos, no hipótesis |
| `Funnel.period_days` | `number \| null` | `period: '30d' \| '90d' \| 'all'` | Coincide con el filtro de la URL |
| `FunnelStage` | `{stage, reached, here, dropped}` | `StageCounts {stage, reached, here, passed, cut, error, alarm}` | ERROR y ALARMA se muestran aparte |
| `StageId` | incluía `reserve` y `production` | `CycleStageId` (e0–e3) \| `LaterPhaseId` (shadow, paper, candidate) | La reserva no es etapa; producción fuera de alcance |
| `CycleSummary.cycle` | `number` | `CycleId` | Ídem |
| `CycleSummary` | `active`, `dropped` | `status`, `closed_at`, `active`, `cut`, `held`, `promoted`, `milestones`, `receipt_id`; `reached_by_stage` con `null` = pendiente | Ciclo abierto/cerrado, hitos y recibo |
| `Hypothesis.idea` | `string` | `string \| null` | Nula en la mayoría |
| `Hypothesis.cycle` | `number` | `CycleId \| null` | Historial |
| `Hypothesis.status` | `'active' \| 'discarded'` | `'active' \| 'promoted' \| 'discarded' \| 'held'` + `verdict: Verdict \| null` | Veredictos reales PASA/CORTA/ERROR/ALARMA |
| `Hypothesis.last_metric` | `{name, value, threshold} \| null` | `{name, value, at} \| null` | Nula en la mayoría; el umbral va en `discard_measure` |
| `Hypothesis.discard_detail` | `string` (texto armado) | `discard_measure: Measure \| null` | El front formatea “Sharpe 0,31 · umbral 0,50” en es-AR |
| nuevos | — | `ReviewRule`, `CycleReceipt`, `Snapshot`, `Attempt`, `StageResult`, `HypothesisDetail`, `HistorySummary`, `Measure` | |
| nuevos (Mercado) | — | `MarketPage`, `MarketAsOf`, `Regime`, `MotorState`, `SectorRow`, `MacroSeries`, `RecessionPeriod`, `EtfRow`, `DailyCandle`, `Returns` | |

---

## 2. Embudo — lista de datos por elemento

### Filtros
| Campo | Tipo | Si es null / falta |
|---|---|---|
| `ciclo` (query) | `'all' \| CycleId \| 'none'` | Sin parámetro → `funnelConfig.defaultCycle` |
| `periodo` (query) | `'30d' \| '90d' \| 'all'` | Sin parámetro → `funnelConfig.defaultPeriod` |
| lista de ciclos | `{id, status}[]` | Vacía → estado “Todavía no hay ciclos” |

### `ReviewCounter` ← `ReviewRule`
| Campo | Tipo | Si es null / falta |
|---|---|---|
| `limit` | `number` | Usa `cycleConfig.reviewAfterCycles` (6) |
| `streak` | `number` | — (siempre viene; 0 es un valor real cuando hay ciclos cerrados) |
| `reached` | `boolean` | Se recalcula `streak >= limit` |
| `last_cycles[]` | `{cycle, promoted}` | Vacío → “Sin datos: todavía no hay ciclos cerrados para contar” (no “0 de 6”) |

### `CycleSummaryCard` ← `CycleSummary`
| Campo | Tipo | Si es null / falta |
|---|---|---|
| `cycle` | `CycleId` | — |
| `status` | `'open' \| 'closed'` | — |
| `started_at` / `closed_at` | ISO / ISO \| null | `closed_at` null = abierto |
| `entered` | `number` | 0 en ciclo recién abierto → texto “Todavía no entró ninguna hipótesis”; Siguen/Cayeron/Promov. muestran “—” con esa frase |
| `active`, `cut`, `held`, `promoted` | `number` | — |
| `reached_by_stage[e]` | `number \| null` | null → “pendiente · arranca {fecha}” |
| `milestones[]` | `CycleMilestone` | Falta → el front los calcula con `cycleConfig.milestones` |
| `headline` | `string \| null` | null → frase armada con los conteos |
| `receipt_id` | `string \| null` | null → botón “Ver informe parcial” |

### `FunnelChart` ← `Funnel`
| Campo | Tipo | Si es null / falta |
|---|---|---|
| `stages[]` | `StageCounts` × 4 | Siempre las 4; `reached = 0` → pista punteada “Todavía ninguna” |
| `later[]` | `LaterPhaseCounts` × 3 | `reached = 0` → “todavía ninguna” atenuado |
| `promoted` | `number` | 0 → no se muestra la línea “promovidas” |
| `sample_too_small` | `boolean` | true → recorrido de la única hipótesis + aviso de muestra mínima |

### `DropReasons` ← `Funnel.drop_reasons`
| Campo | Tipo | Si es null / falta |
|---|---|---|
| `stage`, `verdict`, `cause_code`, `cause_label`, `count` | | Sin CORTA en una etapa → “Ninguna cayó en esta etapa” |
| `example` | `Measure \| null` | null → no se muestra la línea “ej.: …” |

### `HypothesisCard` ← `Hypothesis`
| Campo | Tipo | Si es null / falta |
|---|---|---|
| `name` | `string` | — |
| `cycle` | `CycleId \| null` | null → “sin ciclo” (mono) |
| `stage` + `verdict` + `status` | | `verdict` null (fase posterior) → badge de fase |
| `idea` | `string \| null` | null → la línea **no se renderiza** (sin hueco, sin “sin idea”) |
| `attempts_valid` / `attempts_invalid` | `number` | `invalid = 0` → no se menciona |
| `last_metric` | `{name, value, at} \| null` | null → el pie muestra solo intentos |
| `discard_reason` + `discard_measure` | `string \| null`, `Measure \| null` | Activa o promovida → no hay bloque; `measure` null → solo el motivo |

### `CycleReceipt` ← `CycleReceipt`
| Campo | Tipo | Si es null / falta |
|---|---|---|
| `id`, `hash` | `string \| null` | null (parcial) → “parcial · se fija al cerrar” |
| `partial` | `boolean` | true → `Notice info` “Ciclo abierto, informe parcial” |
| `stages[].pending` | `boolean` | true → “pendiente · arranca {fecha}” en lugar de números |
| `drops[]` | | Vacío → “Ninguna cayó todavía” |
| `attempts[]` | `{name, valid, invalid}` | — |
| `snapshots[]` | `{name, as_of, hash}` | Vacío (parcial muy temprano) → “Todavía no se usaron datos” |

### `StageTimeline` / `AttemptList` / enlaces ← `HypothesisDetail`
| Campo | Tipo | Si es null / falta |
|---|---|---|
| `timeline[]` | `StageResult` | Etapas ausentes → “no llegó” (punto punteado) |
| `timeline[].measure` | `Measure \| null` | null → solo motivo y fecha |
| `later_phases[]` | | Vacío → no se muestra el separador “Fase posterior” |
| `used_reserve` | `boolean` | false → sin chip |
| `attempts[].sharpe/trades/pnl_pct` | `number \| null` | null → “sin datos” (el intento falló antes de calcular) |
| `attempts[].invalid` + `invalid_reason` | | true → fila atenuada + “no cuenta” + motivo |
| `backtests[]` | | Vacío → “Sin backtests vinculados todavía” |
| `paper_account` | `{id, name, since} \| null` | null → no se muestra el bloque |

### Sin ciclo ← `HistorySummary`
| Campo | Tipo | Si es null / falta |
|---|---|---|
| `stage_events`, `attempts` | `number` | No hay historial → la opción “Sin ciclo” no aparece en el filtro |
| `events_by_stage[]` | E0–E3 con veredictos | Etapa sin eventos → fila con ceros reales (hay datos) |
| `attempts_by_label[].label` | `string \| null` | Texto libre, se muestra tal cual; null → “(sin etiqueta)”. **Nunca** se mapea a E0–E3 |

---

## 3. Mercado — lista de datos por elemento

### Encabezado ← `MarketAsOf` + `MarketClock`
| Campo | Tipo | Si es null / falta |
|---|---|---|
| `session_date` | fecha | Sin snapshots → estado “Todavía no hay snapshots de mercado” en toda la pantalla (eventos siguen) |
| `updated_at` | ISO | — |
| `stale` + `missing_sessions[]` | | `stale` → `Notice warn` “Faltan N sesiones: …” y cifras en text-2 |

### `RegimeCard` / `RegimeStrip` ← `Regime`
| Campo | Tipo | Si es null / falta |
|---|---|---|
| `regime`, `since`, `sessions` | | `regime` null (todo) → card con “sin datos” y motivo |
| `ma200` | `number \| null` | null → “sin datos: hacen falta 200 sesiones para la media” |
| `vol20`, `vol_median_1y` | `number \| null` | null → eje de volatilidad “sin datos” y el régimen no se calcula |
| `strip[].regime` | `RegimeId \| null` | null → tramo punteado “sin datos” |

### `MotorStateCard` ← `MotorStateResponse`
| Campo | Tipo | Si es null / falta |
|---|---|---|
| (todo) | `MotorState \| null` | null → bloque punteado “el motor todavía no calculó el score semanal · se calcula los viernes” |
| `state_description` | `string \| null` | null → no se muestra el bloque; la escala de 3 estados (texto fijo) sigue |

### `SectorTable` / `SectorHeatmap` ← `SectorRow[]`
| Campo | Tipo | Si es null / falta |
|---|---|---|
| `returns.{d1…ytd}` | `number \| null` | null → “sin datos” (celda transparente en el mapa) |
| `momentum_score`, `rank` | `number \| null` | null → “score: sin datos”; orden por defecto pasa a 3m |
| `enabled`, `blocked`, `blocked_reason` | | `blocked_reason` null con `blocked` → “Bloqueado” sin motivo + HelpTip genérico |
| `held_in[]` | de posiciones reales (M4, M5) | Vacío → sin badge; si además `enabled` → “sin posición todavía” |
| `is_reference` | `boolean` | true → al final, sin ranking ni badges |

### `MacroLineChart` / `MacroIndicatorCard` ← `MacroSeries` + `RecessionPeriod[]`
| Campo | Tipo | Si es null / falta |
|---|---|---|
| `status` | `'active' \| 'coming_soon'` | `coming_soon` → tarjeta fantasma “próximamente” |
| `last` | `{value, observation_date, published_at} \| null` | null → “sin datos” |
| `change_1m` | `number \| null` | null → no se muestra el cambio |
| `stale` | `boolean` | true → `Notice stale` “Se esperaba un dato el {expected_release} y pasaron 2 días hábiles sin dato nuevo”; cifra atenuada |
| `expected_release` | `string \| null` | null (sin calendario) → la regla pasa a 1,5 × frecuencia; no se muestra la fecha esperada |
| `recessions[]` | | Ninguna en el rango → nota “No hubo recesiones en este período” |

### `EtfTable` / `CompareChart` / `EtfDetail` ← `EtfRow[]`, series, `DailyCandle[]`
| Campo | Tipo | Si es null / falta |
|---|---|---|
| `close` | `number \| null` | null → “sin datos” |
| `returns.*` | `number \| null` | null → “sin datos” + `history_sessions` en el HelpTip (“13 sesiones; 1m pide 21”) |
| `spark_30d` | `number[] \| null` | null → “N sesiones” en lugar de la línea |
| `held_in[]` | | Vacío → sin badge |
| series de comparación | `{symbol, points}[]` | Un símbolo sin puntos en el rango → se lista en la leyenda como “sin datos en este período” |
| velas | `DailyCandle[]` | < 2 → “sin datos suficientes para el gráfico” + tabla |

### `EventList` ← `CalendarEvent[]`
| Campo | Tipo | Si es null / falta |
|---|---|---|
| eventos de 14 días | | Vacío → “Sin eventos macro ni feriados en los próximos 14 días” |
| FOMC | | Sin reunión en 14 días → línea “próxima: {fechas}” (FOMC se carga a mano) |

---

## 4. Prerrequisitos y preguntas para el backend

1. **Veredictos y estado derivado**: el backend devuelve `verdict` y `status` (`held` para ERROR/ALARMA). El front no infiere.
2. **Motivo con valor vs umbral** como `Measure` numérico (no texto armado) para formatear en es-AR.
3. **Recibo**: se genera al cerrar por CLI y se guarda inmutable con su hash; el endpoint de un ciclo abierto devuelve el mismo formato con `partial: true`.
4. **Regla de revisión**: calculada en el backend sobre ciclos cerrados.
5. **Mercado**: endpoint agregado `GET /api/market` (una llamada por pantalla) + series y velas bajo demanda. Snapshots diarios propios después del cierre.
6. **Motor**: `state_description` del estado actual lo manda el backend (el canvas usa texto de ejemplo marcado). Definiciones fijas: risk-off = sin compras nuevas; neutral = estrés macro elevado sin que SPY esté en risk-off; risk-on = el resto.
8. **Historial**: 55 eventos de etapa E0–E3 marcados `imported`; 202 intentos con etapa en texto libre, que se devuelven por etiqueta.
10. **Sectores**: el motor de M5 rankea los 11; `held_in` sale de posiciones reales, no del ranking.
11. **Período**: `GET /api/funnel` y `/api/hypotheses` aplican `periodo`; `/api/funnel/cycles/current` y `/api/funnel/review` lo ignoran. Con `ciclo` concreto, `periodo` se ignora.
9. **Feed**: evento `review_rule_reached` al cerrar un ciclo con racha ≥ 6 (ver EVENTOS.md).
7. **FRED**: T10Y2Y + USREC (mensual; el front la usa como períodos). Las demás series quedan definidas pero apagadas.
