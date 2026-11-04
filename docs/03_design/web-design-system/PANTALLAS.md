# Pantallas: composición y datos

Referencia visual: canvas, páginas **Cuenta**, **Embudo v0.4**, **Ficha de hipótesis**, **Mercado**, **Inicio** y **Rendimiento** (teléfono 390 px y escritorio 1440 px, claro y oscuro).
Endpoints sugeridos: todos `GET`, solo lectura. La app **nunca** habla con el broker: el backend lee Alpaca y expone esto.

Rutas: `/` (Inicio), `/cuenta/:accountId`, `/backtests`, `/embudo` (+ `/embudo/ciclo/:cycleId` en teléfono), `/hipotesis/:id`, `/mercado` (+ `/mercado/etf/:symbol` en teléfono), `/rendimiento/...`. Navegación principal: 5 destinos activos (Inicio, Cuenta, Backtests, Embudo, Mercado).

---

## Inicio `/`

| Orden | Sección | Componentes | Datos |
|---|---|---|---|
| 0 | Encabezado: "Inicio" + estado del mercado ("abierto, cierra 16:00 ET (17:00 ART)") + ajustes (teléfono) | — | `GET /api/market/clock` → `MarketClock` |
| 1 | Estado general | `HomeStatusBanner` | `GET /api/home/status` → `HomeStatus` (derivado de alarmas activas) |
| 2 | Desde tu última visita | `FeedList` | `GET /api/feed?since=` → `FeedEvent[]` (ver EVENTOS.md) |
| 3 | Cuentas | `AccountCard` × n (variante `no_data` si corresponde) | `GET /api/accounts` → `AccountSummary[]` |
| 4 | Embudo | `CycleSummaryCard` compacta (sin `ReviewCounter`: la regla de revisión es de proceso y no entra en el estado general; al llegar a 6 aparece solo como evento informativo `review_rule_reached` en el feed, una vez por cierre) | `GET /api/funnel/cycles/latest` → `CycleSummary` |
| 5 | Próximos eventos | `EventList` | `GET /api/calendar?days=14` → `CalendarEvent[]` |

Escritorio: banner a todo el ancho → 3 cuentas en grilla → fila de 3 (feed, embudo, eventos).
**Sin total general** de cuentas (no mezclar paper con live).
Una cuenta `no_data` aparece como tarjeta punteada "Sin datos todavía" y **no** cuenta como problema en el estado general ni genera eventos en el feed.

---

## Cuenta `/cuenta/:accountId`

| Orden (teléfono) | Sección | Componentes | Datos |
|---|---|---|---|
| 0 | `AccountSelect` + "Actualizado 15:32 ART · hace 4 min" | | `GET /api/accounts` |
| 1 | Resumen | `AccountSummaryCard` | `GET /api/accounts/:id` → `AccountSummary` |
| 2 | Alarmas activas | `Notice` × n (hoy: hueco "Próximamente") | `AccountSummary.alarms` |
| 3 | Posiciones (largas y cortas mezcladas) | línea de términos con `HelpTip` + `PositionCard` × n con `Badge long\|short` | `GET /api/accounts/:id/positions` → `Position[]` |
| 4 | Órdenes | `DateStepper` + `ChipGroup` (Todas, Rechazadas, Pendientes, Ejecutadas, Canceladas, Vencidas) + `OrderCard` agrupadas | `GET /api/accounts/:id/orders?date=YYYY-MM-DD` → `Order[]` |
| 5 | Curva de valor · próximamente | `EquityChart` + `SegmentedControl` rango | `GET /api/accounts/:id/equity?range=1W\|1M\|3M\|YTD\|ALL` → `EquityPoint[]` |
| 6 | Exposición por sector · próximamente | `DivergingBar` | `GET /api/accounts/:id/exposure` → `SectorExposure[]` |
| 7 | Dividendos · próximamente | `MetricCard` × 3 + lista | `GET /api/accounts/:id/dividends?month=YYYY-MM` → `Dividend[]` |

Escritorio: resumen (360 px) + curva lado a lado → tabla de posiciones (con columna Lado) → órdenes (tabla) + columna derecha (exposición, dividendos).

Resumen: valor, hoy, semana; pie con Largo · Corto (si hay cortas) · Efectivo · Órdenes de hoy.

Estados diseñados: cargando, error API, cuenta vacía (tiene datos, sin actividad), desactualizado + alarma (tablero "Cuenta · Estados") y **cuenta sin datos** (v0.2, tableros 3: `AccountNoData` reemplaza toda la pantalla; nada de ceros).

Filtro de órdenes: chips Todas / Rechazadas / Pendientes / Ejecutadas / Canceladas / Vencidas, con contador. Las rechazadas van arriba incluso con "Todas". Las vencidas explican el motivo (orden del día al cierre) en la misma tarjeta.

Posiciones cortas: badge "Corta", barra de stop que crece desde la derecha (stop arriba) y, en teléfono, una línea "Gana si el precio baja…" / "Pierde si el precio sube…".

---

## Embudo `/embudo` (v0.4, sobre el registro real de hipótesis)

| Orden (teléfono) | Sección | Componentes | Datos |
|---|---|---|---|
| 0 | Filtros: ciclo (Todos / cada `YYYY-MM` / Sin ciclo) + período (30d / 90d / Todo), en la URL | `CycleFilter` + `SegmentedControl` | `?ciclo=all\|2026-10\|none&periodo=30d\|90d\|all` |
| 1 | Regla de revisión | `ReviewCounter` (+ `Notice strong` arriba de todo si llega a 6) | `GET /api/funnel/review` → `ReviewRule` |
| 2 | Ciclo actual + 3 anteriores | `CycleSummaryCard` (con `CycleMilestones`) + tira de ciclos anteriores (tocar = filtrar) | `GET /api/funnel/cycles/current` → `CycleSummary`; `GET /api/funnel/cycles?limit=4` |
| 3 | Hipótesis por etapa | `FunnelChart` E0–E3 + fase posterior atenuada; vista Tabla | `GET /api/funnel?ciclo=&periodo=` → `Funnel` |
| 4 | Por qué cayeron | `DropReasons` (acordeón) + bloque “Detenidas por error o alarma” + “Ver las N descartadas” | `Funnel.drop_reasons` |
| 5 | Hipótesis | `ChipGroup` (Activas · Descartadas · Todas · E0 · E1 · E2 · E3 · Fase posterior) + `HypothesisCard` × 8 + “Ver N más” | `GET /api/hypotheses?ciclo=&periodo=&estado=&etapa=` → `Hypothesis[]` |
| 6 | Recibo del ciclo | `CycleReceipt` (pantalla propia en teléfono; panel lateral en escritorio) | `GET /api/funnel/cycles/:id/receipt` → `CycleReceipt` |

Escritorio: encabezado con filtros a la derecha → fila `ReviewCounter` (1/3) + ciclo actual (2/3, con anteriores y botón de recibo a la derecha) → fila embudo (1,35) + motivos en tarjetas abiertas de a 2 (1) → hipótesis en grilla de 3 tarjetas (hasta `funnelConfig.cardsUpTo` = 30; con más, `DataTable` Etapa · Hipótesis · Ciclo · Veredicto · Intentos).

Reglas:
- **Todos** = todos los ciclos mensuales. El historial importado solo aparece con **Sin ciclo**: son **55 eventos de etapa** E0–E3 (no hipótesis) y **202 intentos** con la etapa en texto libre. Se muestra una tabla de eventos por etapa y veredicto (sin embudo) y los intentos por su etiqueta tal cual, sin asumir E0–E3. Sin `ReviewCounter` ni recibo.
- **Período 30d / 90d** = hipótesis con algún evento de etapa en los últimos N días; el estado mostrado es el de su último evento dentro de la ventana. **Aplica solo al embudo y a la lista** (rótulo “Período: aplica al embudo y a la lista”). La tarjeta del ciclo actual muestra siempre todas sus hipótesis (el ciclo es su propia ventana) y la regla de revisión no cambia. Con un ciclo concreto elegido, el período se deshabilita.
- **ERROR y ALARMA** nunca suman a “cayeron”: segmento propio en el embudo, línea “Detenidas aparte” y bloque propio en los motivos.
- **Promovida** = pasó E3 con PASA; es lo único que cuenta para la regla de revisión.
- Ciclo con 1 hipótesis (`sample_too_small`): no se dibuja el embudo, se muestra su recorrido con aviso de muestra mínima.
- Solo lectura. Los comandos se mencionan en texto, nunca con botón: `quantagent registro ciclo abrir 2026-11 H1 H2`, `quantagent registro ciclo cerrar 2026-11`, `quantagent registro ciclo informe 2026-11`, `quantagent registro corte`.

Estados diseñados (tablero “Estados”): cargando · sin ciclos todavía (“el primer ciclo se abre el día 1”) · ciclo abierto sin movimiento · error por sección con reintentar · desactualizado · Sin ciclo (historial). Variantes: ciclo abierto vs cerrado · revisión alcanzada (6 de 6) · tarjeta con y sin idea/última métrica · descartada por CORTA, ERROR y ALARMA · ciclo con una sola hipótesis · tabla para más de 30.

---

## Ficha de hipótesis `/hipotesis/:id` (v0.4, mínima)

| Orden | Sección | Componentes | Datos |
|---|---|---|---|
| 0 | “‹ Embudo” (teléfono) / ruta (escritorio) | — | — |
| 1 | Encabezado: nombre, ciclo (mes + id), etapa y veredicto, chip “Usó la reserva 2023+” si corresponde, idea si existe | `StageVerdictBadge`, `Badge` | `GET /api/hypotheses/:id` → `HypothesisDetail` |
| 2 | Recorrido por etapa (veredicto, fecha, motivo, valor vs umbral; “no llegó”; fase posterior atenuada) | `StageTimeline` | `HypothesisDetail.timeline`, `later_phases` |
| 3 | Intentos (fecha, activo o cartera, snapshot, Sharpe, operaciones, PnL, etapa; inválidos atenuados) | `AttemptList` | `HypothesisDetail.attempts` |
| 4 | Backtests vinculados → `/rendimiento/backtest/:id`; vacío: “Sin backtests vinculados todavía” | lista de enlaces | `HypothesisDetail.backtests` |
| 5 | Si corre en paper: “Corre en {cuenta} desde {fecha} · Ver cuenta ›” | `Card` | `HypothesisDetail.paper_account` |

Escritorio: encabezado (máx. 860 px) → recorrido + backtests + cuenta a la izquierda (1) e intentos en tabla a la derecha (1,5). Nada más (el rendimiento por estrategia sigue detrás de `featureFlags.runPerformance`).

---

## Backtests `/backtests` (v0.3.1, sección propia)

Destino propio en la navegación (no depende de Embudo: hay backtests sueltos). Lista con `DataTable`: Nombre · Hipótesis (o “suelto”) · Período · Retorno acumulado; búsqueda; casillas con máximo 2 → “Comparar 2 ›”. Tocar una fila abre su Rendimiento. Datos: `GET /api/backtests` → `BacktestListItem[]`. Primer corte: lista mínima (fragmento en la hoja “Rendimiento · Qué cambia por sujeto”). Búsqueda, filtros y estados, después, cuando haya backtests suficientes para necesitarlos.

---

## Rendimiento `/cuenta/:id/rendimiento` · `/rendimiento/run/:id` · `/rendimiento/backtest/:id` (v0.3)

Especificación completa en **RENDIMIENTO.md**. Resumen:

| Orden | Bloque | Componentes | Datos |
|---|---|---|---|
| 0 | Volver / migas | enlace “‹ origen” | origen de navegación |
| 1 | Encabezado + Período + Comparar con | `SubjectHeader`, `CompareSelector` | `PerformanceReport.{subject,period,as_of,headline,available_ranges}` |
| 2 | Aviso de muestra corta | `ShortSampleBanner` | `sample` |
| 3 | Métricas (Rendimiento · Riesgo · Contra SPY) | `MetricGroup` × 3 | `metrics`, `benchmark_metrics` |
| 4 | Curva de valor · Retorno acumulado · Caídas · Mapa mensual · Retornos por año/mes · Distribución · Métricas móviles | `ChartCard` + `EquityChart`, `CumulativeChart`, `DrawdownChart`, `MonthlyHeatmap`, `ReturnsBars`, `ReturnsHistogram`, `RollingChart` | `series`, `monthly_returns`, `annual_returns`, `distribution`, `rolling` |
| 5 | 5 peores caídas · Retorno por año contra SPY | `DataTable` | `worst_drawdowns`, `annual_returns` |
| 6 | Convenciones | `CalcConventions` | `conventions` |

Endpoint: `GET /api/performance/{type}/{id}?range=&align_with=` → `PerformanceReport`.

**Flags (v0.3.2)**: la pestaña Rendimiento de Cuenta está oculta (`featureFlags.accountPerformanceTab = false`) hasta que el backend registre aportes y calcule el retorno ponderado por tiempo; con el flag apagado, Cuenta queda como antes, sin pestañas. El Rendimiento de una estrategia en ejecución también está oculto (`featureFlags.runPerformance = false`) hasta que haya capital por estrategia. **Primer corte: solo backtests.**

**Cambio en Cuenta**: arriba de la pantalla, pestañas “Hoy · Rendimiento”. “Hoy” es la pantalla Cuenta actual; “Rendimiento” es este informe con la cuenta como sujeto. La sección “Curva de valor · Próximamente” de “Hoy” pasa a ser un resumen de una línea con enlace a la pestaña Rendimiento.

---

## Mercado `/mercado` (v0.4)

Contexto, no señales: sin recomendaciones, sin precios en vivo. Todo es del cierre de la sesión anterior, con snapshots propios. Siempre: “Al cierre del {fecha} · actualizado {hora ART}”.

| Orden (teléfono) | Sección | Componentes | Datos |
|---|---|---|---|
| 1 | Encabezado: “Mercado”, `MarketClock`, fecha del último dato, aviso si está atrasado | `MarketAsOfLine`, `Notice stale` | `GET /api/market` → `MarketPage.as_of`, `clock` |
| 2 | Régimen de SPY (2 ejes, 4 regímenes, desde cuándo, tira de 12 meses) | `RegimeCard` + `RegimeStrip` | `MarketPage.regime` |
| 2b | Estado del motor (risk-on / neutral / risk-off, cuándo se calculó, `state_description` del backend) | `MotorStateCard` | `MarketPage.motor` (`null` = estado vacío) |
| 3 | Sectores: 11 ETF + SPY, EEM, EWZ de referencia; 1d/1s/1m/3m/Año, score y ranking de los 11 (motor de M5), habilitados (top 3 con score positivo) y bloqueados con motivo; “En cartera · {cuenta}” con enlace según posiciones reales de M4 y M5; habilitado sin posición todavía | `SectorTable` / `SectorHeatmap` (teléfono: tarjetas + “Ordenar por” + Lista/Mapa) | `MarketPage.sectors` |
| 4 | Macro: T10Y2Y (1A / 5A / Todo, valor, cambio, recesiones, explicación) + grilla de indicadores “próximamente” | `MacroLineChart` (con `recessions`), `MacroIndicatorCard` × 9 | `MarketPage.macro`, `recessions` |
| 5 | 29 ETF del universo de las carteras M4 (no el de M5) agrupados por clase (Acciones EE. UU. 3 · Sectores 9 · Internacional 5 · Bonos 6 · Materias primas 4 · Moneda 1 · Inmobiliario 1; sin efectivo); casillas (máx. 5) → comparación; símbolo → velas | `EtfTable`, `CompareChart`, `EtfDetail` | `MarketPage.etfs`; `GET /api/market/series?symbols=&range=` → `{symbol, points}[]`; `GET /api/market/etf/:symbol/candles?range=` → `DailyCandle[]` |
| 6 | Próximos eventos (macro y feriados, 14 días) + enlace a Inicio | `EventList` filtrado | `MarketPage.events` |

Escritorio: encabezado a todo el ancho → régimen (1,45) + motor (1) → sectores con conmutador Tabla / Mapa de calor (1,9) + macro (1) → ETF a todo el ancho: tabla (2,2) + comparación fija al costado (1) → eventos. Teléfono: una columna, tablas como tarjetas, tooltip fijado al tocar, gráficos apilados. Detalle de ETF: pantalla en teléfono, panel lateral de 560 px en escritorio.

Estados diseñados: cargando por sección · sin snapshots todavía · desactualizado (sesiones faltantes) · el motor no calculó el score semanal · serie macro vencida (fecha esperada de FRED + 2 días hábiles; sin calendario, 1,5 × frecuencia) · ETF sin historia suficiente (“sin datos”, nunca un retorno inventado) · error por sección con reintentar.

Fuera de alcance: botones de escritura, recomendaciones, precios en vivo, noticias, países fuera de EEM y EWZ.

---

## Comportamiento común

- **Refresco**: polling (`config.ts → dataConfig`). Sin websockets en la primera versión.
- **Dato viejo**: si `as_of` supera el umbral → `Notice stale` arriba de la sección y la cifra en text-2.
- **Errores por sección**: si falla un endpoint, solo esa sección muestra `ErrorState`; el resto sigue.
- **Rutas con estado en la URL**: filtros, fecha y rango van en query params (compartibles y sobreviven al recargar).
- **Preferencias por dispositivo** (tema, densidad, último filtro): `localStorage` con try/catch.
- **Datos de ejemplo**: para desarrollo, un modo `?mock=1` o MSW con los mismos números del canvas (cuadran entre pantallas).
