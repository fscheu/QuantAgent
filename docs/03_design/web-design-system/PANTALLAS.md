# Pantallas: composición y datos

Referencia visual: canvas, páginas **Cuenta**, **Embudo** e **Inicio** (teléfono 390 px y escritorio 1440 px, claro y oscuro).
Endpoints sugeridos: todos `GET`, solo lectura. La app **nunca** habla con el broker: el backend lee Alpaca y expone esto.

Rutas sugeridas: `/` (Inicio), `/cuenta/:accountId`, `/embudo`, `/hipotesis/:id` (fase posterior), `/mercado` (fase posterior).

---

## Inicio `/`

| Orden | Sección | Componentes | Datos |
|---|---|---|---|
| 0 | Encabezado: "Inicio" + estado del mercado ("abierto, cierra 16:00 ET (17:00 ART)") + ajustes (teléfono) | — | `GET /api/market/clock` → `MarketClock` |
| 1 | Estado general | `HomeStatusBanner` | `GET /api/home/status` → `HomeStatus` (derivado de alarmas activas) |
| 2 | Desde tu última visita | `FeedList` | `GET /api/feed?since=` → `FeedEvent[]` (ver EVENTOS.md) |
| 3 | Cuentas | `AccountCard` × n (variante `no_data` si corresponde) | `GET /api/accounts` → `AccountSummary[]` |
| 4 | Embudo | `CycleSummaryCard` compacta | `GET /api/funnel/cycles/latest` → `CycleSummary` |
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

## Embudo `/embudo`

| Orden | Sección | Componentes | Datos |
|---|---|---|---|
| 0 | Filtros: ciclo (todos / N) + período (30 días, 90 días, Todo) | `AccountSelect`-like + `SegmentedControl` | query params `?cycle=&period=` |
| 1 | Último ciclo | `CycleSummaryCard` | `GET /api/funnel/cycles/latest` + `GET /api/funnel/cycles?limit=3` |
| 2 | Hipótesis por etapa | `FunnelChart` | `GET /api/funnel?cycle=&period=` → `Funnel` |
| 3 | Por qué cayeron | `DropReasons` | `Funnel.drop_reasons` |
| 4 | Hipótesis | `ChipGroup` (Activas / Descartadas / Todas / Etapa) + `HypothesisCard` × n | `GET /api/hypotheses?status=&stage=&cycle=` → `Hypothesis[]` |

Escritorio: embudo + último ciclo lado a lado → motivos en grilla de 3 → hipótesis en grilla de 3 (cuando haya muchas, la lista pasa a `DataTable` con columnas Nombre · Etapa · Métrica · Fecha; ver DECISIONES.md #10).
Orden de hipótesis: etapa más avanzada primero, luego fecha descendente.

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

## Mercado (fase posterior) — lo que ya está definido

El patrón "comparar activos" usa `DataTable` con casillas (máximo 5 tildadas), muestra de serie junto al símbolo y orden por columna; las filas tildadas alimentan `CumulativeChart` con el mismo color y trazo. En teléfono, tarjetas con casilla y selector "Ordenar por".

---

## Comportamiento común

- **Refresco**: polling (`config.ts → dataConfig`). Sin websockets en la primera versión.
- **Dato viejo**: si `as_of` supera el umbral → `Notice stale` arriba de la sección y la cifra en text-2.
- **Errores por sección**: si falla un endpoint, solo esa sección muestra `ErrorState`; el resto sigue.
- **Rutas con estado en la URL**: filtros, fecha y rango van en query params (compartibles y sobreviven al recargar).
- **Preferencias por dispositivo** (tema, densidad, último filtro): `localStorage` con try/catch.
- **Datos de ejemplo**: para desarrollo, un modo `?mock=1` o MSW con los mismos números del canvas (cuadran entre pantallas).
