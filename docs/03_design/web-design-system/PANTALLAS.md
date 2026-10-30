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
| 3 | Cuentas | `AccountCard` × n | `GET /api/accounts` → `AccountSummary[]` |
| 4 | Embudo | `CycleSummaryCard` compacta | `GET /api/funnel/cycles/latest` → `CycleSummary` |
| 5 | Próximos eventos | `EventList` | `GET /api/calendar?days=14` → `CalendarEvent[]` |

Escritorio: banner a todo el ancho → 3 cuentas en grilla → fila de 3 (feed, embudo, eventos).
**Sin total general** de cuentas (no mezclar paper con live).

---

## Cuenta `/cuenta/:accountId`

| Orden (teléfono) | Sección | Componentes | Datos |
|---|---|---|---|
| 0 | `AccountSelect` + "Actualizado 15:32 ART · hace 4 min" | | `GET /api/accounts` |
| 1 | Resumen | `AccountSummaryCard` | `GET /api/accounts/:id` → `AccountSummary` |
| 2 | Alarmas activas | `Notice` × n (hoy: hueco "Próximamente") | `AccountSummary.alarms` |
| 3 | Posiciones | línea de términos con `HelpTip` + `PositionCard` × n | `GET /api/accounts/:id/positions` → `Position[]` |
| 4 | Órdenes | `DateStepper` + `ChipGroup` + `OrderCard` agrupadas | `GET /api/accounts/:id/orders?date=YYYY-MM-DD` → `Order[]` |
| 5 | Curva de valor · próximamente | `EquityChart` + `SegmentedControl` rango | `GET /api/accounts/:id/equity?range=1W\|1M\|3M\|YTD\|ALL` → `EquityPoint[]` |
| 6 | Exposición por sector · próximamente | `DivergingBar` | `GET /api/accounts/:id/exposure` → `SectorExposure[]` |
| 7 | Dividendos · próximamente | `MetricCard` × 3 + lista | `GET /api/accounts/:id/dividends?month=YYYY-MM` → `Dividend[]` |

Escritorio: resumen (360 px) + curva lado a lado → tabla de posiciones → órdenes (tabla) + columna derecha (exposición, dividendos).

Estados diseñados (tablero "Cuenta · Estados"): cargando, error API, cuenta vacía, desactualizado + alarma (Live, con orden rechazada por "Mercado cerrado").

Filtro de órdenes: chips Todas / Rechazadas / Pendientes / Ejecutadas / Canceladas, con contador. Las rechazadas van arriba incluso con "Todas".

---

## Embudo `/embudo`

| Orden | Sección | Componentes | Datos |
|---|---|---|---|
| 0 | Filtros: ciclo (todos / N) + período (30 días, 90 días, Todo) | `AccountSelect`-like + `SegmentedControl` | query params `?cycle=&period=` |
| 1 | Último ciclo | `CycleSummaryCard` | `GET /api/funnel/cycles/latest` + `GET /api/funnel/cycles?limit=3` |
| 2 | Hipótesis por etapa | `FunnelChart` | `GET /api/funnel?cycle=&period=` → `Funnel` |
| 3 | Por qué cayeron | `DropReasons` | `Funnel.drop_reasons` |
| 4 | Hipótesis | `ChipGroup` (Activas / Descartadas / Todas / Etapa) + `HypothesisCard` × n | `GET /api/hypotheses?status=&stage=&cycle=` → `Hypothesis[]` |

Escritorio: embudo + último ciclo lado a lado → motivos en grilla de 3 → hipótesis en grilla de 3.
Orden de hipótesis: etapa más avanzada primero, luego fecha descendente.

---

## Comportamiento común

- **Refresco**: polling (`config.ts → dataConfig`). Sin websockets en la primera versión.
- **Dato viejo**: si `as_of` supera el umbral → `Notice stale` arriba de la sección y la cifra en text-2.
- **Errores por sección**: si falla un endpoint, solo esa sección muestra `ErrorState`; el resto sigue.
- **Rutas con estado en la URL**: filtros, fecha y rango van en query params (compartibles y sobreviven al recargar).
- **Preferencias por dispositivo** (tema, densidad, último filtro): `localStorage` con try/catch.
- **Datos de ejemplo**: para desarrollo, un modo `?mock=1` o MSW con los mismos números del canvas (cuadran entre pantallas).
