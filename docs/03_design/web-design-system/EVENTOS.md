# Eventos: "Desde tu última visita" y "Próximos eventos"

Dos cosas distintas en Inicio:

| | Desde tu última visita | Próximos eventos |
|---|---|---|
| Mira | El pasado: qué pasó | El futuro: qué va a pasar |
| Fuente | Registro de eventos del propio sistema | Calendario externo + eventos propios |
| Modelo | `FeedEvent` | `CalendarEvent` |

La ventana del feed queda como está (última visita o 24 h; ver `config.ts`). Lo que se define acá es **qué hechos entran**.

---

## 1. Feed: catálogo de tipos

Criterio para que un hecho entre: **cambia algo que deberías saber o revisar**. Si no cambia nada (un refresco de precios), no entra.

| Tipo | Severidad | Se dispara cuando… | Agrupa por | Título (plantilla) | Subtítulo | Lleva a |
|---|---|---|---|---|---|---|
| `drawdown_limit` | crítica | El drawdown de una cuenta supera su límite | cuenta | "Drawdown sobre el límite en {cuenta}" | "−8,4 % desde el máximo; límite 8 %" | Cuenta |
| `stop_triggered` | advertencia | Se ejecuta una venta por stop | cuenta + día | "Saltó el stop de {símbolo}" / "Saltaron {n} stops en {cuenta}" | Larga: "Vendidas 18 a 468,00 · P&L −255,24" · Corta: "Recompradas 40 a 81,50 · P&L −184,00" | Cuenta › órdenes |
| `order_rejected` | advertencia | El broker rechaza una orden | cuenta + día | "{n} órdenes rechazadas en {cuenta}" | Motivos resumidos ("Poder de compra, fracciones y 1 sin motivo") | Cuenta › órdenes, filtro Rechazadas |
| `stop_near` | advertencia | Una posición (larga o corta) **entra** en zona "cerca" (< 2 %) | posición | "{símbolo} ({lado}) quedó a {pct} del stop" | "{cuenta} · stop {arriba\|abajo} {stop} · hoy {precio}" | Cuenta › posiciones |
| `stop_breached` | advertencia | El precio pasó el stop y la posición sigue abierta | posición | "{símbolo}: el precio pasó el stop" | "Stop {stop} · hoy {precio} · cierre pendiente" | Cuenta › posiciones |
| `shadow_divergence` | advertencia | Broker y sombra difieren más que el umbral | cuenta | "Sombra desalineada en {cuenta}" | "Difieren {gap} desde {fecha}" | Cuenta |
| `data_stale` | advertencia | Una cuenta supera el umbral sin datos | cuenta | "Sin datos de {cuenta} desde hace {t}" | "Último dato {hora} ART" | Cuenta |
| `alarm_resolved` | neutro | Se cierra una alarma | alarma | "Se resolvió: {título}" | "Duró {duración}" | Cuenta |
| `hypothesis_promoted` | info | Una hipótesis llega a **sombra, paper, candidata o producción** | hipótesis | "{nombre} pasó a {etapa}" | Métrica clave | Ficha |
| `hypothesis_discarded_late` | info | Cae una hipótesis en **etapa 3 o posterior** (las tempranas son ruido) | día | "Se descartó {nombre} en {etapa}" | Motivo | Ficha |
| `cycle_completed` | info | Termina un ciclo del embudo | ciclo | "Terminó el ciclo {n} del embudo" | "{entraron} entraron, {siguen} siguen, {cayeron} cayeron" | Embudo filtrado por ciclo |
| `order_filled` | neutro | Se ejecuta una orden (que no sea por stop) | cuenta + día | "{n} compras/ventas ejecutadas" | "MSFT 25 a 431,55 · SPY 4 a 565,98" | Cuenta › órdenes |
| `position_closed` | neutro | Se cierra una posición entera | posición | "Se cerró {símbolo}" | "P&L realizado {monto} · {pct}" | Cuenta |
| `dividend_received` | neutro | Entra un dividendo | cuenta + día | "Cobraste dividendos de {símbolos}" | "Neto {monto} · retenido {monto}" | Cuenta › dividendos |

Fuera del feed a propósito: aperturas de posición (ya se ven como órdenes ejecutadas), hipótesis descartadas en etapas 0–2 (van al resumen de ciclo), variaciones de precio, órdenes canceladas o vencidas (no requieren acción; se ven en Cuenta) y cuentas sin datos (no es un problema).

### Reglas de presentación

1. **Orden**: severidad (crítica → advertencia → info → neutro), y dentro de cada una, lo más reciente primero.
2. **Agrupación**: los hechos del mismo tipo, cuenta y día se juntan en un ítem (`group_count`). Nunca 3 líneas de "orden rechazada".
3. **Tope**: `homeConfig.feedMaxItems` (6) + "Ver todo" si hay más.
4. **Deduplicación**: si una alarma sigue activa, no se repite en el feed: ya está en el banner de estado. Entra solo cuando aparece o se resuelve.
5. **Feed vacío**: "Nada nuevo desde {hora}." en texto apagado; no se oculta la sección.
6. **Glifos** (cuadrado 28 px): crítica octógono/neg, advertencia triángulo/warn, rechazo cruz/neg, embudo flecha/accent, neutro +/neutral.

### Implementación sugerida (backend)

- Tabla `events` (append-only): `id, type, severity, occurred_at, account_id, entity_id, payload JSON`.
- La escriben los procesos que ya existen: sincronización con Alpaca (órdenes, posiciones, dividendos), el comparador broker/sombra, el motor del embudo.
- `GET /api/feed?since=` lee, agrupa y arma `title`/`subtitle` con las plantillas de arriba. Las plantillas viven en el backend (un solo lugar).
- Ventaja: queda un historial auditable y la pantalla no tiene que comparar estados "antes/después".

---

## 2. Próximos eventos: fuentes

Modelo único `CalendarEvent` (`api-types.ts`) con `kind` y `source`. Todo se guarda en UTC y se muestra `ET · ART`.

| Tipo | Fuente | Costo | Cómo |
|---|---|---|---|
| Días hábiles, feriados y cierres anticipados | **API de calendario de Alpaca** (`/v2/calendar`): devuelve los días de rueda con hora de apertura y cierre | Gratis, ya integrado | Job diario en el backend: los días hábiles que faltan son feriados; cierre antes de 16:00 = cierre anticipado. Alternativa sin red: `exchange_calendars` o `pandas_market_calendars` (Python). |
| Datos macro (inflación, empleo, PIB, etc.) | **API de FRED** (`fred/release/dates`), con `include_release_dates_with_no_data=true` para traer fechas futuras programadas | Gratis con API key | Job semanal. Lista blanca de publicaciones a seguir; los `release_id` se confirman con `fred/releases`. FRED da la fecha; la hora (08:30 ET típica) va en la lista blanca. |
| Reuniones de la Fed (FOMC) | Calendario publicado por la Reserva Federal (sin API) | Gratis | Carga manual una vez por año (8 reuniones) como evento propio. |
| Rebalanceos | **Propio**: configuración de cada estrategia (YAML) + calendario de mercado | — | El backend calcula la próxima fecha (por ejemplo, "primer día hábil del mes, 09:35 ET") cruzando la regla con el calendario de Alpaca. |
| Otros propios | **Propio**: tabla `calendar_events` con `source='manual'` | — | Recordatorios propios (revisar el ciclo, vencimientos, etc.). Carga por archivo YAML en el repo para no construir pantallas de edición (la app es de solo lectura). |

Notas:
- La app web no llama a Alpaca ni a FRED: lo hace el backend y guarda en su base. Así el teléfono no depende de servicios externos.
- Cambios de horario: Nueva York cambia de horario en marzo y noviembre; Argentina no. Con UTC + `Intl` con zona horaria (`format.ts`) la diferencia de 1 o 2 h sale sola.
- Prioridad sugerida: Alpaca (feriados) y rebalanceos primero; FRED después; FOMC manual cuando haga falta.

### Contrato de payload

Las plantillas del feed consumen estas claves opcionales en `payload`; si faltan, el texto se muestra con los datos disponibles:

- `drawdown_limit`: `drawdown_pct`, `limit_pct`.
- `stop_triggered`: `symbol`, `side`, `quantity`, `price`, `pnl`.
- `order_rejected`: `reason`.
- `stop_near`: `symbol`, `side`, `pct`, `stop`, `price`, `direction`.
- `stop_breached`: `symbol`, `stop`, `price`.
- `shadow_divergence`: `gap`, `date`.
- `data_stale`: `duration`, `last_seen`.
- `alarm_resolved`: `title`, `duration_seconds`, `alarm_type`.
- `hypothesis_promoted`: `name`, `stage`, `metric`.
- `hypothesis_discarded_late`: `name`, `stage`, `reason`.
- `cycle_completed`: `cycle`, `entered`, `remaining`, `dropped`.
- `order_filled`: `symbol`, `side`, `quantity`, `price`.
- `position_closed`: `symbol`, `pnl`, `pct`.
- `dividend_received`: `symbols`, `net`, `withheld`.
