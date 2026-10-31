# QuantAgent-e452 · Borrador de tickets del lote M-web

- **Fecha:** 2026-10-30. **Estado:** lista aprobada por Fede el 2026-10-30. Cargado en Beads el 2026-10-30 (IDs entre paréntesis; WEB-01..05 ya existían de L34; WEB-14 y FE-08 en estado blocked). Decisión de Fede: la web muestra posiciones cortas; `lote/M-web` va antes que L33.
- **Rama del lote:** `lote/M-web` (creada desde `main`). Absorbe L34 (`lote/web-cuenta`, epic `QuantAgent-e452`) y suma el frontend y las piezas de backend que pide el design system.
- **Fuentes:** `docs/03_design/web-design-system/` (en particular `EVENTOS.md`, `PANTALLAS.md`, `api-types.ts`), `docs/03_design/QuantAgent-e452-DS-analisis-design-system-web.md`, `PLAN-POST-M5.md` §M-web.
- **Restricción:** ningún ticket toca `quantagent/fabrica` ni `quantagent/data`. Lo que dependa de esas áreas (escritores de eventos del embudo, API de Embudo) queda marcado como diferido.
- **Cómo cargar:** una sesión en la VM crea los tickets con `bd create` desde esta lista, bajo el epic `QuantAgent-e452`, y registra los IDs en el README de `docs/02_planning/`.

## A. Backend (L34 original, con las correcciones del análisis)

| Borrador | Título | Cambia respecto de L34 |
|---|---|---|
| WEB-01 (`QuantAgent-e452.1`) | Arquitectura y ADR | Debe definir qué es una "cuenta" en la API (bucket, cuenta Alpaca o ambos), la política de fechas (UTC con `Z`, día de NY para `orders?date=`), nombres de campos (el esquema Pydantic traduce `quantity→qty`, etc.), estados de orden (`cancelled`/`expired`) y si se exponen posiciones cortas. |
| WEB-02 (`QuantAgent-e452.2`) | Esqueleto FastAPI | Sin cambios. Incluye `GET /api/accounts` y `GET /api/market/clock`. |
| WEB-03 (`QuantAgent-e452.3`) | Resumen y posiciones | Stop vigente por join con `ActivePosition`; `as_of` y `stale`; `alarms: []` hasta que exista WEB-11. Resumen según WEB-07. |
| WEB-04 (`QuantAgent-e452.4`) | Órdenes con rechazadas | `reject_reason` traducido desde `trade_intents.reject_reason` (tabla de códigos en backend); `reject_detail` y `reject_reason_raw` en `null` mientras no se guarden. |
| WEB-05 (`QuantAgent-e452.5`) | Contrato OpenAPI y guía | El contrato publicado reemplaza a `api-types.ts` como fuente; los tipos de la web se generan de OpenAPI o se verifican contra él. |

## B. Backend nuevo (sale del design system)

| Borrador | Título | Depende de | Notas |
|---|---|---|---|
| WEB-06 (`QuantAgent-e452.6`) | Snapshots de equity de la cuenta | WEB-01 | Tabla y job que guarda equity, efectivo e invertido por cuenta. Habilita variación del día y de la semana, `spark_30d` y la curva de valor. Hoy no hay historia. |
| WEB-07 (`QuantAgent-e452.7`) | Tabla `events` (append-only) y escritores de cuenta | WEB-01 | Esquema de `EVENTOS.md`: `id, type, severity, occurred_at, account_id, entity_id, payload`. Escritores iniciales: órdenes rechazadas, stop ejecutado, orden ejecutada, posición cerrada, dividendo. Los eventos de embudo se difieren (tocan `fabrica`). |
| WEB-08 (`QuantAgent-e452.8`) | `GET /api/feed` | WEB-07 | Agrupa por tipo + cuenta + día, orden por severidad, plantillas de título y subtítulo en el backend, deduplicación contra alarmas activas, `since` y tope. |
| WEB-09 (`QuantAgent-e452.9`) | Calendario: modelo y feriados de Alpaca | WEB-02 | `CalendarEvent`, job diario con `/v2/calendar` (feriados y cierres anticipados) y `GET /api/calendar?days=`. |
| WEB-10 (`QuantAgent-e452.10`) | Calendario: rebalanceos de las estrategias | WEB-09 | Calcula la próxima fecha de cada regla de rebalanceo cruzándola con el calendario de mercado. Lee la configuración de estrategias, no la modifica. |
| WEB-11 (`QuantAgent-e452.11`) | Calendario: FRED y FOMC | WEB-09 | Job semanal con lista blanca (inflación, empleo; los `release_id` se confirman) y archivo YAML en el repo para FOMC y recordatorios propios (`source='manual'`). Necesita API key de FRED de Fede. |
| WEB-12 (`QuantAgent-e452.12`) | Alarmas mínimas y `GET /api/home/status` | WEB-06, WEB-07 | Catálogo inicial: drawdown, sombra desalineada, dato viejo. Alimenta `AccountSummary.alarms`, `HomeStatus` y los eventos `alarm_resolved`. Umbrales a `config` (decisiones #11 y #12). |
| WEB-13 (`QuantAgent-e452.13`) | Curva vs SPY y sombra, exposición por sector, dividendos | WEB-06 | Los tres bloques "próximamente" de Cuenta. Dividendos parte de `alpaca_dividendos.py`. Se puede abrir después del primer corte de Cuenta. |
| WEB-14 (`QuantAgent-e452.14`) | API de Embudo (ciclos, hipótesis, descartes) | M-datos y registro único del embudo | **Diferido.** Toca `fabrica`. No se abre hasta que existan. |

## C. Frontend

| Borrador | Título | Depende de | Notas |
|---|---|---|---|
| FE-01 (`QuantAgent-e452.15`) | Base de `web/` | WEB-01 | Vite + React + TS, `tokens.css`, `tokens.ts`, `format.ts`, `config.ts`, `help-texts.es.json`, Geist local por Fontsource, temas y densidad. Rutas de destino según el README del design system. |
| FE-02 (`QuantAgent-e452.16`) | Primitivas y controles con catálogo Ladle | FE-01 | `Card`, `Num`, `Delta`, `Badge`, `HelpTip`, `Notice`, `Skeleton`, estados vacío y error, `SegmentedControl`, `Chip`, navegación. Claro y oscuro, todos los estados. Punto de control: Fede revisa antes de seguir (como pide el README). |
| FE-03 (`QuantAgent-e452.17`) | Componentes de dominio de Cuenta | FE-02 | `StopDistanceBar`, `PositionCard/Row`, `OrderCard/Row`, `RejectReason`, `AccountCard`, `AccountSummaryCard`. |
| FE-04 (`QuantAgent-e452.18`) | Mocks MSW con los números del canvas | FE-01, WEB-05 | Modo `?mock=1`. Se generan desde el contrato OpenAPI. |
| FE-05 (`QuantAgent-e452.19`) | Pantalla Cuenta | FE-03, FE-04 | Primero con mocks, después contra WEB-03 y WEB-04. Secciones planeadas como fantasma punteado detrás de un flag. |
| FE-06 (`QuantAgent-e452.20`) | Pantalla Inicio | FE-05, WEB-08, WEB-09, WEB-12 | Banner, tarjetas de cuenta, feed, próximos eventos. La tarjeta de Embudo queda fantasma. |
| FE-07 (`QuantAgent-e452.21`) | Gráficos de Cuenta | FE-05, WEB-13 | `EquityChart`, `DivergingBar`, `Sparkline`; Recharts. |
| FE-08 (`QuantAgent-e452.22`) | Pantalla Embudo | FE-02, WEB-14 | **Diferido** junto con WEB-14. |

## D. Orden sugerido

1. En paralelo: WEB-01 → WEB-02 → WEB-03/04 → WEB-05 (backend de Cuenta) y FE-01 → FE-02 (con parada de revisión).
2. Luego: FE-03, FE-04, FE-05 y WEB-06.
3. Luego: WEB-07 → WEB-08, WEB-09 → WEB-10/11, WEB-12, FE-06.
4. Al final: WEB-13, FE-07. Embudo (WEB-14, FE-08) cuando existan M-datos y el registro único.

## E. Pendiente antes de cargar en Beads

- Los cortos se muestran: WEB-01 y WEB-03 exponen `side`, y el diseño de `StopDistanceBar` para cortos se pide al designer.
- Definición de cuenta (entra en WEB-01).
- API key de FRED (WEB-11).
