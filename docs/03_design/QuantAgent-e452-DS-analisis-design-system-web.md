# QuantAgent-e452 · Análisis del design system de la web contra L34

- **Fecha:** 2026-10-30. **Epic:** `QuantAgent-e452` (L34 web-cuenta, WEB-01 a WEB-05).
- **Paquete analizado:** `docs/03_design/web-design-system/` (v0.1, generado en Claude Design a partir del prompt `docs/02_planning/2026-10-30_prompt_claude_design_m_web.md`). Canvas: https://claude.ai/artifact/G3CGyvdzFTek2bSSRTr1uc
- **Alcance de este documento:** solo análisis. No cambia código ni tickets. Los archivos del paquete se copiaron tal cual, sin edición.

## 1. Qué cubre el paquete

| Área | Archivos | Estado |
|---|---|---|
| Tokens (claro/oscuro, densidad, tipografía, espaciado, movimiento, 8 colores categóricos, escala de calor, bandas de recesión) | `tokens.css`, `tokens.ts` | Completo. Listo para copiar a `web/src/styles/`. |
| Formato es-AR (montos, %, signo, horas ET/ART, relativo) | `format.ts` | Completo. |
| Inventario de componentes (primitivas, controles, dominio, gráficos, checklist de accesibilidad) | `COMPONENTES.md` | Completo para Cuenta, Embudo e Inicio. |
| Composición de pantallas y endpoints | `PANTALLAS.md` | Cuenta, Embudo, Inicio. |
| Contrato de datos propuesto | `api-types.ts` | Cuenta, Embudo, Inicio, calendario. |
| Catálogo del feed y fuentes del calendario | `EVENTOS.md` | Completo, pero depende de backend que no existe. |
| Textos "¿qué es esto?" | `help-texts.es.json` | 14 términos. |
| Decisiones abiertas con valor provisorio | `DECISIONES.md`, `config.ts` | 18 abiertas, 7 tomadas. |

Contra el prompt de #675: están los tokens, claro/oscuro, las dos densidades, los componentes pedidos, los gráficos (línea, velas, acumulado, barras, calor, sparkline, recesión) y las tres pantallas de la primera entrega con sus estados vacío, cargando, error y "motivo no disponible". Mercado y Ficha de hipótesis quedan afuera a propósito (el prompt los difería).

## 2. Qué toca L34 y dónde el contrato no calza con el backend

L34 es solo API (sin frontend): WEB-03 resumen y posiciones, WEB-04 órdenes con rechazadas, WEB-05 contrato OpenAPI. `api-types.ts` es una propuesta para Pydantic, y contra `quantagent/models.py` hay diferencias que WEB-03/04 tienen que resolver antes de congelar el contrato:

| # | Tema | Diseño asume | Backend hoy | Qué hacer |
|---|---|---|---|---|
| 1 | **Cuentas** | `AccountRef{id,name,env}`, tres cuentas (`paper-m4`, `paper-m5`, `live`) | `Order` y `Position` solo tienen `environment` (paper/live). `Position.symbol` es único. `trade_intents` tiene `bucket_id`, pero `Order` no. | WEB-01 (ADR) define qué es una "cuenta" en la API: bucket, cuenta de Alpaca o ambos. Sin eso no hay selector de cuenta. |
| 2 | **Stop vigente** | `Position.stop_price` (null = "Sin stop vigente") | El stop está en `ActivePosition.stop_loss` (NOT NULL), por símbolo, sin FK a `Position`. | WEB-03 hace el join por símbolo y devuelve null si no hay fila. Ojo con varias filas por símbolo: tomar la vigente. |
| 3 | **Motivo de rechazo** | `reject_reason` (lenguaje claro), `reject_detail`, `reject_reason_raw` | Solo `trade_intents.reject_reason` (String 100, código), enlazado por `order_id`. No hay detalle ni texto crudo del broker. | WEB-04 devuelve `reject_reason` traducido desde una tabla de códigos en el backend; `detail` y `raw` quedan `null` hasta que L26 los guarde. El estado "motivo no disponible" ya cubre ese caso. |
| 4 | **Nombres de campos** | `qty`, `limit_price`, `filled_qty`, `filled_avg_price`, `submitted_at`, `unrealized_pl(_pct)`, `type` | `quantity`, `price`, `filled_quantity`, `average_fill_price`, `created_at`, `unrealized_pnl(_pct)`, `order_type` | Decidir de qué lado se renombra. Recomiendo que el esquema Pydantic traduzca y los tipos de la web queden como están. |
| 5 | **Estados de orden** | `canceled` | `cancelled` (con dos L) y además `expired` | Alinear ortografía y agregar `expired` al diseño (Badge y filtro). |
| 6 | **Resumen de cuenta** | `equity`, `invested`, `cash`, `day_change`, `week_change`, `as_of`, `stale` | No hay tabla de snapshots ni de equity. | Vienen del broker (Alpaca) o de un snapshot nuevo. Variación de semana y `as_of` necesitan historia. Es el campo más riesgoso de WEB-03. |
| 7 | **Posiciones cortas** | Solo largas (`stopDistance` asume largas, decisión #6) | `Position.side` existe y F2 aprobó cortos en paper | La decisión #6 deja de ser provisoria: hay que decidir antes de WEB-03 si la API expone `side` y cómo se dibuja el stop en corto. |
| 8 | **Fechas** | ISO 8601 UTC | Columnas `DateTime` naive con `utcnow` | La API tiene que serializar con `Z`. Además, `orders?date=YYYY-MM-DD` debería ser fecha de Nueva York; fijarlo en el contrato. |
| 9 | **Alarmas** | `AccountSummary.alarms[]` | No existe catálogo ni tabla | Hoy la sección se muestra como "Próximamente" (así lo diseñó el paquete). WEB-03 devuelve `[]`. |

Lo que el paquete diseña pero **no está en L34** (queda como "Próximamente" o ghost): curva de valor vs SPY y sombra, exposición por sector, dividendos (existe `quantagent/trading/alpaca_dividendos.py` como punto de partida), feed y banner de estado de Inicio, calendario, todo Embudo.

## 3. Qué falta para las pantallas planeadas

**Cuenta** (la única que L34 alimenta): resumen, posiciones y órdenes se pueden construir con las correcciones 1 a 8. Faltan los tres bloques "próximamente".

**Embudo:** el diseño está completo, pero la API no existe y espera M-datos y el registro único del embudo. Las etapas 0 a 3 y los motivos de descarte son provisorios (decisiones #1 a #3). Hoy no hay ciclo ni hipótesis con esa forma en la base.

**Inicio:** depende de lo anterior más dos piezas nuevas de backend que no están en ningún lote: la tabla `events` (append-only) con `/api/feed`, y los jobs de calendario (Alpaca primero, FRED después, FOMC manual). Se puede construir con el banner, las tarjetas de cuenta y secciones fantasma, sin feed.

**Fuera del paquete (a propósito):** Mercado, Ficha de hipótesis, "Ver todo" del feed, hoja de ajustes en teléfono (tema y densidad).

**Huecos del diseño que vi al leerlo:**
- No hay componente genérico de tabla ordenable (solo `PositionRow` y `OrderRow` con anchos fijos). Si Mercado o Embudo van a tener tablas, conviene un `DataTable`.
- Estado "sin cuenta live todavía" (hoy la cuenta live no existe): no hay vista para una cuenta listada sin datos.
- El prompt pedía "no basar las velas solo en rojo/verde": el paquete usa hueco/lleno como respaldo, lo cumple.
- Fuentes Geist: el README pide servirlas desde npm (Fontsource). Hay que agregar esa dependencia y revisar la licencia (SIL OFL, sin problema).

## 4. Frontend: tickets que habría que abrir

L34 no incluye frontend, y el prompt de #675 dice que el diseño aprobado es la condición para abrir esos tickets. Propuesta, para que Fede la apruebe, sin crearlos acá:

1. **Base del frontend:** Vite + React + TS en `web/`, copiar tokens/format/config, fuentes locales, temas y densidad.
2. **Catálogo de primitivas y controles** (Ladle), claro y oscuro, con todos los estados.
3. **Componentes de dominio de Cuenta** (`StopDistanceBar`, `PositionCard/Row`, `OrderCard/Row`, `RejectReason`, `AccountCard`).
4. **Mocks (MSW)** con los números del canvas.
5. **Pantalla Cuenta** contra mocks y después contra la API de WEB-03/04.

Embudo e Inicio, después de M-datos y del registro único. Sugiero que 1 a 3 puedan correr en paralelo con L34 porque solo dependen del contrato (WEB-05).

## 5. Decisiones que le tocan a Fede

1. Ubicación: dejé el paquete completo en `docs/03_design/web-design-system/`. Cuando se abra el ticket de base, los archivos de código (`tokens.css`, `tokens.ts`, `format.ts`, `config.ts`, `help-texts.es.json`, `api-types.ts`) se copian a `web/src/...` según la tabla del README del paquete.
2. Posiciones cortas en la web (#7 de la sección 2): hay que decidirlo antes de WEB-03.
3. Qué es una "cuenta" en la API (#1): entra en WEB-01.
4. Si se abren los cinco tickets de frontend de la sección 4 y cuándo.

## 6. Cómo validar

- `ls docs/03_design/web-design-system/` muestra los 11 archivos del zip sin cambios.
- Las diferencias de la sección 2 se pueden verificar en `quantagent/models.py` (clases `Order`, `Position`, `TradeIntentRecord`, `ActivePosition`, enums `OrderStatus` y `OrderType`).
