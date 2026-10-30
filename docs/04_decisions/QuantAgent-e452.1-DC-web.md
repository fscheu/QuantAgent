# QuantAgent-e452.1 · ADR: arquitectura de la web

- **Ticket:** WEB-01.
- **Fecha:** 2026-10-30.
- **Estado:** aprobado por Fede el 2026-10-30.
- **Documento técnico:** [Arquitectura de la web](../03_design/QuantAgent-e452.1-DS-web.md).
- **Decisión anterior superada:** [ui_framework_decision.md](./ui_framework_decision.md).

## Contexto

M-web necesita una interfaz para consultar cuentas desde compu y teléfono, con una API estable entre el modelo Python y React. El documento anterior proponía Streamlit y luego Angular. El roadmap y el design system posteriores definen una web responsiva, React/TypeScript y endpoints de lectura. El modelo existente separa el entorno de ejecución (`paper`, `prod`) del `bucket_id` guardado en `trade_intents`; no toda fila operativa tiene ambos.

## Decisiones (aprobadas por Fede el 2026-10-30)

### 1. Qué significa «cuenta»

**Aprobada por Fede (2026-10-30):** cuenta API = bucket/estrategia operativa dentro de un entorno (`paper` o `live`), por ejemplo `paper-m4`. La cuenta Alpaca es el vínculo externo que corresponde a ese bucket y entorno, no la identidad que reemplaza al bucket. La configuración declara cuentas aunque aún no tengan datos; se modelan como `AccountSummary` con `status: 'no_data'`, `no_data_reason` y `checked_at`, sin ceros. Una cuenta activa entrega `status: 'active'` y sus valores. `AccountRef.id/name/env` representa esa identidad (`paper-m4`, `Paper M4`, `paper`). Se mantienen separadas M4, M5 y live; no se suma paper con live.

**Alternativa descartada:** exponer solo cada cuenta Alpaca. El modelo persistido identifica entorno, pero no siempre la cuenta Alpaca; además pierde la separación por bucket/estrategia necesaria para M4 y M5. Exponer solo bucket también es insuficiente cuando dos entornos comparten el bucket. El backend debe definir el mapeo explícito antes de publicar el selector.

### 2. Fechas

**Aprobada por Fede (2026-10-30):** toda fecha-hora de la API se serializa como ISO 8601 UTC con sufijo `Z`; el frontend la muestra en la zona apropiada. `orders?date=YYYY-MM-DD` interpreta `date` como día calendario de Nueva York, con límites convertidos a UTC para consultar órdenes. Campos de calendario que son días enteros conservan `YYYY-MM-DD` en fecha de NY.

**Alternativa descartada:** usar fecha local del servidor o ART para filtrar órdenes; la medianoche del servidor no coincide con la rueda de Nueva York ni con sus cambios de horario.

### 3. Traducción de campos

**Aprobada por Fede (2026-10-30):** el esquema Pydantic adopta `api-types.ts` como contrato público y traduce los nombres del ORM. La traducción no cambia el almacenamiento.

| Modelo/dato interno | Campo API | Regla |
|---|---|---|
| `Order.quantity` | `Order.qty` | Mantener la cantidad informada; normalizar tipo numérico. |
| `Order.order_type` | `Order.type` | Valor enum en minúsculas (`market`, `limit`, `stop`, `stop_limit`). |
| `Order.filled_quantity` | `Order.filled_qty` | Cantidad ejecutada acumulada. |
| `Order.average_fill_price` | `Order.filled_avg_price` | Nulo si no hay precio medio. |
| `Order.created_at` | `Order.submitted_at` | Instante de creación serializado en UTC `Z`. |
| `Order.price` | `Order.limit_price` | Precio límite cuando aplica; en otros tipos, `null`. |
| `Trade.quantity` / posición | `Position.qty` | Absoluta; el lado se informa por `Position.side`. |
| `Position.average_entry_price` | `Position.avg_entry_price` | Precio medio de entrada. |
| `Position.unrealized_pnl` | `Position.unrealized_pl` | Con signo económico correcto para largos y cortos. |
| `Position.unrealized_pnl_pct` | `Position.unrealized_pl_pct` | Para corto: `(entrada − actual) / entrada`; el backend confirma el cálculo. |
| `Order.environment` | `AccountRef.env` | `paper` o `live`; `PROD` se presenta como `live`. |
| `TradeIntentRecord.reject_reason` | `Order.reject_reason` | Traducir código a texto claro; sin detalle/raw persistido, devolver `null`. |
| `ActivePosition.stop_loss` | `Position.stop_price` | Stop activo vigente unido por símbolo, lado y entorno; `null` si no hay stop vigente. |

Otros campos salen de sus nombres y significado definidos en `api-types.ts`; los enums se normalizan a los literales de ese contrato. El contrato OpenAPI generado por WEB-05 pasa a ser la fuente verificable para la interfaz.

**Alternativa descartada:** exponer nombres SQLAlchemy directamente. Obliga a la interfaz a conocer el esquema interno y no satisface el contrato snake_case ya diseñado (`quantity` frente a `qty`, entre otros).

### 4. Estados de órdenes

**Aprobada por Fede (2026-10-30):** incluir todos los estados del contrato: `pending`, `partially_filled`, `filled`, `rejected`, `cancelled` (dos L, como el enum SQLAlchemy `CANCELLED`) y `expired`. El backend no transforma `cancelled` a la variante de una sola L. La interfaz distingue cancelada de vencida.

**Alternativa descartada:** limitarse a los estados de los primeros bocetos o traducir la ortografía del enum a una sola L; eso pierde estados existentes y descalza el contrato v0.2.

### 5. Posiciones cortas

**Decisión tomada por Fede el 2026-10-30:** la web muestra posiciones cortas. La API expone `Position.side` (`long` o `short`), normaliza `qty` y `market_value` como magnitudes positivas y conserva el signo económico de P&L. Para porcentaje corto se usa `(entrada − actual) / entrada`, según la decisión #20 de `web-design-system/DECISIONES.md`; el cálculo queda a confirmar con el backend. No se reabre la decisión de mostrarlas.

## Piezas y operación

**Aprobada por Fede (2026-10-30):** FastAPI + Pydantic brinda la frontera de lectura y contrato; React + TypeScript consume esa API. La API consulta Postgres para el estado operativo SQLAlchemy y DuckDB para series/snapshots analíticos, siempre en lectura. Ambos servicios web conviven en la VM con el scheduler, en puertos locales `8000` (API) y `8080` (web); Tailscale Serve publica el acceso privado y enruta API/UI. No se arma autenticación propia por usuario: el acceso a la VM se controla con Tailscale. La configuración de la API determina qué cuentas aparecen y cada una conserva su entorno.

**Regla tomada para el alcance:** la web nunca toca el broker. Sus rutas son `GET`; no coloca, cancela ni modifica órdenes, posiciones o stops y no expone credenciales en el navegador.

**Alternativa descartada:** Flask o Django para la API y Angular para la interfaz, dirección del documento anterior. FastAPI encaja con Pydantic y genera OpenAPI; React/TypeScript es lo que adoptó el design system y permite contrastar los tipos del cliente con el contrato. Streamlit no satisface la interfaz final responsiva prevista.

## Consecuencias

- Antes de exponer varias cuentas hay que resolver el mapeo bucket-entorno-cuenta Alpaca en backend; los modelos actuales no lo llevan de punta a punta.
- Postgres y DuckDB tienen responsabilidades distintas; las consultas web no actualizan sus fuentes. La historia que aún no se persiste requiere snapshots u otros tickets.
- La autenticación de red limita el acceso, mientras que el listado de cuentas sigue siendo configuración explícita.
- La aplicación es de monitoreo, no de ejecución.

## Referencias

- [Diseño del sistema web](../03_design/QuantAgent-e452.1-DS-web.md)
- [Roadmap M-web, §3](../02_planning/2026-10-28_roadmap_post_m5.md#3-m-web-aplicación-web-que-reemplaza-streamlit)
- [Tickets del lote M-web, §A y §B](../02_planning/QuantAgent-e452-PL-tickets-lote-m-web.md)
- [Tipos del design system](../03_design/web-design-system/api-types.ts)
- [Decisiones del design system](../03_design/web-design-system/DECISIONES.md)

Estado: aprobado por Fede el 2026-10-30. Las propuestas de las decisiones 1 a 4 y la arquitectura quedan firmes.
