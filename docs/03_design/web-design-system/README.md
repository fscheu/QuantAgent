# QuantAgent Web — traspaso de diseño a código (v0.2)

Paquete para empezar a programar el frontend (React + TypeScript) a partir del canvas de diseño.
Canvas: https://claude.ai/artifact/G3CGyvdzFTek2bSSRTr1uc (páginas: Sistema de diseño, Cuenta, Embudo, Inicio, v0.2 · Ajustes).

## Cambios en v0.2 (30/10/2026, revisión contra el backend real)

1. **Posiciones cortas**: `Position.side`; `stopDistance(side, …)` con zona `breached`; barra de stop con eje de precio (la corta crece desde la derecha); badge Larga/Corta; P&L de cortas con línea explicativa. Decisión #6 cerrada.
2. **Estados de orden**: `cancelled` (dos L) y `expired`; badge y chip "Vencidas", con explicación según `time_in_force`.
3. **Cuenta sin datos**: `AccountSummary` pasa a ser `active | no_data`; nuevo `AccountNoData` en Cuenta, variante en `AccountCard` y en `AccountSelect`. Nunca ceros.
4. **`DataTable` genérica**: orden por columna con `aria-sort`, numéricos a la derecha, casillas con máximo, estados, densidades y versión en tarjetas. `PositionRow` y `OrderRow` pasan a construirse sobre ella.

Archivos tocados: `api-types.ts`, `config.ts`, `help-texts.es.json`, `COMPONENTES.md`, `PANTALLAS.md`, `EVENTOS.md`, `DECISIONES.md`, `README.md`. Sin cambios: `tokens.css`, `tokens.ts`, `format.ts`.

## Contenido

| Archivo | Para qué | Dónde va en el repo (sugerido) |
|---|---|---|
| `tokens.css` | Variables `--qa-*`: color claro/oscuro, tipografía, espaciado, radios, movimiento, densidad | `web/src/styles/tokens.css` |
| `tokens.ts` | Los mismos colores en TS, para lightweight-charts y gráficos que no leen CSS; `candleTheme()`, `heatStep()` | `web/src/styles/tokens.ts` |
| `format.ts` | Formato es-AR: montos, %, signo, horas ET/ART, relativo | `web/src/lib/format.ts` |
| `config.ts` | Umbrales y parámetros provisorios (stop, dato viejo, etapas, feed) | `web/src/config.ts` |
| `help-texts.es.json` | Textos del "¿qué es esto?" | `web/src/content/help-texts.es.json` |
| `api-types.ts` | Contrato de datos que necesitan las pantallas (propuesta para Pydantic) | `web/src/api/types.ts` + espejo en `api/schemas.py` |
| `COMPONENTES.md` | Inventario: props, variantes, estados, accesibilidad, gráficos | `docs/design/` |
| `PANTALLAS.md` | Composición de cada pantalla, endpoints y comportamiento común | `docs/design/` |
| `EVENTOS.md` | Catálogo del feed "Desde tu última visita" y fuentes del calendario | `docs/design/` |
| `DECISIONES.md` | Decisiones abiertas con valor provisorio + registro de las tomadas | `docs/design/` (o raíz) |

## Reglas de trabajo

1. **Tokens: el código manda.** Un color o tamaño nuevo se agrega en `tokens.css` (y en `tokens.ts` si es color). Ningún componente usa valores literales.
2. **Estructura de pantallas: el canvas manda.** Agregar una sección, reordenar o hacer una pantalla nueva → primero en el canvas, después en código. Los ajustes visuales chicos van directo al código.
3. **Lo no decidido va a `config.ts` o al backend**, nunca dentro de un componente.
4. **Cada componente se construye con todos sus estados** (cargando, vacío, error, desactualizado) antes de usarse en una pantalla.

## Orden de construcción sugerido

1. **Base**: Vite + React + TS, `tokens.css`, fuentes Geist y Geist Mono servidas localmente desde npm (por ejemplo, los paquetes de Fontsource; la app corre en Tailscale, conviene no depender de Google Fonts). Tema por `data-theme`, densidad por `data-density`.
2. **Catálogo de componentes** (Storybook o Ladle) que reproduzca las hojas 03 y 04 en claro y oscuro: `Num`, `Delta`, `Badge`, `HelpTip`, `Notice`, `Skeleton`, estados vacío/error, `SegmentedControl`, `Chip`, navegación.
3. **`DataTable`** y después los componentes de dominio: `StopDistanceBar` (larga y corta), `PositionCard/Row`, `OrderCard/Row` + `RejectReason`, `AccountCard` (con variante sin datos), `AccountNoData`, `FunnelChart`, `HypothesisCard`.
4. **Mocks** (MSW) con los datos del canvas, que cuadran entre pantallas.
5. **Pantallas**: Cuenta → Embudo → Inicio, primero con mocks y después con la API real.
6. **Backend**: tabla `events` + `/api/feed`, jobs de calendario (Alpaca primero, FRED después).

## Prompt sugerido para arrancar con Claude Code

> Leé `docs/design/README.md`, `COMPONENTES.md` y `PANTALLAS.md`. Armá el proyecto `web/` con Vite + React + TypeScript, copiá `tokens.css`, `tokens.ts`, `format.ts` y `config.ts` a las rutas indicadas y construí las primitivas de la sección 1 de `COMPONENTES.md` con un catálogo en Ladle, en claro y oscuro, con todos sus estados. No uses colores literales: solo variables `--qa-*`. Pará al terminar las primitivas para que las revise.

## Pendiente de diseño

- Mercado y Ficha de hipótesis (cuando se pidan).
- Pantalla "Ver todo" del feed.
- Ajustes en teléfono (tema y densidad): hoja simple, no diseñada en detalle.
