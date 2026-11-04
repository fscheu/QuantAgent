# QuantAgent Web — traspaso de diseño a código (v0.4.2)

Paquete para empezar a programar el frontend (React + TypeScript) a partir del canvas de diseño.
Canvas: https://claude.ai/artifact/G3CGyvdzFTek2bSSRTr1uc (páginas: Embudo v0.4, Ficha de hipótesis, Mercado, v0.4 · Componentes, Sistema de diseño, Cuenta, Embudo v1 (reemplazada), Inicio, v0.2 · Ajustes, Rendimiento).

## Cambios en v0.4.2 (04/11/2026)

- Sectores: los 11 con score y ranking (motor de M5, universo de 14); habilitados = top 3 con score positivo, sean cuales sean. Se corrigió el supuesto #55.
- “En cartera” sale de posiciones reales de M4 y M5 (hoy M5 en efectivo, USD 3.000): estados “habilitado con posición” y “habilitado sin posición todavía”.
- Embudo: “Período: aplica al embudo y a la lista”; el ciclo actual y la regla no dependen del período; con un ciclo elegido, el período se deshabilita.
- FXE sin bloque propio (no es el dólar amplio). `config.ts` suma `m5Universe`.

## Cambios en v0.4.1 (04/11/2026, respuestas de Fede)

- “Sin ciclo”: 55 **eventos de etapa** (no hipótesis) + 202 intentos por etiqueta de texto libre; sin embudo. Tipo `HistorySummary` nuevo.
- Período 30d/90d = “con actividad en los últimos N días”; estado del último evento en la ventana.
- Motor: `state_description` del backend (texto de ejemplo marcado) y definiciones fijas de risk-on / neutral / risk-off.
- Comandos reales `quantagent registro …` en los textos de ayuda (`cycleConfig.cli`).
- Mercado: universo real de 29 (sin efectivo, con Sectores 9 y Moneda FXE); ranking del motor sobre los 9 sectores del universo, XLC y XLRE como contexto (supuesto corregido en v0.4.2); “En cartera” también en sectores.
- Evento informativo `review_rule_reached` en el feed (EVENTOS.md) y regla de dato macro viejo (`marketConfig.macroStale`).

## Cambios en v0.4 (03/11/2026)

- **Embudo rediseñado** sobre el registro real: ciclos `YYYY-MM` abiertos/cerrados, hitos del mes (1 · 8 · 25), máximo 5 hipótesis, etapas E0 Admisión · E1 Filtro · E2 Validación en el motor · E3 Holdout + fase posterior atenuada, veredictos PASA / CORTA / ERROR / ALARMA (ERROR y ALARMA aparte, como “detenidas”), promovida, intentos con inválidos, recibo inmutable con snapshots, “Sin ciclo (historial)” y regla de revisión 6 de 6.
- **Ficha de hipótesis** mínima (`/hipotesis/:id`).
- **Mercado** (`/mercado`): régimen de SPY y estado del motor, sectores (tabla y mapa de calor), macro (T10Y2Y con recesiones + indicadores “próximamente”), 29 ETF con comparación y velas, eventos de 14 días.
- Navegación: Embudo y Mercado activos.
- Archivo nuevo: `EMBUDO-MERCADO.md` (cambios de contrato y lista de datos por elemento con qué pasa si es null).
- Tocados: `api-types.ts` (Funnel, CycleSummary, Hypothesis corregidos + tipos nuevos de Embudo y Mercado), `config.ts` (`funnelConfig`, `cycleConfig`, `marketConfig`, `sectorHeatStep()`), `help-texts.es.json` (etapas, veredictos, promovida, holdout, snapshot, régimen, risk-on/off, momentum, curva invertida…), `COMPONENTES.md` (§3 actualizada, §7 y §8 nuevas), `PANTALLAS.md`, `DECISIONES.md` (#1 cerrada, #2/#3/#9/#10 actualizadas, #41–#54), `tokens.css` (solo clases utilitarias; **ninguna variable nueva**).

## Cambios en v0.3.2 (01/11/2026)

- **Primer corte = Rendimiento de backtests** + lista mínima de Backtests (casillas, hasta dos).
- `featureFlags` en `config.ts`: `accountPerformanceTab` (pestaña en Cuenta; se enciende cuando el backend calcule TWR) y `runPerformance` (rendimiento por estrategia; se enciende cuando haya capital por estrategia). Ambos apagados.
- Estrategia en ejecución: nuevo estado “rendimiento no aislable” (`not_computable_reason: 'capital_not_isolated'`). Cartera: el estado pasa a “1 aporte sin registrar”, para cuando el flag esté encendido.
- DECISIONES #38 (decide Fede), #39 y #40.

## Cambios en v0.3.1 (01/11/2026)

- Umbrales de Rendimiento (21/63/252 días, 12 rebalanceos) como parámetros del backend (`sample.thresholds`), no fijos.
- Distribución: mensual por defecto + vista diaria secundaria (desde 63 días) con VaR 95 % y CVaR.
- Backtests pasa a ser sección propia de la navegación (5 destinos); fuera de Embudo.
- Cartera: nuevo estado `returns_not_computable` (“hay aportes sin registrar”) y prerrequisito de backend (registro de flujos + id de cuenta + TWR, junto con QW-04). Ver RENDIMIENTO.md §9.

## Cambios en v0.3 (01/11/2026)

Nueva pantalla **Rendimiento** (tearsheet nativo) para backtest, estrategia en ejecución y cartera. Archivo nuevo: `RENDIMIENTO.md` (rutas, navegación, comparación, layout, métricas con fórmulas y mínimos, gráficos, estados y contrato de datos). Tocados: `COMPONENTES.md` (§6, 11 componentes nuevos o extendidos), `PANTALLAS.md` (sección Rendimiento y pestañas en Cuenta), `api-types.ts` (`PerformanceReport` y afines), `config.ts` (`performanceConfig`, `monthlyHeatStep()`), `help-texts.es.json` (términos de métricas), `DECISIONES.md` (#28–#37).

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
| `RENDIMIENTO.md` | Pantalla Rendimiento completa + contrato de datos para el backend | `docs/design/` |
| `EMBUDO-MERCADO.md` | Embudo v0.4, Ficha y Mercado: cambios de contrato y datos por elemento (null → qué se muestra) | `docs/design/` |
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
5. **Pantallas**: Cuenta → Embudo → Inicio, primero con mocks y después con la API real. Después: Ficha de hipótesis y Mercado (v0.4).
6. **Backend**: tabla `events` + `/api/feed`, jobs de calendario (Alpaca primero, FRED después).

## Prompt sugerido para arrancar con Claude Code

> Leé `docs/design/README.md`, `COMPONENTES.md` y `PANTALLAS.md`. Armá el proyecto `web/` con Vite + React + TypeScript, copiá `tokens.css`, `tokens.ts`, `format.ts` y `config.ts` a las rutas indicadas y construí las primitivas de la sección 1 de `COMPONENTES.md` con un catálogo en Ladle, en claro y oscuro, con todos sus estados. No uses colores literales: solo variables `--qa-*`. Pará al terminar las primitivas para que las revise.

## Pendiente de diseño

- Pantalla "Ver todo" del feed.
- Ajustes en teléfono (tema y densidad): hoja simple, no diseñada en detalle.
