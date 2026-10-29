# Inventario de Koyfin como referencia para M-web

- **Fecha:** 2026-10-29. **Tipo:** investigación, sin código ni tickets.
- **Pedido de Fede:** Koyfin tiene muchos gráficos y vistas de análisis por sector, país y macro; es "todo lo que querría como app web". Ver qué ofrece y qué nos sirve.
- **Fuente:** centro de ayuda público de Koyfin (`koyfin.com/help/`) y su página de dashboards de mercado. `robots.txt` permite todo; se leyeron unas 15 páginas puntuales, sin descarga masiva. Koyfin no ofrece API ni permite reusar sus gráficos; esto es solo una referencia de qué hacen, no una fuente de datos.
- **Límite del inventario:** la ayuda describe poco los dashboards de sector y país (los ven dentro de la app, que requiere cuenta). Lo que no está en una página leída va marcado como **no confirmado**. Conviene que Fede mire la versión gratuita de Koyfin si quiere ver esas pantallas antes del diseño visual (W3).

Relacionado: `2026-10-28_roadmap_post_m5.md` §3 (M-web), `PLAN-POST-M5.md`.

---

## 1. Qué es Koyfin, en una frase

Un terminal financiero en el navegador: una barra de comandos para ir a cualquier activo, un gráfico histórico muy flexible, tablas con miles de columnas, un tablero propio que cada usuario arma con widgets, y una docena de dashboards de mercado ya armados (sectores de EE. UU., países, factores, tasas, monedas, materias primas, crédito, macro mundial). Datos con licencia de S&P Capital IQ (acciones), Morningstar (fondos), FRED y Trading Economics (macro), True FX (monedas) y Polygon (cripto).

## 2. Dashboards de mercado ya armados

Es la parte que más le interesa a Fede. Lista de la página oficial de dashboards:

| Dashboard | Qué muestra | Confianza |
|---|---|---|
| Today's Markets | Resumen de acciones, crédito, monedas y materias primas | Nombre confirmado |
| World Equity Indices | Índices bursátiles del mundo | Nombre confirmado |
| US Sectors | Los sectores de EE. UU. | Nombre confirmado; el contenido (¿heatmap, tabla de ETFs sectoriales?) **no confirmado** |
| Countries | Rendimiento por país | Nombre confirmado; contenido **no confirmado** |
| Factors | Factores (valor, momentum, calidad, etc.) | Nombre confirmado |
| Global Yields | Compara tasas de bonos entre países | Confirmado en la ayuda |
| Global Yield Curves | Curvas de tasas por país | Confirmado en la ayuda |
| Currencies | Monedas | Confirmado en la ayuda |
| World Economics | Indicadores económicos clave por país | Nombre confirmado |
| Commodities | Materias primas | Nombre confirmado |
| Corporate Credit | Crédito corporativo | Nombre confirmado |
| Fixed Income Factors | Factores de renta fija | Nombre confirmado |
| Recent IPOs | Salidas a bolsa recientes | Nombre confirmado |

Estructura común (confirmada en la ayuda de bonos y monedas): cada dashboard junta tres widgets enlazados.

1. **Tabla** con los tickers, columnas elegibles, agrupable, filtrable por país, exportable a CSV (menos fundamentales y estimaciones).
2. **Gráfico**: al tildar filas de la tabla se dibujan en el gráfico para comparar.
3. **Matriz**: una vista resumen propia de cada dashboard para ver el panorama global de un vistazo (qué es exactamente la matriz, p. ej. si es un mapa de calor, **no está confirmado**).

La idea de fondo, que sí es copiable: **tabla de series + gráfico comparativo que se llena con lo que se tilda**.

## 3. Tipos de gráfico y herramientas de análisis

| Función | Descripción | Fuente |
|---|---|---|
| Gráfico histórico (G) | Línea, velas, barras OHLC, área; mínimo diario (sin velas intradía) | ayuda |
| Varias series | Cualquier activo más métricas: precio, medias móviles, múltiplos, fundamentales | ayuda |
| Ejes | Escala log o lineal; un eje por serie o un eje compartido por unidades | ayuda |
| Superposiciones | Medias móviles con ventana ajustable, correlación móvil, volumen | ayuda |
| Rendimiento acumulado | "Performance Graph": % de cambio desde el primer día del rango, para comparar activos | ayuda |
| Relativo A/B | Fuerza relativa (cociente de precios, equivale a largo A / corto B rebalanceado a diario) y spread relativo (%A − %B, sin rebalanceo) | ayuda |
| Tabla de los datos del gráfico | "Show Table" | ayuda |
| Plantillas de gráfico | Guardar series y formato y reaplicarlos a otro ticker | ayuda |
| Anotaciones | Dibujos sobre el gráfico, guardados por plantilla y ticker | ayuda |
| Series propias y fórmulas | Custom Formulas, Custom Data Series | ayuda (solo menciones) |
| Dispersión (scatter) | En listas, dashboards y carteras; con recta de regresión y R² | ayuda |
| Curvas de tasas | Por país, por plazo | ayuda |
| Contribución | Peso × rendimiento de cada componente de un ETF, por sector y subsector; la suma da el rendimiento del ETF | ayuda (ETF Holdings) |
| Exposición de cartera | Torta por sector, región/país, calidad crediticia, plazo; barra contra benchmark (sobre/subponderado) | ayuda (Portfolio Exposures) |
| Percentiles | Puntaje 0–100 de una empresa contra su grupo (ej. tecnología de EE. UU.) en valuación, márgenes, rendimiento | ayuda |
| Valuación de ETF | P/E trailing y forward del ETF (promedio armónico ponderado), P/S, EV/EBITDA; solo ETFs de EE. UU. | ayuda |
| Calendario económico | Por país, con consenso y valor previo; cada evento abre su gráfico; selector de zona horaria | ayuda |
| Calendario de resultados | Con sorpresa contra consenso | ayuda |
| Screener | Más de 5.900 filtros sobre más de 100.000 valores | ayuda |
| Carteras y modelos | P/L por lote, atribución, tabla de rebalanceo, matriz de tenencias | ayuda |

No está en la ayuda leída (no afirmar que existe): mapa de calor de sectores, sombreado de recesiones, mapa del mundo.

## 4. Mapa contra QuantAgent

Pantallas previstas hoy (`roadmap_post_m5` §3): Cuenta y Embudo primero, solo lectura; Inicio sale de esas dos; Ficha y Mercado después. Datos: FRED/ALFRED (macro, ya se lee en `macro_vivo.py`), snapshots Parquet con manifiesto y DuckDB (M-datos), Sharadar solo si se decide pagar (L25).

Leyenda de datos: **Tenemos** = ya existe o está decidido y barato. **Falta** = hay que bajarlo o armarlo con FRED/Yahoo/ETFs. **Sharadar** = depende del pago.

| Función Koyfin | Equivalente en QuantAgent | Pantalla | Datos | Esfuerzo | Fase sugerida |
|---|---|---|---|---|---|
| Gráfico histórico línea/velas de un activo | Gráfico de velas con `lightweight-charts` | Mercado | Tenemos (precios en Parquet) | Bajo | 2 |
| Comparar varios activos tildando una tabla | Tabla + gráfico de rendimiento acumulado | Mercado | Tenemos | Bajo | 2 |
| Rendimiento acumulado y relativo A/B | Misma pantalla; cálculo en el backend | Mercado | Tenemos | Bajo | 2 |
| US Sectors (los 11 ETFs sectoriales: tabla con 1d/1s/1m/YTD y gráfico) | Tabla de ETFs sectoriales; es el mismo universo que usa el motor top-down de M5 | Mercado (y Inicio) | Tenemos para ETFs; Falta si se quiere más historia | Bajo | 2 |
| Mapa de calor de sectores | Cuadrícula coloreada con los mismos números | Mercado | Tenemos | Medio | 3 |
| Contribución por sector a un ETF/cartera | Cuánto aportó cada sector a la cuenta o a SPY | Cuenta | Tenemos para las nuestras; ETF requiere pesos históricos (Falta; Koyfin mismo los estima) | Medio | 3 |
| Exposición de cartera por sector contra benchmark | Barras sobre/subponderado de la cuenta contra SPY | Cuenta | Tenemos (posiciones) + mapa símbolo→sector (Falta, o Sharadar) | Medio | 2–3 |
| Countries (rendimiento por país) | Tabla de ETFs de país (EWZ, EWJ, EEM, etc.) | Mercado | Falta (bajar precios de esos ETFs) | Bajo | 3 |
| World Economics (indicadores por país) | Series macro de EE. UU. con glosario por término | Mercado | Tenemos (FRED); otros países Falta (FRED trae varios, calidad dispareja) | Medio | 2 (EE. UU.), 4 (resto) |
| Global Yields / Yield Curves | Curva de tasas de EE. UU. (2a, 10a, spread 10a−2a) | Mercado | Tenemos (FRED: DGS2, DGS10, T10Y2Y) | Bajo | 2 |
| Curvas de otros países | — | Mercado | Falta, calidad dudosa en datos gratuitos | Alto | No |
| Sombreado de recesiones en gráficos macro | Bandas con la serie USREC de FRED | Mercado | Tenemos (FRED) | Bajo | 2 |
| Calendario económico | Próximos eventos macro en Inicio | Inicio | Falta (FRED publica fechas de release; consenso no) | Medio | 3 |
| Factors | Retornos de factores | Mercado | Falta; ETFs de factores o Fama-French como alternativa | Medio | 4 |
| Commodities / Currencies | Tabla de futuros/monedas | Mercado | Falta | Bajo | 4 |
| Percentiles, screener, valuación de ETF | Lo hace el screener de M5 | (otra pantalla) | Sharadar | Alto | No por ahora |
| Plantillas de gráfico, dashboards armables, anotaciones | — | — | — | Alto | No |
| Noticias, transcripciones, insiders, integraciones con brokers | — | — | — | Alto | No |
| Reportes, compartir, equipos, app móvil nativa | La web responsiva y Tailscale alcanzan | — | — | — | No |

## 5. Lista corta sugerida (sin inflar M-web)

La revisión crítica pidió no sobrediseñar: la primera entrega sigue siendo **solo Cuenta y Embudo**. Nada de esto la cambia. Para la fase de Mercado, que ya estaba prevista, propongo copiar de Koyfin solo cuatro ideas, todas con datos que tenemos o casi:

1. **Tabla + gráfico que se llena al tildar filas**, con rango de fechas y modo "rendimiento acumulado". Es el patrón de todo Koyfin y se construye una vez.
2. **Dashboard "Sectores de EE. UU."**: los 11 ETFs sectoriales con rendimientos 1d/1s/1m/YTD. Es el mismo universo del motor top-down, así que la pantalla también sirve para revisar qué sectores eligió el motor. Un mapa de calor viene después, con los mismos datos.
3. **Dashboard "Macro EE. UU."**: series FRED que ya usamos (tasas, spread 10a−2a, inflación, desempleo, VIX) con sombreado de recesiones y el glosario por término que Fede ya pidió.
4. **Cuenta contra benchmark por sector**: sobre/subponderación de la cuenta contra SPY, que es la versión propia de "Portfolio Exposures".

Lo demás (países, otros países en macro, factores, calendario con consenso, screener, percentiles) queda anotado para cuando exista M-datos y Sharadar, sin ticket. Lo que Koyfin hace pero no vale la pena copiar: plantillas de gráfico, dashboards armables por el usuario, anotaciones, noticias, transcripciones, integraciones con brokers. Son meses de trabajo y no ayudan a decidir si una estrategia sirve.

## 6. Riesgos y cosas a tener en cuenta

- **Los datos gratuitos tienen límites.** Koyfin paga licencias de varios proveedores; nosotros tendríamos FRED (gratis, bueno para EE. UU.) y precios de ETFs. La parte "países" y "otros países en macro" es donde más se nota.
- **Licencia de Sharadar:** si se deja de pagar hay que borrar los datos (ver roadmap §4). Una pantalla que dependa de ellos desaparece con la suscripción.
- **No copiar diseño ni marca.** Es referencia de funciones, no de aspecto; el diseño visual se hace en las sesiones con Fede (W3).
- **Para cerrar las dudas** de §2 (qué muestran Countries, US Sectors y la matriz), alguien tiene que abrir la versión gratuita de Koyfin. La ayuda no lo explica.

## 7. Fuentes

- Centro de ayuda: `koyfin.com/help/` y sus artículos (Getting Started, Historical Graph, Relative Performance, My Dashboards, Global Bonds Yield Curves & FX, Data Overview, ETF Holdings, ETF Exposure, ETF Valuation Metrics, Portfolio Exposures, Global Equities, Economic Calendars, FAQ de fuentes de datos y API).
- Página de producto: `koyfin.com/features/market-dashboards/`.
