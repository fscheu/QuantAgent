# QuantAgent-fnpr.4.8 · Cobertura del screener M5-32 (2010-03 a 2018-12)

Medición, sin cambios de producción. Reproducir: `python scripts/m5/cobertura_screener.py --control` (~11 min, 4 procesos; escribe el CSV de esta carpeta).

Método: 36 fechas (última sesión de cada trimestre), miembros punto-en-el-tiempo de `sp500-2026-10` agrupados por sector (`ETF_SECTOR`). Cada miembro cae en una sola causa, en este orden: `sin_cik` (no está en `fundamentales-sp500-2026-10`), `sin_precio` (no está en `acciones-sp500-2026-10b` o rs3 NaN), `faltan_metricas` (2 o más de las 6 métricas del screener, rs3/rs6 incluidas) o `elegible`. Cobertura = elegibles / miembros. La suma de causas = miembros en cada fila. Las funciones `_retorno` y `_metrica_fund` son las del screener.

## Cobertura anual (% promedio de las 4 fechas del año)

| sector                 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 |
|:-----------------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| Communication Services |    - |    - |    - |    - |    - |    - |    - |    - |   29 |
| Consumer Discretionary |   15 |   35 |   43 |   44 |   47 |   50 |   52 |   36 |   24 |
| Consumer Staples       |   15 |   27 |   29 |   26 |   28 |   36 |   39 |   28 |   16 |
| Energy                 |   19 |   24 |   26 |   31 |   31 |   28 |   33 |   28 |   24 |
| Financials             |   22 |   37 |   46 |   50 |   52 |   60 |   61 |   62 |   61 |
| Health Care            |   28 |   37 |   43 |   44 |   46 |   47 |   53 |   44 |   28 |
| Industrials            |   34 |   47 |   59 |   63 |   65 |   68 |   68 |   48 |   34 |
| Information Technology |   24 |   41 |   54 |   55 |   58 |   55 |   58 |   42 |   30 |
| Materials              |   14 |   19 |   28 |   42 |   43 |   47 |   55 |   39 |   19 |
| Real Estate            |    - |    - |    - |    - |    - |   - |   56 |   48 |   43 |
| Utilities              |   26 |   45 |   59 |   60 |   62 |   68 |   75 |   67 |   62 |

(`-`: el sector no existe en esa fecha con la clasificación GICS vigente; Communication Services solo desde 2018-09, Real Estate desde 2016-09.)

**Sector-años bajo 60%: 71 de 85.** Solo llegan a 60% o más: Financials 2015-2018 (60, 61, 62, 61), Industrials 2013-2016 (63, 65, 68, 68) y Utilities 2013-2018 (60, 62, 68, 75, 67, 62). Industrials 2012 y Utilities 2012 quedan en 59. Ningún sector llega a 60% en 2010-2011, y en 2018 solo Financials y Utilities. Detalle por fecha en el CSV.

## Causas

Sobre todos los miembros-fecha: elegible 43%, faltan_metricas 32%, sin_cik 25%, sin_precio 0,6%. Por año (% de miembros):

| año | elegible | faltan_metricas | sin_cik | sin_precio |
|---:|---:|---:|---:|---:|
| 2010 | 22 | 45 | 32 | 1 |
| 2011 | 36 | 33 | 30 | 1 |
| 2012 | 44 | 27 | 28 | 1 |
| 2013 | 47 | 26 | 26 | 0 |
| 2014 | 49 | 25 | 26 | 0 |
| 2015 | 52 | 23 | 24 | 0 |
| 2016 | 56 | 23 | 21 | 0 |
| 2017 | 44 | 37 | 19 | 0 |
| 2018 | 34 | 47 | 17 | 1 |

Causa dominante por sector (miembro-fecha excluidos): `sin_cik` en Energy (576 vs 441) y Health Care (635 vs 547), empate en IT (675 vs 679) y Communication Services (22 vs 11); `faltan_metricas` en el resto. `sin_precio` es marginal (Energy 27, Health Care 20 son los máximos).

Tres métricas que más faltan por sector (entre los excluidos por `faltan_metricas`):

| sector | 1ª | 2ª | 3ª |
|:--|:--|:--|:--|
| Communication Services | rs6 (8) | crecimiento_eps (6) | margen_op_yoy (5) |
| Consumer Discretionary | margen_op_yoy (984) | aceleracion_ingresos (878) | roic_4t (488) |
| Consumer Staples | margen_op_yoy (675) | aceleracion_ingresos (642) | crecimiento_eps (316) |
| Energy | margen_op_yoy (431) | roic_4t (314) | aceleracion_ingresos (313) |
| Financials | margen_pretax_yoy (907) | aceleracion_ingresos (757) | crecimiento_eps (447) |
| Health Care | margen_op_yoy (527) | aceleracion_ingresos (479) | roic_4t (251) |
| Industrials | margen_op_yoy (588) | aceleracion_ingresos (517) | roic_4t (297) |
| Information Technology | margen_op_yoy (632) | aceleracion_ingresos (614) | crecimiento_eps (372) |
| Materials | margen_op_yoy (357) | aceleracion_ingresos (315) | roic_4t (141) |
| Real Estate | margen_op_yoy (120) | aceleracion_ingresos (110) | roic_4t (59) |
| Utilities | margen_op_yoy (291) | aceleracion_ingresos (279) | roic_4t (165) |

margen_op_yoy, margen_pretax_yoy y aceleracion_ingresos son las que necesitan la serie `ingresos` completa de 6 a 8 trimestres seguidos. Eso apunta al hallazgo siguiente.

## Hallazgo: la caída de 2017-2018 es un bug de extracción, no falta de datos

`extraer_hechos` descarta un hecho si otro tag de mayor prioridad ya cubrió el mismo `(start, end)`. Cuando una empresa cambia de tag (adopción de ASC 606: `SalesRevenueNet` → `Revenues` / `RevenueFromContract...`), los 10-K de 2018 reportan los trimestres comparativos con el tag nuevo y la fecha `filed` nueva, y el hecho original del tag viejo (presentado en 2017) se tira. Punto-en-el-tiempo, a 2018-03-29 esos trimestres todavía "no existen" y la ventana de ingresos tiene huecos.

Caso AAPL, trimestre 2017-01-01 → 2017-04-01 (valores de companyfacts): `SalesRevenueNet` 52.896M, 10-Q filed 2017-05-03; `Revenues` 52.896M, 10-K filed 2018-11-05. En el snapshot solo queda la segunda. Resultado: AAPL, MSFT, AMZN, INTC (mismo patrón de métricas faltantes; AAPL verificado contra companyfacts) pierden margen_op_yoy y aceleracion_ingresos en 2017-2018 (cobertura de faltan_metricas: 23% en 2016 a 47% en 2018). Corregir la deduplicación para que no pise hechos con `filed` anterior es el cambio de mayor retorno para H3 (no implementado acá).

## 10 grandes excluidas (volumen en dólares medio de las últimas 63 sesiones del snapshot, última fecha en que cada una queda excluida)

| ticker | sector | fecha | causa | faltan |
|:--|:--|:--|:--|:--|
| AMZN | Consumer Discretionary | 2018-12-31 | faltan_metricas | margen_op_yoy, aceleracion_ingresos |
| AAPL | Information Technology | 2018-09-28 | faltan_metricas | margen_op_yoy, aceleracion_ingresos |
| MSFT | Information Technology | 2018-06-29 | faltan_metricas | margen_op_yoy, aceleracion_ingresos |
| FB | Communication Services | 2018-12-31 | sin_cik | (ticker hoy META) |
| NVDA | Information Technology | 2010-03-31 | faltan_metricas | roic_4t, margen_op_yoy, aceleracion_ingresos, crecimiento_eps |
| GOOGL | Communication Services | 2018-09-28 | faltan_metricas | crecimiento_eps, rs6 |
| AMD | Information Technology | 2018-12-31 | faltan_metricas | margen_op_yoy, aceleracion_ingresos, crecimiento_eps |
| GOOG | Communication Services | 2018-09-28 | faltan_metricas | crecimiento_eps, rs6 |
| BAC | Financials | 2011-12-30 | faltan_metricas | margen_pretax_yoy, crecimiento_eps |
| JPM | Financials | 2010-03-31 | faltan_metricas | roe_4t, margen_pretax_yoy, aceleracion_ingresos, crecimiento_eps |

GOOGL/GOOG: rs6 falta porque XLC arranca en 2018-06 (sin 126 sesiones de historia a 2018-09); crecimiento_eps no se diagnosticó contra los tags de Alphabet.

## Diagnóstico sin_cik (287 tickers en el manifiesto; 210 distintos en la medición)

`mapa_tickers_cik()` usa `company_tickers.json`, que solo trae tickers vigentes. Cinco casos contra `data.sec.gov`:

| ticker | CIK real | qué pasó | submissions / companyfacts |
|:--|--:|:--|:--|
| ANTM | 1156039 | renombrada (Anthem → Elevance, hoy ELV) | submissions OK, `formerNames` incluye Anthem y WellPoint |
| ABC | 1140859 | renombrada (AmerisourceBergen → Cencora, hoy COR) | submissions y companyfacts OK |
| ATVI | 718877 | deslistada (compra de Microsoft, 2023) | submissions y companyfacts OK hasta 2023-06 |
| AET | 1122304 | deslistada (CVS, 2018) | submissions OK, último filing 2019-02 |
| ALXN | 899866 | deslistada (AstraZeneca, 2021) | submissions y companyfacts OK hasta 2021-03 |

Conclusión: el CIK existe en los cinco casos y los datos históricos están en la API; falta solo el mapa ticker histórico → CIK. Hay dos grupos: renombrados / cambio de ticker (ANTM, ABC, FB, BK→BNY, DISCA→WBD, WBA...) y deslistados por fusión (ATVI, AET, ALXN, CELG...). Los renombrados se resuelven con el ticker actual; los deslistados no están en `company_tickers.json`.

Fuente más simple propuesta: un CSV versionado `ticker, cik` para los 287 tickers del manifiesto, armado una vez por nombre de empresa. `sp500.parquet` / el CSV de salidos del snapshot de composición ya traen el nombre de la empresa; se busca en `https://www.sec.gov/Archives/edgar/cik-lookup-data.txt` (nombre → CIK, incluye nombres viejos y empresas deslistadas; se verificó que responde 200) y se revisa a mano los casos ambiguos, y `crear_snapshot_fundamentales` lo usa como segundo mapa junto a `company_tickers.json` (patrón de `CIK_PREVIOS`). Verificado contra submissions que el CIK devuelve el nombre correcto. No se implementó.

## Diagnóstico de métricas faltantes (INTC, V, BK)

| ticker | qué falta | causa |
|:--|:--|:--|
| INTC | margen_op_yoy y aceleracion_ingresos desde 2017-2018 | no es tag no contemplado: `Revenues` / `RevenueFromContract...` están en `CONCEPTOS_FALLBACK`; es el bug de deduplicación entre tags de arriba |
| V | roic_4t desde 2012 y crecimiento_eps siempre | sí, tags no contemplados: `StockholdersEquity` solo llega a 2011-09 y después Visa usa `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest`; no publica `EarningsPerShareDiluted` consolidado (EPS por clase de acciones) |
| BK | todo: es `sin_cik` | el ticker cambió a BNY y no está en `company_tickers.json`. Con el CIK (1390777) los tags sí existen salvo el resultado antes de impuestos: `IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest` termina en 2016-06 y después solo hay `...Domestic` y `...Foreign` (habría que sumarlos), así que margen_pretax_yoy faltaría desde 2016 |

## Control cruzado, 2018-06-29, XLK + XLF

142 miembros: 59 elegibles, 52 faltan_metricas, 28 sin_cik, 3 sin_precio. Se llamó a `candidatos_semanales(2018-06-29, [XLK, XLF])` con paneles propios y se contó el largo de la serie con la que calcula el cuartil de FR (= miembros que pasan el filtro de datos): 59, igual que los elegibles. Excluidos por datos: 83 (la revisión de M5-32b daba 84; con el snapshot nuevo se recupera un ticker). Miembros con rs3 NaN: 22, igual que M5-32b; 19 de ellos ya caen antes por `sin_cik` y por eso `sin_precio` muestra 3.
