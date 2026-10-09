# QuantAgent-iuf — Qué datos reales usa M2 (filas V01 y V32)

Estado: **propuesta**, 2026-10-06; mediciones reales 2026-10-08. Plan: `docs/02_planning/QuantAgent-lp1-PL-m2-validacion-datos-reales.md` (rama `docs/QuantAgent-lp1-plan-m2`, sin mergear). Elige datos para probar estrategias, no una cartera.

## Resumen

1. **Diario:** 12 ETFs (opción B de D1), todos de Yahoo, que ya es la fuente del código.
2. **Fechas del diario:** desde el 2007-01-03 hasta el último cierre. Incluye 2008, 2020 y 2022. ~19,8 años.
3. **4h:** SPY (con horario de mercado) y BTC (24/7).
4. **Fuente 4h:** Alpaca para SPY (desde 2016, gratis con clave; sin medir) y Binance para BTC (desde 2017-08, sin clave; medido).
5. **Snapshot:** fuera del repo; adentro solo un manifiesto con hashes. Yahoo no deja publicarlos; Binance Vision, solo sin fin comercial; Alpaca, sin verificar.

## D1. Lista de ETFs diarios

| Opción | Qué es | Cuesta |
|---|---|---|
| A | 8 núcleo: SPY, IWM, EFA, EEM, TLT, SHY, GLD, DBC | Menos corridas. Sin sectores: menos variedad de comportamiento |
| **B (recomendada)** | Las 8 de A + 4 sectores: XLK, XLE, XLF, XLU | +50% de corridas en `compare`. 5,5 MB en CSV en total |
| C | B + QQQ, IEF, VNQ (15) | Se pasa del tope de 12. QQQ se parece mucho a XLK y SPY |

| ETF | Representa | Desde (primera fila en Yahoo, medido) |
|---|---|---|
| SPY / IWM | Acciones de EE.UU.: empresas grandes (S&P 500) / chicas (Russell 2000) | 1993-01-29 / 2000-05-26 |
| EFA / EEM | Acciones fuera de EE.UU.: países desarrollados / emergentes | 2001-08-27 / 2003-04-14 |
| TLT / SHY | Bonos del Tesoro de EE.UU.: largos (+20 años) / cortos (1 a 3 años) | 2002-07-30 |
| GLD | Oro físico | 2004-11-18 |
| DBC | Canasta de futuros de commodities (petróleo, metales, granos) | 2006-02-06 |
| XLK, XLE, XLF, XLU | Sectores: tecnología, energía, financiero, servicios públicos | 1998-12-22 |

## D2. Rango de fechas del diario

| Opción | Desde | Cuesta |
|---|---|---|
| **A (recomendada)** | 2007-01-03, igual para los 12 | Pierde 2000–2002 en los ETFs viejos. Todos tienen la misma ventana |
| B | Toda la historia de cada ETF | Ventanas distintas: comparar activos se vuelve injusto |
| C | 2016-01-04 (lo que da Alpaca) | Sin 2008. Solo ~10 años para tres tramos |

Con A: 4.973 sesiones por ETF hasta 2026-10-08 (la última puede ser parcial). 2007 deja ~1 año de colchón tras DBC (2006-02) para la media de 200 días.
Corte de ejemplo (lo decide V21): ajuste 2007–2018 (con 2008), prueba 2019–2022 (con 2020 y 2022), reserva 2023 en adelante.

## D3. Los 2 activos en 4h

| Opción | Activos | Cuesta |
|---|---|---|
| **A (recomendada)** | SPY + BTC | SPY ya está en el diario: se compara el mismo activo en dos horizontes |
| B | QQQ + BTC | QQQ no está en la lista diaria. Más volátil, menos comparable |
| C | SPY + ETH | ETH tiene menos historia y se mueve muy pegado a BTC |

- **SPY** opera 9:30–16:00 de Nueva York, días hábiles: 2 velas por día (9:30 y 13:30, la segunda más corta). Huecos de noche, fin de semana y feriados; los días de cierre a las 13:00 tienen 1 vela. ~504 velas al año. Medido en Yahoo: 720 días con 2 velas y 10 con 1, todas a las 9:30 o 13:30 de Nueva York.
- **BTC** opera 24/7: 6 velas iguales por día (~2.190 al año). Binance no está libre de huecos: faltan 16 velas de 20.037 (mantenimientos). El Sharpe anualizado usa otro factor (V34).

## D4. Fuente de datos por horizonte

| Fuente | Historia | Costo / límites | Clave | Licencia para repo público | Verificado |
|---|---|---|---|---|---|
| Yahoo (`yfinance`) | Diario: toda. 1h y 4h: ~2 años (`max`) | Gratis, sin límite publicado | No | Términos: prohíben extraer datos con herramientas automáticas y reusar con fin comercial; `yfinance`: "uso personal" | Sí (términos y medición) |
| Alpaca acciones | Desde 2016 | Basic gratis, 200 llamadas/min; SIP completo solo con los últimos 15 min restringidos. Sin esa restricción: USD 99/mes (Algo Trader Plus) | Sí (gratis) | La página de planes no dice nada. Sin verificar | Sí (página oficial); sin medir |
| Alpaca cripto | "Más de 5 años" | Gratis | No (con clave, más límite) | Sin verificar | Sí (página oficial); sin medir |
| Binance spot | BTCUSDT 4h desde 2017-08-17 | Gratis; hasta 1000 velas por pedido (500 por defecto); 4h nativo; UTC | No | Datasets de Vision: CC BY-NC-SA 4.0, sin fin comercial, con atribución y misma licencia. Los términos de la API REST no se leyeron | Sí (spec y medición) |
| Stooq | Diario largo | Desde 2026 pide clave (captcha) y tiene cuota diaria | Sí | No verificado | No (fuente secundaria) |
| Tiingo | Diario largo; intradía IEX | Gratis: 50 pedidos/h, 1000/día, 500 símbolos/mes | Sí | No verificado | No (fuente secundaria) |
| Polygon (Massive) | Gratis: 2 años, solo cierre diario | Gratis: 5 llamadas/min | Sí | No verificado | No (fuente secundaria) |
| OpenBB | No es fuente: envuelve Yahoo, Polygon, Tiingo, FMP, Alpha Vantage y otros. No suma historia propia | Gratis (código AGPL-3.0) | La del proveedor | La del proveedor | Parcial (buscador); el link oficial da 404. Lo estudia `QuantAgent-6ie` |

| Opción para 4h | Cuesta |
|---|---|
| A. Yahoo 4h nativo | Sin trabajo nuevo, pero ~2 años (2,9 con `period="730d"`): con tres tramos la reserva queda en meses. No alcanza |
| **B (recomendada). Alpaca para SPY + Binance para BTC** | 2 adaptadores nuevos. SPY ~10,7 años (sin medir), BTC 9,1 años (medido). Gratis. Alpaca es el broker futuro |
| C. Alpaca para los dos | 1 adaptador. Historia de cripto en Alpaca no verificada ("más de 5 años") |

Diario: queda Yahoo (ya integrado, gratis, sin clave, historia desde antes de 2007). En B pedir el feed SIP: el IEX (default gratis) es ~2,5% del volumen de EE.UU.; los precios son casi iguales, el volumen no sirve. Binance: `api.binance.com` da 451 (región) desde el entorno de medición. `data-api.binance.vision` respondió y sirve las mismas velas. El adaptador debe permitir cambiar el host.

## D5. Dónde vive el snapshot

| Opción | Cuesta |
|---|---|
| A. CSV o Parquet dentro del repo | 5,5 MB CSV / 2,6 MB Parquet (diario) + ~2 MB (4h, con BTC de Binance). Simple, CI lo lee. **Publica datos que la licencia de Yahoo no deja publicar** |
| B. Git LFS | Mismo problema de licencia: el repo es público y LFS también se descarga. Suma cuota de GitHub |
| **C (recomendada). Fuera del repo + manifiesto adentro** | Respeta licencias. El manifiesto (símbolo, fuente, rango, filas, sha256, fecha de descarga, versión de `yfinance`) prueba que es el mismo dato. Si se pierde la carpeta no se recrea igual: hace falta copia privada (por ejemplo Drive). El golden de V07 sobre datos reales no corre en CI. Formato: Parquet (35–55% menos que CSV) |

## D6. Calidad

| Problema | Por qué importa | Chequeo |
|---|---|---|
| Ajuste retroactivo | `auto_adjust=True` recalcula toda la historia en cada dividendo: bajar de nuevo cambia el hash | **Obligatorio.** Snapshot congelado; nunca se re-baja con el mismo nombre |
| Splits y dividendos | Sin ajuste, un split parece una caída. TLT y SHY pagan mucho: sin dividendos el retorno se subestima | **Obligatorio** (V04). Guardar también el cierre sin ajustar |
| Sesiones faltantes o de más | Velas que faltan o días que no existen en NYSE | **Obligatorio** (V03), contra `pandas_market_calendars`. Yahoo diario: 0 de 15 ETFs; Binance 4h: 16 velas |
| Zona horaria y horario | `provider.py` borra la zona (`tz_localize(None)`) y deja hora de Nueva York; `market_calendar.py` asume UTC: corre las velas 4–5 h. Yahoo 4h de SPY viene en `America/New_York`; BTC en UTC | **Obligatorio** en 4h: todo en UTC explícito y cero velas de SPY fuera de horario (V33) |
| OHLC incoherente o saltos raros | Medido: 1 a 29 filas por ETF con máximo o mínimo fuera de rango, hasta 0,06% (SPY: 29). Parece redondeo del ajuste. Volumen cero: 0 | Recomendado |
| Sesgo de supervivencia | Los 12 ETFs siguen vivos hoy | Se acepta y se anota en el informe |

## Mediciones

Fecha 2026-10-08. Yahoo: `yfinance` 1.7.0, script del PR #28. Binance: `data-api.binance.vision`, velas 4h de a 1000. Días faltantes contra NYSE (`pandas_market_calendars`): **0 en los 15 ETFs diarios**, y 0 días de más. Alpaca (SPY 4h SIP y BTC/USD 4h) no se midió: sin claves.

| Serie | Desde | Filas (sesiones o velas) | Faltan | CSV | Parquet |
|---|---|---:|---:|---:|---:|
| SPY diario, historia completa | 1993-01-29 | 8.481 | 0 | 892 KB | 430 KB |
| 12 ETFs desde 2007 (total) | 2007-01-03 | 59.676 | 0 | 5,5 MB | 2,6 MB |
| SPY 4h, `730d` (Yahoo) | 2023-11-09 | 1.450 | n/d | 146 KB | 63 KB |
| SPY 1h, `max` (Yahoo) | 2024-10-08 | 3.478 | n/d | 349 KB | 141 KB |
| BTC 4h, `730d` (Yahoo) | 2024-10-09 | 4.372 | n/d | 377 KB | 185 KB |
| BTC 4h desde 2017 (Binance) | 2017-08-17 | 20.021 | 16 | 1,4 MB | 886 KB |

`period="730d"` en Yahoo devolvió 730 sesiones (2,9 años de 1h), no 730 días. Con `max` salen 2 años. Los datos reales no cambian ninguna recomendación. Se corrigieron números (Parquet ahorra más de lo estimado) y se sumó el host espejo de Binance.

## Qué no pude verificar

- Alpaca: `APCA_API_KEY_ID` y `APCA_API_SECRET_KEY` no existen en el entorno. SPY 4h con SIP y BTC/USD 4h: primera vela y cantidad sin medir.
- Licencia y redistribución de Alpaca: la página de planes no las menciona. Tampoco desde cuándo hay cripto.
- Binance: se leyeron los términos de los datasets Vision, no los de la API REST. `api.binance.com` no se pudo medir (451).
- Yahoo: sus términos prohíben extraer datos con herramientas automáticas sin permiso. No se consultó si `yfinance` entra.
- OpenBB (link 404), Stooq, Tiingo y Polygon: no medidos ni leídos en la fuente oficial.

## Fuentes

- Alpaca: [planes y feeds](https://docs.alpaca.markets/docs/about-market-data-api), [cripto sin clave](https://alpaca.markets/sdks/python/market_data.html)
- Binance: [velas](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md) (la página de `developers.binance.com` no abre con `curl`), [términos Vision](https://data.binance.vision/Binance_Vision-Terms_of_Use.pdf)
- Yahoo: [términos](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html), [aviso de `yfinance`](https://pypi.org/project/yfinance/)
- OpenBB: [extensiones de datos](https://docs.openbb.co/platform/usage/extensions/data_extensions) (404 al 2026-10-08). Secundarias: [Stooq](https://github.com/pydata/pandas-datareader/issues/1012), [Tiingo](https://www.quantstart.com/articles/evaluating-data-coverage-with-tiingo/), [Polygon/Massive](https://qveris.ai/guides/stock-api-free-comparison/?lang=en)
