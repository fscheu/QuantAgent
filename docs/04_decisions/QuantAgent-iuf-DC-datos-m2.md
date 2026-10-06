# QuantAgent-iuf — Qué datos reales usa M2 (filas V01 y V32)

Estado: **propuesta**, 2026-10-06. Plan: `docs/02_planning/QuantAgent-lp1-PL-m2-validacion-datos-reales.md` (rama `docs/QuantAgent-lp1-plan-m2`, sin mergear). Elige datos para probar estrategias, no una cartera.

## Resumen

1. **Diario:** 12 ETFs (opción B de D1), todos de Yahoo, que ya es la fuente del código.
2. **Fechas del diario:** desde el 2007-01-03 hasta el último cierre. Incluye 2008, 2020 y 2022. ~19,7 años.
3. **4h:** SPY (con horario de mercado) y BTC (24/7).
4. **Fuente 4h:** Alpaca para SPY (desde 2016, gratis con clave) y Binance para BTC (desde 2017, sin clave).
5. **Snapshot:** fuera del repo; adentro solo un manifiesto con hashes. Yahoo no deja publicarlos; Alpaca y Binance, sin verificar.

## D1. Lista de ETFs diarios

| Opción | Qué es | Cuesta |
|---|---|---|
| A | 8 núcleo: SPY, IWM, EFA, EEM, TLT, SHY, GLD, DBC | Menos corridas. Sin sectores: menos variedad de comportamiento |
| **B (recomendada)** | Las 8 de A + 4 sectores: XLK, XLE, XLF, XLU | +50% de corridas en `compare`. ~5,4 MB en CSV en total |
| C | B + QQQ, IEF, VNQ (15) | Se pasa del tope de 12. QQQ se parece mucho a XLK y SPY |

| ETF | Representa | Desde (lanzamiento del fondo; no medido) |
|---|---|---|
| SPY / IWM | Acciones de EE.UU.: empresas grandes (S&P 500) / chicas (Russell 2000) | 1993-01 / 2000-05 |
| EFA / EEM | Acciones fuera de EE.UU.: países desarrollados / emergentes | 2001-08 / 2003-04 |
| TLT / SHY | Bonos del Tesoro de EE.UU.: largos (+20 años) / cortos (1 a 3 años) | 2002-07 |
| GLD | Oro físico | 2004-11 |
| DBC | Canasta de futuros de commodities (petróleo, metales, granos) | 2006-02 |
| XLK, XLE, XLF, XLU | Sectores: tecnología, energía, financiero, servicios públicos | 1998-12 |

## D2. Rango de fechas del diario

| Opción | Desde | Cuesta |
|---|---|---|
| **A (recomendada)** | 2007-01-03, igual para los 12 | Pierde 2000–2002 en los ETFs viejos. Todos tienen la misma ventana |
| B | Toda la historia de cada ETF | Ventanas distintas: comparar activos se vuelve injusto |
| C | 2016-01-04 (lo que da Alpaca) | Sin 2008. Solo ~10 años para tres tramos |

Con A: 4.970 sesiones hasta 2026-10-05. 2007 deja ~1 año de colchón tras DBC (2006-02) para la media de 200 días.
Corte de ejemplo (lo decide V21): ajuste 2007–2018 (con 2008), prueba 2019–2022 (con 2020 y 2022), reserva 2023 en adelante.

## D3. Los 2 activos en 4h

| Opción | Activos | Cuesta |
|---|---|---|
| **A (recomendada)** | SPY + BTC | SPY ya está en el diario: se compara el mismo activo en dos horizontes |
| B | QQQ + BTC | QQQ no está en la lista diaria. Más volátil, menos comparable |
| C | SPY + ETH | ETH tiene menos historia y se mueve muy pegado a BTC |

- **SPY** opera 9:30–16:00 de Nueva York, días hábiles: 2 velas por día (9:30–13:30 y 13:30–16:00, más corta).
  Huecos de noche, fin de semana y feriados; los días de cierre a las 13:00 tienen 1 vela. ~504 velas al año.
- **BTC** opera 24/7: 6 velas iguales por día, sin huecos. ~2.190 velas al año.
- Por eso el Sharpe anualizado usa factores distintos (V34), y SPY puede saltar de cierre a apertura sin velas.

## D4. Fuente de datos por horizonte

| Fuente | Historia | Costo / límites | Clave | Licencia para repo público | Verificado |
|---|---|---|---|---|---|
| Yahoo (`yfinance`) | Diario: toda. 1h: 730 días | Gratis, sin límite publicado | No | Uso personal; no redistribuir | Parcial (buscador) |
| Alpaca acciones | Desde 2016 | Plan Basic gratis, 200 llamadas/min; SIP (todo el mercado) salvo los últimos 15 min. Sin esa restricción: USD 99/mes | Sí (gratis) | No verificado; contratos típicos prohíben redistribuir | Parcial (buscador) |
| Alpaca cripto | "Más de 5 años" | Gratis | No (con clave, más límite) | No verificado | Parcial (buscador) |
| Binance spot | BTCUSDT desde 2017-08 | Gratis; hasta 1000 velas por pedido; 4h nativo | No | Términos propios (Binance Vision); no leídos | Parcial (buscador) |
| Stooq | Diario largo | Desde 2026 pide clave (captcha) y tiene cuota diaria | Sí | No verificado | No (fuente secundaria) |
| Tiingo | Diario largo; intradía IEX | Gratis: 50 pedidos/h, 1000/día, 500 símbolos/mes | Sí | No verificado | No (fuente secundaria) |
| Polygon (Massive) | Gratis: 2 años, solo cierre diario | Gratis: 5 llamadas/min | Sí | No verificado | No (fuente secundaria) |

| Opción para 4h | Cuesta |
|---|---|
| A. Yahoo 1h agregado a 4h | Sin trabajo nuevo, pero solo ~2 años: con tres tramos la reserva queda en meses. No alcanza |
| **B (recomendada). Alpaca para SPY + Binance para BTC** | 2 adaptadores nuevos. SPY ~10,7 años, BTC ~9 años. Gratis. Alpaca es el broker futuro |
| C. Alpaca para los dos | 1 adaptador. Historia de cripto en Alpaca no verificada ("más de 5 años") |

Diario: queda Yahoo (ya integrado, gratis, sin clave, historia desde antes de 2007). En B pedir el feed SIP: el IEX (default gratis) es ~2,5% del volumen de EE.UU.; los precios son casi iguales, el volumen no sirve.

## D5. Dónde vive el snapshot

| Opción | Cuesta |
|---|---|
| A. CSV o Parquet dentro del repo | ~5,4 MB CSV / ~3,2 MB Parquet (diario) + ~3 MB (4h). Simple, CI lo lee. **Publica datos que la licencia no deja publicar** |
| B. Git LFS | Mismo problema de licencia: el repo es público y LFS también se descarga. Suma cuota de GitHub |
| **C (recomendada). Fuera del repo + manifiesto adentro** | Respeta licencias. El manifiesto (símbolo, fuente, rango, filas, sha256, fecha de descarga, versión de `yfinance`) prueba que es el mismo dato. Si se pierde la carpeta no se recrea igual: hace falta copia privada (por ejemplo Drive). El golden de V07 sobre datos reales no corre en CI. Formato: Parquet (40% menos que CSV) |

## D6. Calidad

| Problema | Por qué importa | Chequeo |
|---|---|---|
| Ajuste retroactivo | `auto_adjust=True` recalcula toda la historia en cada dividendo: bajar de nuevo cambia el hash | **Obligatorio.** Snapshot congelado; nunca se re-baja con el mismo nombre |
| Splits y dividendos | Sin ajuste, un split parece una caída. TLT y SHY pagan mucho: sin dividendos el retorno se subestima | **Obligatorio** (V04). Guardar también el cierre sin ajustar |
| Sesiones faltantes o de más | Velas que faltan o días que no existen en NYSE | **Obligatorio** (V03), contra `pandas_market_calendars` |
| Zona horaria y horario | `provider.py` borra la zona (`tz_localize(None)`) y deja hora de Nueva York; `market_calendar.py` asume UTC: corre las velas 4–5 h | **Obligatorio** en 4h: todo en UTC explícito y cero velas de SPY fuera de horario (V33) |
| OHLC incoherente o saltos raros | Máximo menor que el cierre, volumen cero, salto grande sin split | Recomendado |
| Sesgo de supervivencia | Los 12 ETFs siguen vivos hoy | Se acepta y se anota en el informe |

## Mediciones

**No hay medición real:** la red bloqueó Yahoo, Alpaca, Binance, Stooq, Tiingo y Polygon. Filas: calendario NYSE
(sin red). Tamaños: series simuladas con el mismo formato. Días faltantes contra NYSE: no medido (lo mide V03).

| Serie | Desde | Filas (sesiones o velas) | CSV | Parquet |
|---|---|---:|---:|---:|
| SPY diario, historia completa | 1993-01-29 | 8.478 | 763 KB | 462 KB |
| 12 ETFs desde 2007 (total) | 2007-01-03 | 59.640 | ~5,4 MB | ~3,2 MB |
| SPY 4h, 730 días (Yahoo) | 2024-10-07 | 995 | 105 KB | 54 KB |
| SPY 4h desde 2016 (Alpaca) | 2016-01-04 | 5.387 | 574 KB | 291 KB |
| BTC 4h, 730 días (Yahoo) | 2024-10-06 | 4.375 | 467 KB | 237 KB |
| BTC 4h desde 2017 (Binance) | 2017-08-17 | 20.017 | 2,1 MB | 1,1 MB |

## Qué no pude verificar

- Ningún dato descargado: primera fecha real, filas, faltantes y tamaño real en Yahoo. Todo es estimado.
- Historia de 1h en Yahoo hoy: el límite de 730 días sale del error que cita `provider.py`. Si acepta `4h`.
- Ninguna página oficial abrió: licencias y límites salen de resultados de buscador, no de leer la página.
- Desde cuándo tiene BTC Alpaca; términos de redistribución de Alpaca y Binance. Stooq, Tiingo, Polygon: secundarias.
- Para medir: habilitar `*.finance.yahoo.com`, `fc.yahoo.com`, `data.alpaca.markets`, `api.binance.com`.

## Fuentes

- Alpaca: [planes y feeds](https://docs.alpaca.markets/docs/about-market-data-api), [cripto sin clave](https://alpaca.markets/sdks/python/market_data.html)
- Binance: [velas](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints), [términos Vision](https://data.binance.vision/Binance_Vision-Terms_of_Use.pdf)
- Yahoo: [términos](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html), [aviso de `yfinance`](https://pypi.org/project/yfinance/)
- Secundarias: [Stooq](https://github.com/pydata/pandas-datareader/issues/1012), [Tiingo](https://www.quantstart.com/articles/evaluating-data-coverage-with-tiingo/), [Polygon/Massive](https://qveris.ai/guides/stock-api-free-comparison/?lang=en)
