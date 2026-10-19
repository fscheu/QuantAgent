# QuantAgent-7nle — De dónde salen las velas al operar en paper (M3)

Estado: **decidido** por Fede el 2026-10-12 (`R:` del PR #116); las decisiones están al final, en "Decisiones de Fede". Propuesta del 2026-10-11. No cambia código. Relacionado: [iuf (datos de M2)](./QuantAgent-iuf-DC-datos-m2.md), fila V39a de `docs/02_planning/QuantAgent-lp1-PL-m2-validacion-datos-reales.md` (§11, línea 303). Alcance: solo la cuenta gratuita de Alpaca; sin suscripciones ni depósito mínimo. El backtest y sus snapshots no se tocan.

## Resumen

1. **Fuente:** Alpaca (feed SIP, pedido con `end` ≥ 15 min atrás) para acciones/ETFs; Yahoo queda como respaldo y para el desarrollo sin claves. Cripto en Alpaca: sin verificar.
2. **Arquitectura:** interfaz `CandleSource` de 3 métodos con `YahooSource` y `AlpacaSource`; `DataProvider` conserva tabla, huecos y modo sin conexión. Recomendada: son 9 tickets y el T9 (Alpaca) se solapa con V39a del plan de M2.
3. **Convivencia:** columna `source` en `market_data` con clave única; las filas viejas quedan como `legacy`.
4. **Ajuste:** guardar con ajuste de splits y sin ajuste de dividendos; detectar cambios retroactivos comparando la vela solapada.
5. **Zona horaria:** UTC con zona en la frontera de cada fuente; en la base, UTC sin zona (la columna no cambia).

## 1. Fuente de las velas en paper

Lo que el código hace hoy: `DataProvider` pide a Yahoo con `auto_adjust=True` (`provider.py:230-236`), borra la zona (`provider.py:269-270`) y tiene una ventana intradía de 729 días (`provider.py:78`). Paper pide `1h` con 168 h de historia (`settings.py:150-151`, `run_paper_pilot.py:75-76`).

| | Yahoo (`yfinance`) | Alpaca, cuenta gratuita |
|---|---|---|
| Feed | Consolidado, no documentado | Acciones: **IEX** en tiempo real; SIP histórico si `end` tiene ≥ 15 min. Pedir SIP más reciente devuelve "subscription does not permit querying recent SIP data" |
| Demora | Sin dato oficial | 15 min en SIP; sin demora en IEX, pero IEX es una sola bolsa (AAPL, 2023-09-29: 923.134 de volumen contra 51.861.083 en SIP) |
| Historia | Diario: completa. 1h/4h: 729 días | Acciones desde 2016 |
| Límites | Sin publicar; sin SLA; los términos prohíben extraer con herramientas automáticas (iuf D4) | 200 llamadas/min en historia, 10.000 puntos por página (`limit`), paginación con `page_token`; 30 símbolos por websocket |
| Zona horaria | Equities intradía en Nueva York (iuf D6) | RFC-3339 con zona, nanosegundos |
| Ajuste | Retroactivo, dividendos incluidos | Parámetro `adjustment`: `raw` (default), `split`, `dividend`, `spin-off`, `all` |

Con velas de 1h, los 15 min no estorban: la vela de las 14:00 cierra a las 15:00 y se puede pedir a partir de las 15:15. Si el ciclo del scheduler necesita la vela apenas cierra, queda IEX, con precios casi iguales pero volumen que no sirve (iuf D4). **Recomendación:** SIP con `end = ahora − 15 min`. Qué demora tolera la estrategia lo decide Fede.

Fuentes: [Alpaca, planes y feeds](https://docs.alpaca.markets/docs/about-market-data-api), [referencia de `stocks/bars`](https://docs.alpaca.markets/reference/stockbars), [FAQ de datos](https://docs.alpaca.markets/docs/market-data-faq). No verifiqué contra la cuenta: no hay claves (`APCA_*`) en este entorno.

## 2. Arquitectura

### 2.1 Opción principal: proveedor abstracto propio

```python
class CandleSource(Protocol):
    name: str                                            # "yahoo" | "alpaca"; va a market_data.source
    def fetch(self, symbol, timeframe, start, end) -> pd.DataFrame: ...   # timestamp en UTC con zona, open..volume
    def earliest(self, timeframe, now) -> datetime | None: ...             # Yahoo: now-729d en 1h/4h; Alpaca: 2016-01-01
    def latest(self, now) -> datetime: ...                                 # Yahoo: now; Alpaca SIP: now-15min
```

Cada implementación mapea símbolos y timeframes, maneja claves, paginación y reintentos, y devuelve velas con zona UTC. No sabe nada de la base.

| Hoy en `DataProvider` | Pasa a |
|---|---|
| `get_ohlc` 96-187, `_find_gaps` 319-374, `_store_data` 274-317, `_rows_to_df` 376-403, `get_cache_stats` 417, `clear_cache` 456, modo `offline` 80-94 y 139-142 | **Común** (queda una vez) |
| `_fetch_yfinance` 189-272, `SYMBOL_MAPPING` 36-47, `TIMEFRAME_MAPPING` 50-60, `INTRADAY_WINDOW_DAYS` 75-78, `_to_yfinance_symbol` / `_to_yfinance_interval` 405-411 | **`YahooSource`** |
| `PANDAS_FREQ_MAPPING` 63-73, `_to_pandas_freq` 413 | Se mantiene; sin verificar si algo lo usa fuera del archivo |

Cómo se arma: `DataProvider(db, source=None, offline=False)`; con `offline=True` la fuente no se usa (igual que hoy, `provider.py:139-142`), así `backtest.py:242` y el CLI (`cli/backtest.py:117`) no cambian. La fuente se elige con `QUANTAGENT_DATA_SOURCE` (`yahoo` por defecto) leída en `settings.py` como las demás (`settings.py:175-182`) y se pasa en los dos puntos de paper: `apps/paper_trading.py:46` y `scripts/run_paper_pilot.py:427`. `SYMBOL_MAPPING` se lee desde `apps/streamlit/views/configuration.py:12`: ese import hay que reapuntarlo.

Lo que la parte común gana además, sin importar la fuente: descartar velas todavía abiertas (hoy se guarda la última aunque esté parcial: el scheduler pide hasta `utcnow()` en `scheduler.py:418` y `_store_data` salta lo existente en `provider.py:299-300`, así que nunca se corrige) y comparar la vela solapada para detectar ajustes (§4).

### 2.2 Contra las alternativas

| | Costo | Qué deja |
|---|---|---|
| **A. Proveedor abstracto (recomendada)** | 9 tickets (§6); T2, T5 y T6 tocan `provider.py` | Tabla, huecos y offline una vez. Yahoo sigue disponible sin clave. El día que se agregue Binance u otra fuente es una clase |
| B. Una sola fuente (solo Alpaca) | ~4 tickets | Menos código. Sin Yahoo, los tests y el desarrollo exigen claves y red; cripto no está verificada en Alpaca |
| C. Proveedor aparte para paper | ~4 tickets | Duplica tabla y detección de huecos (`provider.py:96-187` y `319-374`) y las dos copias se desvían |

Recomiendo A porque ya hay dos implementaciones reales (Yahoo para backtest/desarrollo, Alpaca para operar) y V39a ya obliga a escribir el cliente de Alpaca. Lo que cuesta A frente a B: unos 5 tickets más (T1–T3 y T7 casi no existirían con una sola fuente), que valen si Fede quiere conservar Yahoo.

## 3. Snapshot y velas de operar en la misma tabla

Hoy: la clave es solo un índice no único sobre `symbol, timeframe, timestamp` (`models.py:279-282`), sin fuente, sin restricción de unicidad. El snapshot se carga con `bulk_save_objects` sin chequear duplicados (`fixtures.py:108-124`, igual `load_fixture` 51-77) y el backtest lee sin filtrar por fuente (`backtest.py:762-769` y `968-974`). Si una base tiene un snapshot de SPY 1d y paper baja SPY 1d, `_find_gaps` los toma por un solo conjunto (`provider.py:349-372`) y se mezclan dos ajustes distintos.

| Opción | Cuesta |
|---|---|
| **A (recomendada). Columna `source` + clave única `(symbol, timeframe, timestamp, source)`** | Migración (quitar duplicados antes, `source='legacy'` en lo existente); `DataProvider` filtra por `source`; el loader de snapshots escribe `snapshot:<nombre>`; y **el backtest necesita filtrar `source` en dos consultas** (`backtest.py:762` y `968`), un cambio fuera del alcance de este documento que decide Fede. Sigue el patrón de `environment` en órdenes y señales (`models.py:101-103`, `205`) |
| B. Una base distinta para paper (`DATABASE_URL`) | Sin cambios de esquema ni del backtest. Pero Streamlit y Flask leen una sola base (sin verificar cómo eligen), y paper y backtest ya comparten tablas separadas por `environment` |
| C. Prefijo en `symbol` (`SPY@alpaca`) | Rompe el mapeo de símbolos y `get_cache_stats`; descartada |

## 4. Precios ajustados o sin ajustar

Operar necesita el precio real de hoy (`current_price = df["close"].iloc[-1]`, `run_paper_pilot.py:289`) y una historia que no se mueva sola. El backtest usa precios con dividendos ajustados (`snapshot.py:101-125`) y Yahoo `auto_adjust=True` recalcula toda la historia en cada dividendo (`provider.py:234`, iuf D6). La vela más reciente es igual con o sin ajuste de dividendos; solo cambia el pasado.

**Recomendación:** `adjustment=split` en Alpaca. Un split, sin ajuste, parece una caída del 50% y rompe indicadores; un dividendo de ETF (~0,5–1,5% por trimestre) es un escalón chico dentro de una ventana de 168 h. Alternativa: `all`, que iguala al backtest pero reescribe la historia en cada dividendo.

Qué pasa con lo guardado: `_store_data` nunca actualiza (`provider.py:299-300`). Regla propuesta: cada pedido arranca en la última vela guardada (ya pasa en `provider.py:371-372`); se compara el `close` guardado con el recibido; si difiere más de 1e-4 relativo, se borran las filas de ese `(symbol, timeframe, source)` anteriores y se vuelve a bajar la ventana. Con `split` salta solo en splits; con Yahoo, en cada dividendo. Queda una diferencia con el backtest (dividendos ajustados allá, no acá): no medí cuánto mueve las señales.

## 5. Zona horaria

Hechos: Yahoo intradía de acciones llega en Nueva York y `provider.py:269-270` quita la zona dejando hora de pared de Nueva York (iuf D6); `market_calendar.py:148-149` interpreta un `datetime` sin zona como UTC; el scheduler arma el rango con `datetime.utcnow()` (`scheduler.py:418`), `INTRADAY_WINDOW` usa `datetime.now()` local (`provider.py:209`) y `MarketData.timestamp` es `DateTime` sin zona (`models.py:269`). Hoy `_find_gaps` compara relojes distintos (`provider.py:356`, `371`). El calendario solo se usa si `market_hours_filter` está activo (`backtest.py:305-306`, `776-781`), y el CLI lo apaga (`cli/backtest.py:117`).

Regla: las fuentes devuelven UTC con zona; `DataProvider` convierte a UTC sin zona antes de tocar la base y recibe `start`/`end` en UTC; se prohíbe `datetime.now()` y `utcnow()` fuera de un único `now()` inyectable. La columna no cambia (en SQLite una columna con zona no la conservaría; sin verificar para el motor que usa Fede). Así `market_calendar.py:149` ya es correcto sin tocarlo.

A resolver antes: (a) las filas intradía existentes de Yahoo están en hora de Nueva York: se rotulan `legacy` (§3) y no se leen en paper, no se convierten; (b) el diario es una fecha de sesión a las 00:00 (`cli/data.py:35`): hay que fijar que `1d` es la fecha y no un instante; la hora que devuelve Alpaca en `1Day` está sin verificar; (c) alineación de velas: Yahoo SPY 4h abre a las 9:30 y 13:30 de Nueva York (iuf D3); cómo alinea Alpaca `4Hour` y si `1Hour` abre en :00 o :30, **sin verificar** (la FAQ solo dice que las horas se arman con barras de minuto). Hay que medirlo (T8) antes de usar 4h o de combinar fuentes.

## 6. Tickets de implementación (cada uno ≤ 100 líneas)

**T1 — Interfaz y `YahooSource`.** Contexto: el código de Yahoo está mezclado con la caché (`provider.py:189-272`). Cambio: `quantagent/data/sources/{base,yahoo}.py` con `CandleSource` y `YahooSource` (copia de `_fetch_yfinance`, mapas y ventana; devuelve UTC con zona; `earliest`/`latest`). Aceptación: un test pide velas con `yfinance` simulado en Nueva York y verifica `tzinfo == UTC` y la hora convertida; `pytest tests/test_data_provider.py` pasa sin cambios. Archivos: los 2 nuevos + test. Fuera: tocar `provider.py`.

**T2 — `DataProvider` delega en la fuente.** Contexto: §2.1. Cambio: `DataProvider(db, source=None, offline=False)`; `get_ohlc` llama `source.fetch`; borrar lo movido a Yahoo; mantener `SYMBOL_MAPPING` accesible para `configuration.py:12`; `earliest`/`latest` recortan el pedido. Aceptación: con una fuente falsa, `get_ohlc` pide solo el hueco y la segunda llamada no llama a la fuente; con `offline=True` nunca la llama. Archivos: `provider.py`, `configuration.py`, `tests/test_data_provider.py`. Fuera: Alpaca, base de datos.

**T3 — Elegir fuente por configuración.** Cambio: `QUANTAGENT_DATA_SOURCE` en `settings.py` + `build_source(name)`; `apps/paper_trading.py:46` y `scripts/run_paper_pilot.py:427` lo usan. Aceptación: valor inválido falla al arrancar con mensaje que lista `yahoo|alpaca`; sin variable devuelve `YahooSource`. Fuera: backtest (`backtest.py:242` sigue en `offline`), `.env.example` más allá de la variable.

**T4 — Columna `source`.** Cambio: migración Alembic que quita duplicados, agrega `source` (`legacy` en lo existente) y la clave única; `models.py:261-282`. Aceptación: `alembic upgrade head` y `downgrade -1` pasan sobre una base con duplicados; insertar la misma clave dos veces falla. Fuera: leer o escribir `source`.

**T5 — Leer y escribir por fuente.** Cambio: `DataProvider` filtra y guarda `source=source.name`; `load_snapshot`/`load_fixture` escriben `snapshot:<nombre>` / `fixture`; **backtest filtra `source` en `backtest.py:762` y `968`** (requiere visto bueno de Fede, §3). Aceptación: con filas `snapshot:x` y `alpaca` del mismo símbolo, un `get_ohlc` de cada fuente devuelve solo las suyas y el backtest offline devuelve solo las del snapshot. Fuera: lógica del backtest.

**T6 — Velas cerradas y UTC en la parte común.** Cambio: descartar velas con `timestamp + timeframe > latest(now)`; convertir a UTC sin zona antes de guardar; un `now()` inyectable en lugar de `provider.py:209` y `scheduler.py:418`. Aceptación: con `now` fijo, no se guarda la vela abierta y al avanzar el reloj se guarda completa. Fuera: ajustes (T7).

**T7 — Detección de ajuste retroactivo.** Cambio: comparar `close` solapado (1e-4 relativo) y renovar la ventana de ese `(symbol, timeframe, source)` (§4). Aceptación: test con un split simulado a 2:1: la historia guardada queda ajustada y `current_price` es la nueva. Fuera: dividendos en `raw`.

**T8 — Medición con claves reales (VM).** Cambio: script en `scripts/` que pide SPY 1h, 4h y 1d con SIP y `adjustment=split` y reporta alineación, hora del 1d, primera vela, cantidad y tiempo por pedido; resultado agregado a este documento. Aceptación: el PR trae las cifras o dice "sin claves". Depende de T9.

**T9 — `AlpacaSource` (solo datos).** Cambio: cliente `data.alpaca.markets/v2/stocks/bars` con `feed=sip`, `adjustment=split`, paginación, `end ≤ ahora − 15 min`; reutiliza el cliente de V39a si ya existe. Aceptación: con respuestas grabadas devuelve UTC; falla con mensaje claro sin claves; un test falla si llama a un endpoint que no sea de datos. Fuera: órdenes, cuenta, cripto.

Orden: T1→T2→T3; T4→T5; T6, T7 después de T2; T9 antes de T8. Costo de la opción A: 9 tickets; B y C rondan los 4.

## Qué decide Fede

1. Fuente de operación: Alpaca SIP con 15 min de demora, o IEX sin demora (§1).
2. Opción de arquitectura: A, B o C (§2.2).
3. Convivencia: columna `source` (toca 2 consultas del backtest) o base separada (§3).
4. Ajuste para operar: `split` (recomendado) o `all` (§4).
5. Si cripto (BTC) se opera en paper: sin verificar en Alpaca; con una sola fuente por corrida, BTC obliga a Yahoo o a otra fuente.
6. Si T9 se junta con V39a o se hace después.

## Sin verificar

- Todo lo de Alpaca salió de la documentación oficial; no se midió: no hay claves en el entorno.
- Alineación de `4Hour` y `1Hour`, y hora de `1Day` en Alpaca.
- Si `adjustment=split` está permitido y es retroactivo en la cuenta gratuita.
- Cripto en Alpaca: historia, límites y símbolos.
- Límite de 200 llamadas/min: figura en la página de planes; no lo vi en la referencia de `stocks/bars`.
- Términos de uso de Alpaca para guardar y reusar datos.
- Qué motor de base usa Fede y cómo la eligen Streamlit y Flask.
- Si `_to_pandas_freq` (`provider.py:413`) tiene algún uso.
- Que 1e-4 sea un umbral adecuado para detectar ajuste.

Índice: [docs/04_decisions/README.md](./README.md)

## Decisiones de Fede (2026-10-12)

Tomadas en la `R:` del PR #116 y confirmadas en la sesión del PM el mismo día. Lo de arriba queda como se propuso; esta sección manda donde difiera.

| # | Pregunta | Decisión |
|---|---|---|
| 1 | Fuente de operación | Alpaca, feed SIP con `end` a 15 minutos; Yahoo queda de respaldo y para desarrollar sin claves |
| 2 | Arquitectura | Opción A: proveedor abstracto `CandleSource` |
| 3 | Convivencia | Columna `source` en `market_data`; incluye filtrar `source` en las dos consultas del backtest (`backtest.py:762` y `968`) |
| 4 | Ajuste para operar | `adjustment=split` |
| 5 | Cripto en paper | Afuera por ahora |
| 6 | T9 y V39a | Se hacen juntos, en un solo ticket |

Zona horaria (§5): UTC con zona en la frontera de cada fuente y UTC sin zona en la base, como se propuso.

Tickets en BEADS (label `m3`; todavía no están en la cola de `PLAN-CONTINUACION.md` §4, se copian al abrir su lote):

| Ticket | BEADS | Qué |
|---|---|---|
| T1 | `QuantAgent-qix2` | Interfaz `CandleSource` y `YahooSource` |
| T2 | `QuantAgent-br4h` | `DataProvider` delega en la fuente |
| T3 | `QuantAgent-vr3z` | Elegir la fuente por configuración |
| T4 | `QuantAgent-1y0i` | Columna `source` y clave única |
| T5 | `QuantAgent-lrf8` | Leer y escribir por fuente, backtest incluido |
| T6 | `QuantAgent-gjmf` | Velas cerradas y UTC en la parte común |
| T7 | `QuantAgent-01kb` | Detección de ajuste retroactivo |
| T9 | `QuantAgent-jpms` | `AlpacaSource`, solo datos (es la fila V39a del plan de M2) |
| T8 | `QuantAgent-v38v` | Medición con claves reales |

Dependencias: T1 → T2 → T3; T4 y T2 → T5; T2 → T6 y T7; T9 → T8.

Pendiente de ordenar: V39a está en el lote 5 de M2 y T1 es de M3. Si T9 se hace antes que T1, `AlpacaSource` define los tres métodos de la interfaz sin importar el protocolo (así quedó escrito en el ticket).

### Novedades del 2026-10-12, sin medir

- Fede tiene claves de Alpaca para paper trading. Según la [página de autenticación de Alpaca](https://docs.alpaca.markets/docs/authentication), las mismas claves de paper autentican contra `data.alpaca.markets`: no hace falta otra clave ni otra suscripción.
- La documentación se contradice sobre si una cuenta que es solo de paper puede pedir histórico SIP: la [página de paper](https://docs.alpaca.markets/docs/paper-trading) habla de datos IEX y la [de planes](https://docs.alpaca.markets/docs/about-market-data-api) de SIP con 15 minutos de demora en el plan gratuito. Fede entiende que la cuenta alcanza; T8 lo deja medido.
- Para T9 se usa un cliente propio sobre `requests` y no el SDK `alpaca-py`: es un solo endpoint y `requests` ya es dependencia. El SDK se reevalúa en M3, para órdenes y cuenta.
- El soporte de Alpaca indica que sus datos no se pueden redistribuir: ningún PR ni snapshot versionado lleva velas de Alpaca.
- Sugerencia del PM incluida en T8: comparar señales de las estrategias diarias con `split` contra `all`, porque el backtest usa precios ajustados por dividendos y esa diferencia no está medida.

### Medición con claves reales (T8, `QuantAgent-v38v`, 2026-10-17)

Corrida de `scripts/medir_alpaca.py` con las claves de paper, SPY desde 2016-01-04, `adjustment=split` salvo donde se indica. No se guardó ninguna vela.

| Pregunta | Resultado |
|---|---|
| ¿Paper alcanza SIP? | Sí. `feed=sip` y `feed=iex` responden, ambos con 701 velas diarias desde 2024. |
| `1Hour` | 42.703 velas, la primera a las 09:00 UTC (04:00 NY): incluye preapertura y postcierre. 217 pedidos, 220 s. |
| `4Hour` | 11.637 velas, alineadas a UTC y no a la sesión de NY (caen a las 03/04, 07/08, 11/12, 15/16, 19/20 hora NY según DST). 215 pedidos, 150 s. |
| `1Day` | 2.713 velas, selladas a las 00:00 NY (05:00 UTC en invierno). 1 pedido, 1 s. |
| Límite de llamadas | Los headers informan `x-ratelimit-limit: 200` (por minuto), lo que confirma la cifra de la página de planes. |
| `split` vs `all` (1d) | Mismas 2.713 velas. El cierre difiere hasta 17,78% (mediana 7,62%): `all` reescribe diez años de historia por dividendos. |
| Señales con `split` vs `all` | `sma-cross`: 4 entradas en ambos, 8 velas con señal distinta. `momentum-12m`: 100 entradas con `split`, 103 con `all`, 3 velas con señal distinta. |

Lectura: la diferencia entre ajustes es grande en nivel de precio pero chica en señales (entre 3 y 8 velas de 2.713). No justifica pasar a `all`; la recomendación de `split` de la sección 4 se mantiene. Al implementar el provider hay que recortar `1Hour` a sesión regular si la estrategia no debe ver preapertura, y tener presente que `4Hour` no cierra con la sesión.

Sin explicar: `1Hour` necesitó 217 pedidos para 42.703 velas (unas 200 por página) aun pidiendo `limit=10000`; `4Hour` se comportó igual. No se investigó la causa.
