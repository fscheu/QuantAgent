# QuantAgent-ld3 · Perfiles de costos de backtest para M2 (fila V13)

Estado: **decidido por Fede el 2026-10-10** (`R:` en el PR #93; ver sección 6). Consulta de fuentes: 2026-10-10. Plan: `docs/02_planning/QuantAgent-lp1-tickets-m2.md` (V13). Unidades: fracción por lado, como `TRADING_SLIPPAGE_PCT` y `--commission-pct` (0,001 = 0,10%). Hoy: slippage 0,0005 y comisión 0.

## 1. Lo que cuesta operar hoy

| Concepto | Valor | Fuente (consultada 2026-10-10) |
|---|---|---|
| Alpaca, comisión ETFs EE.UU. | 0 (cuenta individual vía API) | https://files.alpaca.markets/disclosures/library/BrokFeeSched.pdf y https://alpaca.markets/support/commission-clearing-fees |
| Alpaca, SEC en ventas | 0,0000206 × valor (20,60 USD por millón) | mismo PDF |
| Alpaca, FINRA TAF en ventas | 0 por acción según su tarifario | mismo PDF. Ver duda 1 |
| Alpaca, CAT (compra y venta) | 0,000003 USD por acción | mismo PDF |
| IBKR plan fijo | no verificado (403). QuantConnect lo modela en 0,005 USD/acción, mín 1 USD, máx 0,5% | https://www.interactivebrokers.com/en/pricing/commissions-stocks.php (403); https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/transaction-fees/supported-models |
| IBKR plan escalonado | no verificado (403) | misma página de IBKR |
| SEC sección 31 (ventas) | 20,60 USD por millón desde 2026-04-04 | https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2 |
| FINRA TAF (ventas) | no verificado en FINRA (404). Páginas de brókers: 0,000195 USD/acción, tope 9,79 USD por operación, desde 2026-01-01 | https://www.investrade.com/fees/ (secundaria) |
| Binance spot, taker / maker | 0,100% / 0,100% (usuario regular, sin BNB) | https://www.binance.com/en/fee/schedule |
| Spread SPY | no verificado. SSGA no dio el número (503). Una búsqueda reportó 0,27 pb en jun-2024 | https://ssga.com/insights/spy-liquidity-flexibility-to-navigate-any-market |
| Spread EEM (mediana 30 días) | 0,01% al 2026-10-09 | https://www.ishares.com/us/products/239637/ishares-msci-emerging-markets-etf |
| Spread DBC | no verificado (Invesco y ETF.com sin dato o 403) | https://www.invesco.com/us/financial-products/etfs/ |
| Spread BTC/USDT | 83046,00 / 83046,01 = 0,0012 pb. Una sola observación, sábado 17:23 UTC | https://data-api.binance.vision/api/v3/ticker/bookTicker?symbol=BTCUSDT |

Dudas: (1) Alpaca dice TAF 0 y los brókers citan 0,000195 por acción; con 166 acciones de SPY serían 0,03 USD, irrelevante en cualquier caso. (2) EEM con 0,01% sale redondeado a dos decimales, el valor real está entre 0,005% y 0,015%. (3) IBKR máx 0,5% (QuantConnect) contra 1% (resumen de una búsqueda): sin resolver.

## 2. Perfiles propuestos

| Perfil | Comisión/lado | Slippage/lado | De dónde sale |
|---|---|---|---|
| `etf` | 0,00001 | 0,0002 | Alpaca cobra 0; 0,00001 es la SEC (0,00206% en ventas) promediada entre compra y venta. Slippage: **estimación**, unas 4 veces la mitad del spread de EEM (0,005%) para cubrir DBC, que no verifiqué |
| `etf-conservador` | 0,0002 | 0,0005 | Piso IBKR: 1 USD mínimo sobre 5.000 USD = 0,02% (con el modelo de QuantConnect, no verificado). Slippage: el valor actual, que dejo como techo |
| `cripto` | 0,001 | 0,0005 | Binance 0,100% en maker y taker. Slippage: **estimación**, se mantiene el actual; el spread medido (0,0012 pb) no alcanza para bajarlo y las velas de 4h llenan al cierre |

## 3. Qué hace la industria

- **QuantConnect.** Su modelo por defecto no aplica slippage (`NullSlippageModel`) y el modelo de IBKR usa `InteractiveBrokersFeeModel` con comisiones reales de IBKR. Cada bróker trae su propio modelo de comisiones y el usuario cambia el slippage a mano (https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/slippage/key-concepts).
- **Zipline.** Comisión por defecto `PerShare` de 0,001 USD por acción, sin mínimo. Slippage por defecto `VolumeShareSlippage`: tope del 2,5% del volumen de la barra y coeficiente de impacto 0,1, o sea que depende del tamaño de la orden, no es un porcentaje fijo (https://zipline.ml4trading.io/api-reference.html).
- **Frazzini, Israel y Moskowitz, "Trading Costs" (2018).** Con 1,7 billones de USD de ejecuciones reales de AQR en 21 mercados, estiman costos de un orden de magnitud menores que los estudios previos (https://papers.ssrn.com/abstract=3229719). No leí las tablas: no cito cifras en puntos básicos.

## 4. Efecto medido (lo completa el PM en la VM)

Estrategia rsi, SPY, 2007-01-03 a 2018-12-31. La base es la corrida actual (0,0005 y 0). Comando por perfil:

```
TRADING_SLIPPAGE_PCT=<slip> python -m quantagent.cli backtest run --strategy rsi --snapshot etf-1d-2026-10 --symbol SPY --from 2007-01-03 --to 2018-12-31 --commission-pct <com>
```

| Perfil | slip | com | Trades | Total PnL | Sharpe |
|---|---|---|---|---|---|
| actual | 0.0005 | 0 | 187 | -447.29 | -11.86 |
| `etf` (propuesto) | 0.0002 | 0.00001 | 187 | -315.83 | -11.82 |
| `etf-conservador` | 0.0005 | 0.0002 | 187 | -537.84 | -11.89 |
| `cripto` | 0.0005 | 0.001 | 187 | -899.26 | -11.98 |

## 5. Recomendación

Adoptar `etf` para los 12 ETFs diarios y SPY en 4h, y `cripto` para BTC; guardar `etf-conservador` como piso para la prueba de robustez. Los valores de slippage son estimaciones mías y no están sostenidos por un spread verificado de SPY ni de DBC: antes de fijarlos conviene medir esos dos spreads con Alpaca o con una consulta a la cuenta, y si Fede prefiere no apoyarse en estimaciones, usar 0,0005 en los tres perfiles y que solo cambie la comisión.

Medido por el PM en la VM el 2026-10-10 sobre `lote/costos-y-referencia` @ cb00a864 (con #96). Comprar y mantener en el mismo rango: PnL entre 125452 y 126035 según el perfil.

## 6. Decisión

Fede, 2026-10-10 (PR #93): slippage 0,0005 por lado en los tres perfiles, porque el slippage de ETFs propuesto (0,0002) es una estimación sin spread verificado de SPY ni de DBC. Solo cambia la comisión por lado:

| Perfil | Comisión/lado | Slippage/lado |
|---|---|---|
| `etf` | 0 | 0,0005 |
| `etf-conservador` | 0,0002 | 0,0005 |
| `cripto` | 0,001 | 0,0005 |

Estos son los valores que implementa V14 (`--costos`). La tabla de la sección 2 queda como propuesta original.
