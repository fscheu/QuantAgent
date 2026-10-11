# QuantAgent-cbp · Tamaño de posición y límite diario en las comparaciones de M2
Estado: **propuesta, decide Fede**. Medido el 2026-10-10 sobre `lote/costos-y-referencia` @ 54b761f7 (con #101). Fila V18. Todo es `rsi` sobre SPY diario, 2007-01-03 a 2018-12-31, snapshot `etf-1d-2026-10`, capital 100.000. Los números van como los imprimen las herramientas (punto decimal).
## 1. Cómo se calcula hoy el tamaño y qué pasa al pedir 100%
`cantidad = equity × TRADING_BASE_POSITION_PCT × confianza de la señal / precio` (`quantagent/trading/position_sizer.py`). El 5% es un techo, no el tamaño: la confianza de rsi promedia 0.24 (mediana 0.19), así que la entrada media es de 1.211 USD (1,2% del capital), y como rsi tiene posición el 57,5% de los días, el capital invertido promedio es **0,75%**. `sma-cross` y `momentum-12m` devuelven confianza 1.0 y sí usan el 5% entero.

El motor propio **no puede correr con 100%**. Con `R="--snapshot etf-1d-2026-10 --symbol SPY --from 2007-01-03 --to 2018-12-31"`:
```
$ TRADING_BASE_POSITION_PCT=1.0 TRADING_MAX_POSITION_PCT=1.0 python -m quantagent.cli backtest run --strategy rsi $R --costos etf
  File ".../quantagent/trading/position_sizer.py", line 34, in __init__
    raise ValueError("base_position_pct must be between 0% and 10%")
ValueError: base_position_pct must be between 0% and 10%          (exit 1, no corre ninguna vela)
```
Salteando esa validación desde un script temporal (pegado en el PR, no está en el repo) aparecen dos trabas más. Con `TRADING_MAX_POSITION_PCT` por defecto, `RiskManager.validate_trade` rechaza 389 órdenes (`Order rejected - Position too large: $12516.45 > max $10028.79 (10% limit)`) y quedan 105 trades en vez de 187. Con `TRADING_MAX_POSITION_PCT=1.0` rsi corre entera, sin rechazos. Pero una estrategia con confianza 1.0 pide exactamente todo el efectivo y el control de capital la rechaza: `sma-cross` pierde su primer trade (`Insufficient capital: need $100000.00, have $100000.00`) y hace 4 en vez de 5. Son tres trabas, y las tres le tocan a V19.
## 2. Corridas
```
python -m quantagent.cli backtest run --strategy rsi $R --costos etf --out f1.csv --equity-out f1-eq.csv                         # fila 1
TRADING_SLIPPAGE_PCT=0 python -m quantagent.cli backtest run --strategy rsi $R --out f2.csv --equity-out f2-eq.csv                # fila 2
TRADING_BASE_POSITION_PCT=1.0 python scripts/bt_adapter.py --strategy rsi $R --fill close --out f3.csv                            # fila 3 (trades y PnL)
python scripts/recalc_metrics.py f1.csv --equity f1-eq.csv [--risk-free-rate 0]                                                   # PnL, Sharpe, max drawdown
python -c "import sys,pandas as pd; t,e=pd.read_csv(sys.argv[1]),pd.read_csv(sys.argv[2]); x=e.positions_value.abs()/e.equity; print(f'{(x>0).mean():.1%} {x.mean():.2%} {t.groupby(t.exit_time.str[:10]).pnl.sum().min()/1e5:.3%} {e.equity.pct_change().min():.3%}')" f1.csv f1-eq.csv
python scripts/recalc_metrics.py --comprar-y-mantener $QUANTAGENT_SNAPSHOT_DIR/etf-1d-2026-10/SPY.parquet --from 2007-01-03 --to 2018-12-31 --slippage-pct 0.0005 --capital 5000   # referencia de la fila 1
```
Filas 4 y 5: el CLI del motor con la validación salteada y las dos variables en 1.0. Fila 6: el adaptador con `TRADING_BASE_POSITION_PCT=0.5` y la confianza reemplazada por 1.0. La curva de equity de las filas 3 y 6 sale de la librería (`_equity_curve`), porque el adaptador no la exporta. Los tres scripts temporales están completos en el PR.

| Fila | Opción y origen | Costos | Trades | PnL (retorno) | Max drawdown | Sharpe 2% / 0% |
|---|---|---|---|---|---|---|
| 1 | A · 5%, motor | `etf` | 187 | -447.29 (-0.447%) | 0.606% | -11.86 / -0.22 |
| 2 | A · 5%, motor | slippage 0 | 187 | -220.54 (-0.221%) | 0.512% | -11.78 / -0.11 |
| 3 | B · 100%, librería `close` | slippage 0 | 187 | -4908.13 (-4.908%) | 9.800% | -0.69 / -0.11 |
| 4 | B · 100%, motor sin la validación | slippage 0 | 187 | -4908.13 (-4.908%) | 9.800% | -0.69 / -0.11 |
| 5 | B · 100%, motor sin la validación | `etf` | 187 | -9141.21 (-9.141%) | 11.968% | -0.80 / -0.22 |
| 6 | C · 50% fijo sin confianza, librería `close` | slippage 0 | 187 | 12224.07 (12.224%) | 14.172% | -0.17 / 0.21 |

| Fila | Días con posición | Capital invertido medio | Peor día: realizado / valuado | Comprar y mantener que le corresponde: PnL, max drawdown, Sharpe 2% / 0% |
|---|---|---|---|---|
| 1 | 57.5% | 0.75% | -0.113% / -0.113% | 5% en SPY y 95% en efectivo, `etf`: 6295.20, 3.074%, -1.36 / 0.47 |
| 2 | 57.5% | 0.75% | -0.110% / -0.111% | 5% en SPY, slippage 0: 6306.50, 3.076%, -1.36 / 0.47 |
| 3 y 4 | 57.5% | 14.97% | -2.139% / -2.211% | 100% en SPY, slippage 0: 126129.95, 55.189%, 0.34 / 0.44 |
| 5 | 57.5% | 14.97% | -2.175% / -2.260% | 100% en SPY, `etf` (línea `Comprar y mantener:` del CLI): 125903.93, 55.189%, 0.34 / 0.44 |
| 6 | 57.5% | 28.77% | -3.255% / -2.613% | 50% en SPY, slippage 0: 63064.97, 29.173%, 0.26 / 0.47 |

- **Qué se compara con qué.** Con costos `etf` solo hay filas 1 y 5, las dos del motor. Las filas 3 y 6 salen de la librería, que no aplica slippage: se comparan con las filas 2 y 4, no con la 1. La fila 4 coincide con la 3 en los 187 trades (fechas, lado y motivo iguales; diferencia máxima de PnL por trade 0.00001 USD y de cantidad 5.3e-06 acciones, que excede la tolerancia de 1e-6 del doc de 80x en 60 trades). Las filas 4 y 5 no se reproducen con el repo tal cual.
- **Referencia escalada (opción A).** Comprar SPY con el 5% y dejar el resto en efectivo: el PnL es el 5% de la línea del CLI (125903.93 × 0.05 = 6295.20, igual a `--capital 5000`). El drawdown y el Sharpe sobre los 100.000 no los da ninguna herramienta del repo; salen de un script temporal. Aun escalada, la referencia invierte 5% y rsi 0,75%.
- **Sharpe.** El del CLI anualiza el retorno medio por vela de la curva de equity menos una tasa libre de riesgo de 2% anual, dividido por el desvío. El efectivo del motor no rinde nada, así que con 95% en efectivo el 2% domina: -11.86 para rsi y -1.36 para la propia referencia al 5%. Con tasa 0 el Sharpe no depende del tamaño (-0.22 con 5% y con 100%).
- **Fila 6.** Mismas 187 señales; solo cambia que todas pesan igual. El PnL pasa de negativo a positivo: lo que hace perder a rsi es apostar más cuando el RSI es más extremo. Es un dato visto dentro de la muestra, no una razón para elegir C.
## 3. Práctica profesional
- Tamaño por volatilidad: Moskowitz, Ooi y Pedersen, "Time series momentum", *Journal of Financial Economics* 104 (2012), §4.1: cada posición se dimensiona a 40% de volatilidad anual esperada y "the choice of 40% is inconsequential" (https://pages.stern.nyu.edu/~lpederse/papers/TimeSeriesMomentum.pdf, leído).
- Tamaño por riesgo fijo: *The Original Turtle Trading Rules*, cap. 3 "Position Sizing" (una unidad se calcula para que 1 N, el rango medio de 20 días, valga 1% de la cuenta) y cap. 5 "Stops" (stop a 2 N, 2% de riesgo máximo por trade) (https://oxfordstrat.com/coasdfASD32/uploads/2016/01/turtle-rules.pdf, leído).
- Capital asignado por estrategia, con el tamaño medido contra esa asignación y no contra la cuenta entera: **no verificado**. Que R. Carver, *Systematic Trading* (2015), lo trate en capítulos de "volatility targeting" y "position sizing": **no verificado** (no pude leer el índice).
## 4. Límite diario
Hoy no actúa en backtest (V17, PR #84) y desde #95 y #101 no frena cierres. La fila 5 no registró ningún rechazo de ningún control. El peor día de rsi/SPY es -0.113% del capital con 5%, -2.26% con 100% y -3.26% con 50% fijo: el límite de 5% no se dispara en ninguna opción. No vale para otras estrategias: SPY cayó más de 5% en 12 días del período (el peor, -9.84% el 2008-10-15) y `sma-cross` con 100% tuvo un día de -6.52% valuado y un cierre de -13.26%. Regla propuesta: dejarlo apagado en las corridas diarias de M2, informar "peor día" por fila, y corregirlo (V21) antes del lote de 4h, donde hay varios trades por día y sí puede cambiar resultados.
## 5. Opciones
**A · Todas con 5% y la referencia escalada a 5%.** No toca el motor. V19 cambia de objeto (escalar la referencia en vez de `--tamano-posicion`); V22 y V23 necesitan drawdown y Sharpe de la referencia escalada. El Sharpe con 2% queda inservible y habría que informarlo con tasa 0. Referencias de M1: sin cambio.

**B · 100% del capital asignado, sin apalancamiento, solo en las corridas de M2.** V19 agrega `--tamano-posicion` y tiene que resolver las tres trabas de §1; si no entra en un ticket, se parte. V20 congela números con 100%; V22, V23, V30 y V33 leen esos números. Para rsi "100%" termina siendo 15% invertido en promedio, por la confianza. Referencias de M1: sin cambio, el default sigue en 5%.

**C · Tamaño fijo, sin ponderar por confianza** (fila 6; para rsi, 50% equivale a arriesgar 1% del capital a su stop de 2%). Cambia lo que hace la estrategia, no solo la escala. Dimensionar por distancia al stop no sirve para `sma-cross` ni `momentum-12m`, que salen por regla y llevan stop en 0.01. Afecta V19, V20 y todo lo que compara. Referencias de M1: sin cambio solo si queda detrás de un flag.
## 6. Recomendación
**B**, con dos agregados que no dependen del tamaño: la tabla de `compare` (V22, V23) informa capital invertido medio y Sharpe con tasa 0 al lado del actual. C la dejaría como variante de rsi dentro del barrido con registro de intentos (V31): adoptarla ahora, después de ver que da positivo, es elegir con el resultado a la vista. Me haría cambiar a **A** que V19 no pueda resolver las tres trabas sin tocar el control de riesgo que comparte con paper; y a **C** que Fede quiera que M2 juzgue la señal y no la señal multiplicada por la confianza.
