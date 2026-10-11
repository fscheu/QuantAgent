# QuantAgent-80x · Qué motor da el número oficial de M2

Estado: **decidido por Fede el 2026-10-10 (PR #102): opción C**, ver §8. Medido el 2026-10-10 sobre `lote/costos-y-referencia` @ 3ea25e9a (sin #101). Puede revisar el [ADR del 2026-10-08](./2026-10-ADR-backtest-engine-propio-vs-terceros.md). Segunda parte de la prueba de `QuantAgent-i6n` (PR #92).

## 1. Qué se midió y cómo

SPY diario, 2007-01-03 a 2018-12-31, snapshot `etf-1d-2026-10`, cinco estrategias, capital 100.000, 5% por trade. Cuatro corridas por estrategia y por comisión (`C` = 0 y 0.001 por lado), con `R="--snapshot etf-1d-2026-10 --symbol SPY --from 2007-01-03 --to 2018-12-31"`:
```
TRADING_SLIPPAGE_PCT=0 python -m quantagent.cli backtest run --strategy <E> $R --commission-pct <C> --out m0.csv   # motor, slippage 0
python -m quantagent.cli backtest run --strategy <E> $R --commission-pct <C> --out md.csv                         # motor, por defecto (slippage 0,05%)
python scripts/bt_adapter.py --strategy <E> $R --fill close --commission-pct <C> --out ac.csv                     # librería, llena al cierre
python scripts/bt_adapter.py --strategy <E> $R --fill next-open --commission-pct <C> --out an.csv                 # librería, llena en la apertura siguiente
```

Todas las métricas salen del mismo cálculo sobre el CSV de trades, igual para los dos motores: `python scripts/recalc_metrics.py <csv> --commission-pct <C>` da trades, win rate, profit factor y PnL (en dólares), y verifica el PnL fila por fila. No se usa el profit factor ni el Sharpe que imprime la librería: los calcula sobre retornos porcentuales (para rsi imprime 1.12 con PnL negativo; en dólares da 0.93). El motor se compara con slippage 0 porque el adaptador no aplica slippage.

**Max drawdown:** el adaptador no exporta la curva de equity, así que el drawdown con posiciones valuadas día a día no se puede calcular igual en los dos lados y queda fuera. Lo que sí es comparable es el drawdown de la equity *realizada* (capital + PnL acumulado al cierre de cada trade), que subestima al real: en el motor, sma-cross sin comisión da 0,400% realizado contra 1,290% valuado día a día (`--equity-out`). Comando:
```
python -c "import sys,pandas as pd; e=pd.concat([pd.Series([1e5]),1e5+pd.read_csv(sys.argv[1]).sort_values('exit_time').pnl.cumsum()]); print(f'{((e.cummax()-e)/e.cummax()).max():.3%}')" <csv>
```

## 2. Resultado: PnL

Cada celda: sin comisión / con 0,1% por lado. La cantidad de trades es la misma en las cuatro corridas. En la segunda tabla (win rate, profit factor y drawdown realizado) motor a slippage 0 y librería `close` dan lo mismo en las diez filas y van en una sola columna.

| Estrategia | Trades | Motor (slip 0) | Librería `close` | Librería `next-open` | Motor por defecto |
|---|---|---|---|---|---|
| rsi | 187 | -220.54 / -673.53 | -220.54 / -673.53 | -596.73 / -1048.18 | -447.29 / -899.26 |
| fifty-two-week-high | 5 | 81.34 / 76.15 | 81.34 / 76.15 | 78.55 / 73.36 | 78.75 / 73.56 |
| triple-screen | 1 | -16.02 / -17.64 | -16.02 / -17.64 | -34.46 / -36.06 | -16.83 / -18.45 |
| sma-cross | 5 | 5323.62 / 5266.11 | 5323.62 / 5266.11 | 5186.16 / 5128.82 | 5294.87 / 5237.37 |
| momentum-12m | 3 | 6605.88 / 6567.90 | 6605.88 / 6567.90 | 6582.23 / 6544.27 | 6586.89 / 6548.92 |

| Estrategia | Win rate | PF `close` | PF `next-open` | DD real. `close` | DD real. `next-open` |
|---|---|---|---|---|---|
| rsi | 43,32% (`next-open` 41,71%) | 0.93 / 0.79 | 0.78 / 0.65 | 0,476% / 0,816% | 0,723% / 1,162% |
| fifty-two-week-high | 40% | 2.07 / 1.96 | 2.01 / 1.91 | 0,051% / 0,053% | 0,052% / 0,054% |
| triple-screen | 0% | 0 / 0 | 0 / 0 | 0,016% / 0,018% | 0,034% / 0,036% |
| sma-cross | 80% | 13.80 / 13.37 | 15.31 / 14.78 | 0,400% / 0,409% | 0,349% / 0,359% |
| momentum-12m | 66,67% | 20.95 / 20.25 | 18.73 / 18.16 | 0,314% / 0,324% | 0,352% / 0,362% |

## 3. Trade por trade: motor (slip 0) contra librería `close`

Se compara fila por fila fecha de entrada, fecha de salida, lado, motivo, cantidad y precios (tolerancia 1e-6) y PnL (tolerancia 0,01). Imprime trades del motor, trades de la librería y cuántos coinciden:
```
python -c "import sys,pandas as pd; a,b=[pd.read_csv(p) for p in sys.argv[1:3]]; t=['entry_time','exit_time','side','exit_reason']; n=['qty','entry_price','exit_price','pnl']; ok=(a[t]==b[t]).all(axis=1)&((a[n]-b[n]).abs()<[1e-6,1e-6,1e-6,0.01]).all(axis=1); print(len(a),len(b),int(ok.sum()))" m0.csv ac.csv
```

Resultado, igual sin comisión y con 0,1%: rsi 187 de 187, fifty-two-week-high 5 de 5, triple-screen 1 de 1, sma-cross 5 de 5, momentum-12m 3 de 3. Coinciden los 201 trades en las dos configuraciones. La mayor diferencia de PnL en un trade es 0,000001 USD y la mayor de cantidad 2,8e-07 acciones (redondeo del CSV del motor a 8 decimales). **No hay diferencias que explicar y no queda nada "sin explicar".**

Lo que esta coincidencia prueba y lo que no:
- Prueba la ejecución y la contabilidad: en qué vela se llena, stops y take profit contra máximo y mínimo, efectivo, comisión de entrada y de salida.
- No prueba la lógica de las estrategias. El adaptador llama al mismo `generate_signal` y al mismo `should_exit`, y usa la misma fórmula de tamaño. Un error ahí aparece igual en los dos lados.
- No ejercita el control de riesgo. Con 5% por trade y una posición por vez, ninguna orden choca contra capital, tamaño máximo ni pérdida diaria, que es donde vivían #95 y #101. Y es un solo activo: #101 se encontró en XLK, no en SPY.

## 4. Cuánto pesa cada supuesto

Diferencia de PnL sin comisión, contra motor a slippage 0:

| Estrategia | Llenar en la apertura siguiente | Slippage 0,05% | Comisión 0,1% |
|---|---|---|---|
| rsi | -376.19 | -226.75 | -452.99 |
| fifty-two-week-high | -2.79 | -2.59 | -5.19 |
| triple-screen | -18.44 | -0.81 | -1.62 |
| sma-cross | -137.46 | -28.75 | -57.51 |
| momentum-12m | -23.65 | -18.99 | -37.98 |

Suponer que se opera al cierre de la vela que dio la señal pesa más que el slippage por defecto en las cinco estrategias, y más que una comisión de 0,1% en triple-screen y sma-cross. En rsi los 187 trades salen el mismo día y por el mismo motivo con los dos llenados; cambia el precio de entrada y 3 trades cambian de signo. En triple-screen el único trade pierde el doble (entra a 160,61 en vez de 164,40, en corto, con el mismo stop). El motor propio no tiene modo de apertura siguiente: hoy ese número solo lo da la librería.

## 5. Hechos para decidir

- **Tiempo por corrida en la VM:** rsi 55 s en el motor y 6 s en la librería; sma-cross 37 s y 5 s; momentum-12m unos 5 minutos en los dos (el costo está en la estrategia, no en el motor).
- **Seis defectos del motor propio en una semana**, casi todos encontrados comparando con la librería: stops evaluados solo al cierre (#57, #60); reloj por calendario en vez de por velas (#72); ventana de historia en días de calendario (#91); comisiones fuera del efectivo (#96, #99); cierre rechazado por pérdida diaria (#95); cierre rechazado por capital y tamaño (#101, mergeado el 2026-10-10 después de esta medición, cortaba momentum-12m sobre XLK).
- **Lo que la librería no cubre:** control de riesgo, cartera compartida con paper, varias posiciones a la vez, persistencia. Tampoco slippage en el adaptador, ni más de un activo por corrida.
- **Principio de arquitectura de Fede:** backtest y paper no pueden diferir por construcción.
- **Licencia:** `backtesting.py` es AGPL-3.0. Hoy solo la importa `scripts/` (dependencia de desarrollo); nada bajo `quantagent/` la usa.

## 6. Opciones

**A · Seguir como hoy.** El motor propio da el número; la librería es control en tests de oráculo. Plan de M2: no cambia ninguna fila. AGPL: sin cambio. M3: sin cambio, el motor que da el número es el que va a paper. Riesgo: el control corre sobre fixtures y sobre SPY. Los números de los otros 11 ETFs salen de un motor que esta semana falló seis veces, sin comparación celda por celda.

**B · Invertir.** Los números de M2 salen de la librería; el motor propio pasa a ser prueba de integración, obligado a coincidir.
- Plan de M2: cambian V20 (el golden congela la salida de la librería), V23 y V24 (`compare` corre el adaptador), V33 y V36 (el veredicto lee trades de la librería). El adaptador tiene que entrar al producto y sumar slippage y curva de equity.
- AGPL: `quantagent/` pasa a importar una librería AGPL. Para uso propio no obliga a nada; si algún día se distribuye o se ofrece como servicio, sí. Es el riesgo que el ADR quería evitar.
- M3: paper opera con el motor propio, así que el número oficial vendría de un motor que no es el que opera. Solo respeta el principio de Fede si la coincidencia es obligatoria en cada celda, y en ese caso el número es el mismo y B no agrega nada sobre C.

**C · Motor propio como fuente, coincidencia obligatoria por celda.** Un número de M2 es oficial solo si, para esa estrategia, activo, período y costos, la librería coincide trade por trade. La celda que no coincide se publica como "sin validar" y abre un ticket.
- Plan de M2: V23 suma una columna "coinciden N de N" por par; V20 congela el número junto con esa marca; V33 no da veredicto sobre celdas sin validar; V29 se reemplaza por el adaptador genérico. Hace falta slippage en el adaptador para comparar con el perfil de costos real y no solo a slippage 0 (un ticket chico).
- AGPL: sin cambio, la librería sigue en `scripts/`. M3: sin cambio. Costo: una corrida más por celda. Con 5 estrategias y 12 ETFs son 60 corridas extra; momentum-12m tarda 5 minutos cada una.

## 7. Recomendación

Elegiría **C**. Los 201 trades coinciden, así que hoy el número es el mismo salga de donde salga: lo que se decide es qué pasa cuando dejen de coincidir. En A nadie se entera fuera de SPY. En B el número publicado deja de ser el del motor que va a operar. C mantiene un solo motor entre backtest y paper y convierte en regla lo que esta semana encontró los seis defectos.

Aparte del motor: el supuesto de llenado pesa más que el slippage. Propongo que la tabla de M2 informe también el PnL con apertura siguiente como sensibilidad, hasta que el motor propio tenga ese modo. Es otra decisión y otro ticket.

Qué me haría cambiar de opinión:
- **Hacia B:** que al correr la matriz de 12 ETFs más de unas pocas celdas no coincidan, o que después de #101 sigan apareciendo defectos del motor al ritmo de esta semana. Ahí el motor no sirve como fuente y conviene publicar con la librería mientras se arregla.
- **Hacia A:** que las 60 corridas extra vuelvan impracticable la matriz y no se pueda acelerar el adaptador.
- **Contra las tres:** que una estrategia de M2 necesite varias posiciones o cartera entre activos. La librería no puede controlar eso y haría falta otro control.

## 8. Decisión (Fede, 2026-10-10)

- **Opción C:** el motor propio da el número de M2, y un número es oficial solo si la librería coincide trade por trade en esa celda (estrategia, activo, período y costos). La celda que no coincide se publica como "sin validar" y abre un ticket.
- **Sensibilidad:** la tabla de M2 informa también el PnL con llenado en la apertura siguiente.
- Consecuencias que arma el PM: slippage en el adaptador (ticket nuevo, antes de V20), la marca "coinciden N de N" en V20 y en V23, y la columna de apertura siguiente en la tabla de comparación.
