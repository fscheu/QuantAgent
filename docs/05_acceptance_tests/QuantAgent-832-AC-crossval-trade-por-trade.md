# QuantAgent-832 — Comparación trade por trade contra backtesting.py

Motor del proyecto vs el port RSI a `backtesting.py` (`scripts/crossval_rsi.py`), sobre `tests/fixtures/spy-90d.csv`
(RSI mean-reversion, 2160 velas de 1h), con slippage 0 y comisión 0.

## Resultado

Los dos motores coinciden: 227 trades con el mismo `entry_time`, `exit_time`, `symbol` y `side`, en el mismo
orden, y las mismas métricas finales dentro de tolerancia. No se encontró ninguna divergencia. Todas las cifras
salen de una sola corrida del comando de abajo (exit code 0).

Comando: `python scripts/crossval_compare.py` (≈90 s; agregar `--intrabar` para el experimento informativo).

| Métrica | Motor del proyecto | Port (backtesting.py) | Dentro de tolerancia |
|---|---|---|---|
| Nº de trades | 227 | 227 | SI (exacto) |
| PnL total | 13084.979 | 13084.979 | SI |
| Win rate | 49.78 % (0.49779736) | 49.78 % (0.49779736) | SI (exacto) |
| Profit factor | 2.8947748 | 2.8947748 | SI |
| Max drawdown | 0.0011644963 | 0.0011644962 | SI |

Diferencia máxima por columna (227 trades): qty 8.48e-09, entry_price 0, exit_price 0, pnl 1.39e-07.
Equity punto a punto: 5.79e-05 USD de máxima diferencia en 2160 instantes.

Tolerancias (detalle y justificación en el docstring de `scripts/crossval_compare.py`): fechas, `symbol`, `side`,
nº de trades y win rate exactos; qty 1e-8 (el motor guarda 8 decimales, error ≤ 5e-9, más la cuantización 2^-27 del
port, ≤ 3.7e-9: cota 8.7e-9); precios 1e-8; pnl `1e-8·|salida−entrada| + 1e-8` por trade; profit factor 1e-6
relativo; max drawdown 1e-6; equity 1e-3 USD (el CSV de equity del motor tiene 4 decimales).

Max drawdown: cada lado con **su propia curva** y la misma fórmula (máximo corrido, `(pico−equity)/pico`,
`scripts/recalc_metrics.py::recalc_equity`): la del motor desde `--equity-out`, la del port desde la curva que
devuelve `backtesting.py`. Ambas están muestreadas en las mismas 2160 velas; `backtesting.py` agrega un instante
extra (la vela centinela del port, 2026-04-01T00:00) que se descarta, y el script verifica que los instantes
sean idénticos antes de comparar. Las métricas del motor se recalculan desde su CSV y además se verifica que lo
que imprimió el CLI coincide con eso. El Sharpe no se compara (no está en el alcance del ticket).

## Qué valida esta comparación

- Que el motor ejecuta bien la **mecánica**: cuándo llena las órdenes (al close de la vela), a qué precio, con qué
  tamaño (`equity · 5 % · confianza / precio`), cómo calcula el PnL de cada trade, el efectivo y el cierre forzado
  al final, contra una implementación independiente de esa mecánica (la librería).
- Que la contabilidad del portfolio coincide: la curva de equity del motor y la de `backtesting.py` son iguales
  vela a vela (5.8e-5 USD, redondeo del CSV), y también el max drawdown, el profit factor y el win rate.
- Que el RiskManager no distorsiona nada en este escenario (ver ítem 2) y que el redondeo a 8 decimales de la
  base no cambia ninguna decisión de este fixture (ítem 5, solo en SQLite).

## Qué NO valida

- **Las reglas de entrada y salida son código compartido por diseño**: el port las re-implementa a partir del
  motor (RSI, umbrales, SL 2 %, TP 3 %, trailing 5 %, tamaño). Si una regla estuviera mal, ambos lados
  estarían igual de mal y la comparación no lo detecta.
- **Stops y take profits se evalúan solo contra el close de la vela** (en ambos lados); nunca contra high/low.
  Ver el experimento informativo: esa decisión cambia el resultado de forma grande.
- Una sola estrategia (RSI), un solo fixture sintético (SPY 1h, 90 días, continuo 24/7), slippage 0 y comisión 0.
  No cubre slippage ni comisiones, ni otros timeframes, ni huecos de datos, ni short/long con gaps reales.
- No cubre la apertura en la última vela (el fixture no la ejercita), ni Postgres (solo se corrió SQLite).
- El port usa el motor de órdenes de la librería; que la librería acierte en lo que hace no implica que su
  modelo (mercado al close, sin spread) sea el que se quiere en producción.
- El tamaño de las posiciones (≤ 5 % del portfolio) hace que el riesgo (rechazos, pérdida diaria) nunca
  intervenga: esos caminos de código no quedan ejercitados.

## Divergencias

Ninguna. Todo cae dentro de las tolerancias de arriba: la mayor diferencia observada (qty 8.48e-09, pnl 1.39e-07)
está dentro de la cota teórica del redondeo a 8 decimales más la cuantización del port. Para que esta frase
significara algo, el test `test_comparison_fails_and_names_the_altered_trade` altera 0.5 USD el pnl de un trade
del port y comprueba que el script sale con 1 y nombra ese trade (nº 6) y la columna `pnl`.

Esto es `explicada por semántica` solo en el ruido numérico (ítems 1 y 6); no hay ninguna `NO explicada`.

## Diferencias de semántica

Estado final de los 12 ítems del docstring de `scripts/crossval_rsi.py`:

1. Cantidad cuantizada a 2^-27: **confirmado** — diferencia máxima 8.48e-09 en qty, media con signo −1.3e-10 (sin sesgo).
2. Rechazos de riesgo: **confirmado, 0 rechazos** — corrida instrumentada del mismo CLI (envuelve `validate_trade`):
   454 llamadas (2 por trade), 0 rechazos; el CSV de trades es idéntico al de la corrida normal.
3. Pérdida diaria con `date.today()`: **confirmado que no dispara** — el término realizado usa
   `closed_at >= hoy` (hoy = 2026-10-08, el fixture es de ene-mar 2026) y vale 0; margen mínimo medido al límite
   4918.53 USD (<0 rechazaría); circuit breaker activo 0 veces. Sí podría actuar si el reloj cayera dentro de las fechas del fixture.
4. Valor de portfolio para el tamaño: **confirmado** — qty coincide en los 227 trades y la equity en 2160 velas.
5. Redondeo de DB: **sin confirmar** — en SQLite no tuvo efecto (227 `exit_time` iguales); Postgres no se corrió
   (las corridas del CLI se hacen sobre SQLite), así que un empate exacto en Postgres no se puede descartar desde acá.
6. PnL de salida en Decimal vs float: **confirmado** — diferencia máxima 1.39e-07, media con signo −4.9e-09.
7. Última vela: **parcial** — cierre forzado `backtest_end` confirmado (trade 227: mismo `exit_time` 2026-03-31T23:00
   y precio en ambos); apertura en la última vela **sin confirmar** (la última entrada es 2026-03-31T17:00).
8. Ventana de 7 días vs historial completo: **confirmado para este fixture** — velas continuas cada 1 h (un único
   delta, 2160 filas) y 227 trades idénticos; con huecos no se probó.
9. Hora/precio de fill: **confirmado** — `entry_time`/`exit_time` iguales en los 227 trades; precios con diferencia 0.
10. Orden del CSV: **confirmado** — comparado en orden; 227 `entry_time` únicos y ningún solapamiento.
11. Capital inicial: **confirmado** — ambos 100000.0 (tabla de parámetros del port y primera fila de equity del motor).
12. Trailing stop: **confirmado que no dispara** — 0 salidas `TRAILING_STOP` en el motor (113 SL, 113 TP, 1
    `backtest_end`), e inalcanzable aritméticamente (nivel de trailing < 0.9785·E < 0.98·E = SL, que se evalúa antes).
    El port no se instrumentó: sus 227 salidas coinciden en hora con las del motor.

## Experimento informativo: stops dentro de la vela

**No forma parte del pass/fail.** Variante del port con `sl=`/`tp=` nativos de `backtesting.py` (se evalúan contra
high/low de cada vela; el trailing sigue al close). Comando: `python scripts/crossval_compare.py --intrabar`.

| | Solo al close (motor y port) | Intravela (`sl=`/`tp=`) |
|---|---|---|
| Trades | 227 | 340 |
| PnL total | 13084.98 | 5803.02 |
| Win rate | 49.78 % | 33.53 % |

De los 227 trades del esquema "solo al close", solo 25 tienen una entrada que también existe en la variante
intravela, y los 25 salen en otro instante (el primer trade ya difiere; después las secuencias de entradas
divergen, por lo que ese conteo no mide efecto trade a trade). Conclusión acotada: evaluar los stops solo al close
cambia mucho el resultado en este fixture; el motor del proyecto no se tocó y no se afirma cuál modelo es "correcto".
