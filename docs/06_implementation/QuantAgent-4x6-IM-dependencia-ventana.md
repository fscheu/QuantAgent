# QuantAgent-4x6 — ¿La señal depende del largo de la ventana de historia?

**Respuesta corta:** de las 5 estrategias, solo Triple Screen cambia de respuesta según cuántas velas recibe. En RSI no cambia ninguna señal: lo único que se mueve es desde qué vela empieza a opinar el motor.

Medido el 2026-10-10 sobre SPY 2007-01-03 a 2018-12-31 (3020 velas diarias, snapshot `etf-1d-2026-10`), sin tocar nada bajo `quantagent/`.

## Palabras

- **Ventana:** las últimas N velas que el motor le pasa a la estrategia en cada fecha (`backtest.py:952`, `_get_history_df`).
- **Largo actual:** el N que pide cada estrategia con `required_history_bars`.
- **Fecha comparable:** fecha donde las dos ventanas (la actual y la otra) están completas. Solo ahí tiene sentido decir "cambia la señal".
- **Solo arranque:** fecha donde la ventana actual ya estaba completa y dio señal, pero la más larga todavía no se llenó y el motor no pregunta (`backtest.py:845`). El dato es el mismo; la señal se pierde por empezar más tarde.

## Tabla: fechas comparables donde la señal es distinta a la del largo actual

| estrategia (largo actual) | +1 vela | +5 velas | el doble |
|---|---|---|---|
| rsi (30) | 0 | 0 | 0 |
| fifty-two-week-high (303) | 0 | 0 | 0 |
| triple-screen (78) | **1** | 0 | 0 |
| sma-cross (201) | 0 | 0 | 0 |
| momentum-12m (253) | 0 | 0 | 0 |

Señales perdidas solo por el arranque: rsi 1, 4 y 9 (con 31, 35 y 60 velas, sobre 752 señales); las otras cuatro, 0. La condición de salida de sma-cross y momentum-12m (las únicas con `should_exit` propio) tampoco cambia: 0 fechas en los tres largos.

La salida completa del script, con fechas comparables y cantidad de señales por fila, está en el PR.

## Triple Screen: la tendencia sí depende de la ventana

La tabla muestra 1 sola fecha (2015-12-28: con 79 velas da SHORT, con 78 no da nada) porque sobre SPY diario la estrategia casi no emite: 1 señal en 2943 fechas. El problema está una capa más abajo, en la pantalla 1 (la tendencia "semanal"), que se mide aparte:

| largo | fechas comparables | tendencia distinta a la de 78 |
|---|---|---|
| 79 (+1) | 2942 | 191 (6,5%) |
| 83 (+5) | 2938 | 53 (1,8%) |
| 156 (doble) | 2865 | 285 (9,9%) |

Dos causas, las dos en `quantagent/strategy/triple_screen_strategy.py`:

1. **Líneas 133 y 137** (`_aggregate_weekly_bars`): las "semanas" son bloques de 5 velas contados desde la primera vela de la ventana, y lo que sobra se descarta del final. Con 78 velas entran 15 bloques (75 velas) y las 3 más nuevas no cuentan para la tendencia; con 79 quedan afuera 4. Por eso la tendencia con 79 velas es exactamente la de ayer con 78, y el número 191 coincide con la cantidad de veces que la tendencia de 78 cambia de un día al siguiente. Además los bloques se corren una vela por día: no son semanas de calendario.
2. **Línea 152** (`_ema`, usada en la 166): la media exponencial de 13 períodos arranca en el primer bloque de la ventana y solo tiene 15 puntos, así que todavía arrastra el valor inicial. La fila +5 aísla esta causa (misma fase de bloques, una semana más de historia): 53 fechas.

**Arreglo propuesto:** armar las semanas con el calendario (lunes a viernes según `timestamp`), incluyendo siempre la vela más nueva, y pedir historia suficiente para que la media exponencial ya no dependa del arranque (unas 4 veces su período).

En el fixture `spy-1y-4h` (el de la referencia de M1, 6 señales) el script da 0 cambios en los tres largos: la referencia no se mueve por esto.

## RSI

No hay dependencia: `rsi_strategy.py:141-142` promedia las últimas 14 diferencias, sin memoria de lo anterior. El "corrimiento en el arranque" que se vio en #91 es el del motor: con ventana más larga las primeras velas no se evalúan. No hay nada que arreglar en la estrategia; al comparar dos lookbacks hay que empezar a contar desde la misma fecha.

## Cómo se midió y qué no cubre

- `python scripts/medir_dependencia_ventana.py --snapshot etf-1d-2026-10 --symbol SPY --from 2007-01-03 --to 2018-12-31 --detalle` (unos 15 minutos). Lee las velas con `_get_history_df` y recorta la ventana; un test comprueba que el recorte es igual a lo que devuelve el motor.
- sma-cross y momentum-12m no están en esta rama: se copiaron de `loop/QuantAgent-jgn` (f9f3804b; el archivo de sma-cross es idéntico al de `loop/QuantAgent-cok`) a una carpeta fuera del repo y se pasaron con `--strategy-dir`.
- El script le pregunta a la estrategia en todas las fechas. El motor solo pregunta cuando no hay posición abierta, así que estos números son respuestas de la estrategia, no trades.
- La tabla de tendencia sale de un fragmento que llama a `_aggregate_weekly_bars` y `_screen1_trend` con las mismas ventanas; está en el PR.
- **No medido:** la rama con datos en vivo de `_get_history_df` (`backtest.py:970-978`) pide la ventana por días de calendario, así que la cantidad de velas varía sola de una fecha a otra. Por la causa 1 eso alcanza para mover la tendencia de Triple Screen. Tampoco se revisó qué ventana arma paper.
