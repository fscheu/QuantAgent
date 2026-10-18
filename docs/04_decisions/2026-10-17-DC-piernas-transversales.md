# Piernas transversales: cómo una estrategia que rankea activos entra a las etapas 1 a 3

Fecha: 2026-10-17. Estado: **propuesta técnica; las decisiones de dinero (Q1 pesos, Q6 fracción) ya las tomó Fede el 2026-10-17** (ver al final). No cambia código.
Ticket: P03 de L13 (sin ID en BEADS todavía; lo asigna la VM). Plan: [`QuantAgent-d4cq-PL-l13-l14-reescritura-de-tickets`](../02_planning/QuantAgent-d4cq-PL-l13-l14-reescritura-de-tickets.md) §4 (P03), §7 (Q1, Q6).
Relacionado: [`2026-10-17-DC-cartera-de-primas`](./2026-10-17-DC-cartera-de-primas.md) (D1 a D7), `quantagent/fabrica/{filtro,cartera,validacion,admision}.py`, `quantagent/backtesting/signals.py`.

## En lenguaje llano

**Pierna por activo (lo que hay hoy).** Cada estrategia mira un activo solo y decide si estar comprado o no. Ejemplo: "SPY sube respecto de hace 12 meses, entonces compro SPY". Si hay 29 fondos, hay 29 decisiones independientes; el fondo A no sabe nada del fondo B.

**Pierna transversal (lo que falta).** La estrategia mira **todos los fondos a la vez** y los compara entre sí. Ejemplo: a fin de mes ordeno los 29 fondos por cuánto subieron en 12 meses y compro el tercio de arriba (9 fondos), sin importar si el mercado en general sube o baja. Es la prima "ganadores contra perdedores" (Jegadeesh y Titman 1993). La baja volatilidad funciona igual: ordeno por qué tan calmo es cada fondo y compro los más calmos.

**Por qué hace falta un diseño antes de programar.** El embudo actual está armado para un activo por vez: `filtrar_activo` recibe las velas de **un** fondo y elige los parámetros **para ese fondo**. Un ranking no cabe ahí, por cuatro razones:

1. La decisión sobre un fondo depende de los otros 28. No se puede calcular mirando uno solo.
2. Hay una trampa nueva: si el ranking de hoy usa, sin querer, el precio de mañana de *otro* fondo, la prueba de "no mira el futuro" que hacemos fondo por fondo no lo detecta y el backtest sale inflado.
3. El control contra `backtesting.py` (etapa 2) hoy corre una estrategia por fondo, y esta no se puede correr por fondo.
4. Los pesos: en las piernas actuales cada fondo tiene su lugar fijo en la cartera. Acá el conjunto de fondos comprados cambia todos los meses, y hay que decir cómo se reparte la plata entre los elegidos (esto es de Fede, ya decidido).

Este documento fija las cuatro cosas antes de que alguien toque `filtro.py` y `validacion.py` en medio de la cadena de L12.

## Resumen

| # | Punto | Recomendación | Estado |
|---|---|---|---|
| T1 | Señal | Función pura sobre el panel de cierres (fecha × activo); prueba de truncamiento sobre todo el panel a la vez | propuesta |
| T2 | Etapa 1 | Grilla y ganador **por pierna**; `filtrar_pierna` nueva, no se reutiliza `filtrar_activo` | propuesta |
| T3 | Etapa 2 y 3 | Cada celda (pierna, activo) alimenta a los dos motores con la **misma serie 0/1** | propuesta |
| T4 | Pesos | 1/S entre piernas y 1/σ entre los activos elegidos (opción A) | **decidido por Fede (Q1)** |
| T5 | Fracción | Tercio fijo del universo, sin grilla | **decidido por Fede (Q6)** |
| T6 | Tickets | P05 se parte en cuatro; choca con `admision.py` (P02) y con el test de P01 | propuesta |

## T1. Señal

**Qué se decide.** Cómo se expresa el ranking para que (a) use solo cierres hasta la sesión de decisión y (b) la prueba de truncamiento cubra todos los activos a la vez.

Propuesta (A, recomendada):

- `señales_transversales(cierres, metrica, fraccion=1/3) -> DataFrame` en `quantagent/fabrica/transversal.py`. `cierres` es el panel fecha × activo del universo declarado en el YAML; `metrica` es una función pura del panel (momentum: `cierre[t-21] / cierre[t-252] - 1`, el "12 menos 1" de la literatura; volatilidad: desvío de los retornos diarios de las últimas 252 sesiones, orden ascendente). Las constantes 21 y 252 son fijas: 0 parámetros libres. El detalle numérico de cada métrica lo fija su ticket (P06, P07).
- La decisión se toma solo en la última sesión del mes (`is_last_session_of_month`, el mismo calendario NYSE de `momentum_12m_strategy.py`). Ese día se ordenan los activos **que tienen la historia completa de la métrica** y se eligen los `k = max(1, N_disponibles // 3)` primeros (desempate por ticker, alfabético, para que sea determinista). Con menos de 3 disponibles no hay pierna ese mes (efectivo).
- Devuelve un panel 0/1 fecha × activo: el estado a mantener desde la apertura de la sesión siguiente, repetido todos los días hasta la próxima decisión. Repetir hacia adelante es causal. El motor llena a la apertura siguiente (`--llenado apertura`), igual que hoy.
- Prueba de truncamiento: `check_no_lookahead(signal_fn, candles, cuts)` ya acepta cualquier tabla con índice de fechas, así que se le pasa **el panel de cierres** y `señales_transversales` como `signal_fn`. Trunca todas las columnas en el mismo corte, que es justo lo que hace falta; no se agrega una función nueva.

Alternativa B (rechazada): correr la prueba por activo con el ranking calculado adentro de cada activo. No detecta que el activo A use el futuro del activo B, que es el riesgo propio de esta clase de estrategias.

Sobre `k`: la alternativa de fijar `k` con el tamaño del universo declarado (9 con 29 activos) elige casi todo cuando hay pocos activos con historia (los ETFs que nacieron después de 2007). Por eso `k` sale de los disponibles ese día.

## T2. Etapa 1: parámetros por pierna y retornos fuera de muestra

**Qué se decide.** Hoy `filtrar_activo` elige, por ventana de ajuste, la combinación de mayor `sharpe_tasa_0` **para un activo**. En un ranking, usar parámetros distintos por activo no tiene sentido: todos tienen que competir con la misma métrica.

Propuesta (A, recomendada): `filtrar_pierna(h, pierna, paneles, *, costos, capital, train_years, test_years)` en `filtro.py`:

1. Calcula una vez, por combinación de la grilla de la pierna, el panel 0/1 completo (válido porque la admisión verificó que no mira el futuro, igual que hoy).
2. Con las mismas ventanas (`construir_ventanas`, 4 años de ajuste y 2 de prueba), elige **una** combinación para toda la pierna por el `sharpe_tasa_0` de la curva de la pierna en el ajuste, y la corre en la prueba. Con Q6 resuelto y las constantes fijas la grilla de las clases de P06 y P07 tiene una sola combinación; la función igual debe soportar más, porque una pierna futura puede traer una.
3. Para cada activo corre `run_filter` con su columna 0/1 (así los costos y el llenado son los de siempre) y devuelve `retornos`: retornos diarios fuera de muestra **por activo** (columnas `(pierna, activo)`), que es lo que `cartera.combinar` ya acepta junto a las piernas por activo. `trades` es la suma de las entradas de todos los activos, y cuenta en el mínimo de 30 de la cartera.

Alternativa B (rechazada): reutilizar `filtrar_activo` inyectándole la señal. Seguiría eligiendo parámetros por activo y devolvería un veredicto por activo, que D3 ya descartó como nivel de juicio.

El veredicto (`veredicto_etapa1`) sigue siendo de la cartera (D3); una pierna transversal aporta sus `retornos` y sus trades, no una fila propia de PASA/CORTA por activo.

## T3. Etapas 2 y 3: control contra `backtesting.py`

**Qué se decide.** `validar_etapa2` corre `backtest compare --strategies <clase> --symbols <activo>` y después `compare_control.py` por activo. Una clase transversal no se puede instanciar por activo.

Propuesta (A, recomendada): una estrategia interna `SenalExterna` (`quantagent/strategy/senal_externa_strategy.py`) con un parámetro `ruta`, un CSV `timestamp, posicion` con la columna 0/1 de ese activo. Entra largo cuando el estado pasa de 0 a 1 y sale cuando pasa de 1 a 0. `correr_motor` escribe el CSV de cada celda y lo pasa en `--params`, de modo que **los dos motores ejecutan exactamente la misma serie** y "todas las celdas coinciden" sigue significando lo que significa hoy.

- Lo que el control **no** cubre: que el ranking esté bien calculado. Eso queda en tests de `transversal.py`: valores conocidos, la prueba de truncamiento de T1 y una implementación independiente y lenta (bucle por mes) comparada contra la vectorizada sobre un panel aleatorio. Es una limitación consciente y se escribe en el docstring de la función.
- Una celda sin ningún trade en el motor propio no se controla (no hay nada que comparar). Un activo que nunca entra al tercio superior no es un fallo.
- V30 por celda (`evaluar`) no tiene sentido para un activo que está elegido un tercio del tiempo; para las celdas transversales se exige solo el control, y el juicio de rendimiento queda en la curva de la cartera (D3). Cómo resuelven esto las piernas por activo en `validacion.py` lo define L12 antes de P05; P05 se alinea con lo que haya quedado mergeado.
- Etapa 3: misma mecánica sobre 2019–2022 con el candado abierto. Los paneles siempre se leen con el lector de snapshot que respeta el candado y el panel de etapas 1 y 2 se recorta a 2018-12-31 **antes** de calcular el ranking; no se leen los archivos a mano.
- `senal-externa` no es una clase válida en un YAML de hipótesis (lee una ruta arbitraria): `admitir` la rechaza.
- Las celdas no registran intentos (D7); la fila de la cartera por etapa sí.

Alternativa B (rechazada): que cada estrategia por activo lea el snapshot y recalcule el ranking adentro. Rompe el contrato de las estrategias (reciben solo las velas de su activo) y deja el cálculo del ranking repetido 29 veces dentro de la prueba de truncamiento.

## T4. Pesos (Q1, decidida por Fede)

`cartera.pesos` hoy da a cada pierna 1/S y, dentro de ella, 1/σ sobre **todos** sus activos, recalculado la primera sesión de cada mes con σ de las últimas 60 sesiones hasta la sesión previa. Para una pierna transversal hay tres formas de repartir. Ejemplo con S = 3 piernas, 29 activos y 9 elegidos:

| Opción | Regla | Peso aproximado por activo elegido | Exposición de la pierna |
|---|---|---|---|
| **A (decidida)** | 1/S entre piernas; 1/σ **entre los elegidos** (suma 1/S) | 1/(3·9) ≈ 3,7 % del capital antes de ajustar por σ | Toda la pierna invertida cada mes |
| B | 1/σ sobre **todo el universo**; los no elegidos quedan en efectivo (como hoy cuando una señal por activo está apagada) | 1/(3·29) ≈ 1,1 % | Cerca de un tercio de 1/S: poca ventaja de la pierna |
| C | Igual dinero entre los elegidos (1/(S·k)) | 3,7 % exacto | Igual que A, sin ajustar por σ |

Fede eligió **A** el 2026-10-17. Queda registrado sin reabrirlo. Consecuencias que los tickets deben respetar:

- **Alineación temporal.** El ranking se decide en la última sesión del mes y los pesos se recalculan la primera sesión del mes siguiente: son la misma fecha de rebalanceo. Los pesos de A usan la elección de ese día y σ hasta la sesión previa; ninguno mira el futuro.
- **Implementación mínima.** `pesos(cierres, piernas, seleccion=None)`: `seleccion` es un diccionario pierna → panel 0/1. Las piernas con selección reparten 1/S entre los elegidos del mes; las demás siguen igual. Las piernas transversales cuentan en S. `combinar` no cambia. Los activos no elegidos tienen peso 0.
- **Costos de rebalanceo de pesos.** Hoy la reasignación mensual de pesos no se cobra como operación; sigue sin cobrarse. Los costos de entrar y salir de cada activo sí están en `run_filter` (T2).
- **Riesgo conocido.** Con A, el capital de la pierna está siempre invertido en la renta variable ganadora, sin la salida a efectivo de las piernas por activo. En caídas fuertes que siguen a mercados alcistas el momentum transversal pierde mucho (Daniel y Moskowitz 2016, *Momentum crashes*; DOI sin verificar). No hay parámetro para corregirlo y no se agrega uno (D1); el DSR de la cartera lo va a reflejar.
- Si se agregaran más piernas, el 1/S se achica para todas. Eso mueve los pesos del núcleo ya probado y obliga a volver a correr su etapa 2 con los nuevos pesos.

## T5. Fracción del universo (Q6, decidida por Fede)

Fede eligió el **tercio fijo** de la literatura el 2026-10-17. La alternativa (grilla de 1/4, 1/3, 1/2) suma un parámetro libre por pierna, sube N en el Sharpe deflactado de todas las hipótesis y es justo el tipo de elección que D1 pide no hacer. Consecuencias:

- `k = max(1, N_disponibles // 3)`, con 29 activos son 9. No es parámetro: no aparece en la grilla ni cuenta en `MAX_PARAMETROS_LIBRES`.
- Baja volatilidad (P07) se rankea dentro de **una clase de activo** (acciones). Si el universo declarado de esa clase tiene menos de 6 ETFs, el tercio es 1 solo y la pierna queda demasiado concentrada; P07 tiene que contar los ETFs de acciones del snapshot y declararlo antes de correr.

## T6. Tickets de implementación y qué choca con la cadena de L12

El plan prevé partir P05 en dos; con esto conviene partirlo en cuatro (cada uno ≤ 100 líneas de diff, regla del plan):

| Ticket | Qué | Archivos | Máquina | Choca con |
|---|---|---|---|---|
| P05.1 | `señales_transversales`, `METRICAS` (registro propio) y sus tests: valores conocidos, truncamiento sobre el panel, implementación independiente | `quantagent/fabrica/transversal.py` (nuevo), `tests/` | nube | Nada: archivo nuevo. El plan lo deja esperando `.55`; el PM puede adelantarlo |
| P05.2 | `pesos(..., seleccion=)` y test con pierna transversal + piernas por activo | `quantagent/fabrica/cartera.py`, `tests/` | nube | Todo ticket de L12 que toque `cartera.py` |
| P05.3 | `filtrar_pierna`; `admitir` reconoce clases transversales (hace la prueba de truncamiento sobre el panel, no sobre velas de un activo) | `quantagent/fabrica/filtro.py`, `quantagent/fabrica/admision.py` | VM | `filtro.py` (cadena L12); **`admision.py` lo toca también P02** y el ensanche de `UNIVERSO_M2` |
| P05.4 | `SenalExterna` + alias interno; `correr_motor`/`validar_etapa2`/`validar_etapa3` alimentan celdas transversales | `quantagent/strategy/senal_externa_strategy.py` (nuevo), `quantagent/strategy/registry.py`, `quantagent/fabrica/validacion.py` | VM | `validacion.py` (cadena L12) |

Cosas que el plan no listaba y P05 tiene que tocar:

- **`admision.py`**: `admitir(h, velas)` recibe las velas de **un** activo y chequea `clase` contra `STRATEGY_REGISTRY`/`cli_aliases()`. Una clase transversal necesita el panel y no está en esos registros. Se resuelve con un registro propio (`transversal.METRICAS` y `es_transversal(clase)`) en vez de disfrazar la clase de `TradingStrategy`. Coordinar con P02 (que reescribe `admitir`): el que llegue segundo rebasa.
- **Test de P01** (`tests/test_catalogo_primas.py`): hoy exige que `clase` sea `null` o un alias de `cli_aliases()`; al fijar `momentum-transversal` y `baja-volatilidad-transversal` (P06, P07) el test debe aceptar también `METRICAS`.
- **P02 (crítico)**: no requiere cambios. Compara `clase`, `parametros`, `activos` de las piernas, y las transversales los tienen.
- **P06 y P07** pasan a ser "una entrada en `METRICAS` + el YAML": la clase no es un `TradingStrategy`, así que el "registrada en `registry.py`" de su criterio de aceptación se reemplaza por "registrada en `METRICAS`".

Orden sugerido: P05.1 (puede ir ya) → P05.2 → P05.3 → P05.4 → P06 y P07 en paralelo. P05.3 y P05.4 esperan `.55` y van a la VM porque editan `filtro.py` y `validacion.py` en medio de la cadena.

## Decisiones de Fede

2026-10-17 (plan §7): aprobó el plan y las recomendaciones de Q1 y Q6. Este documento no las reabre:

- **Q1 → T4, opción A:** 1/S entre piernas, 1/σ entre los activos elegidos.
- **Q6 → T5:** tercio fijo del universo; sin grilla.

T1 a T3 y T6 son diseño técnico que no mueve números de dinero. Los revisa el PM de la VM. Si un cambio posterior tocara T4 o T5 (pesos o fracción), vuelve a Fede.

## Estado

Propuesta técnica con T4 y T5 decididas. Cuando el PM la acepte, se propone agregarla a `docs/04_decisions/README.md` (Current Decisions) y enlazarla desde P05 en BEADS.
