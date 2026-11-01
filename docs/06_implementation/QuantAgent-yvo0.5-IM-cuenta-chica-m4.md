# Cuenta chica sobre las carteras de M4 (CC-05)

Ticket `QuantAgent-yvo0.5`, lote L33. Solo datos de desarrollo: 2007-01-03 a 2018-12-31. No se abrió la prueba 2019-2022 ni la reserva 2023+.

## Qué se corrió

Las dos carteras que operaron en paper en M4 (L21): `cartera-nucleo-curva` y `cartera-transversal`. El paper de M4 se apagó el 2026-10-30; hoy solo corre M5.

Cada una a USD 1.000, 2.500 y 5.000 con acciones enteras, 5 % de caja, perfil de costos `etf` más spread por clase, retención de dividendos de 30 % y rebalanceo mensual. Se compara con comprar y mantener SPY con los mismos costos y retención.

Datos: snapshot `etf-1d-2026-10-amplio` (Yahoo, bajado el 2026-10-17) y, para la curva, `macro-2026-10` (ALFRED, 2026-10-17). No se pueden regenerar iguales.

Comandos (desde la raíz del repo, con `QUANTAGENT_SNAPSHOT_DIR=local-data`):

```bash
quantagent backtest cuenta-chica --hipotesis hipotesis/cartera-transversal.yaml \
  --snapshot etf-1d-2026-10-amplio --costos etf --to 2018-12-31
quantagent backtest cuenta-chica --hipotesis hipotesis/cartera-nucleo-curva.yaml \
  --snapshot etf-1d-2026-10-amplio --macro-snapshot macro-2026-10 --costos etf --to 2018-12-31
```

La transversal tarda unos minutos y la núcleo-curva unos cuatro.

## Lo que mejor se lee

La referencia, SPY comprar y mantener, rinde 113,92 % con Sharpe 0,42 en todo el período. Ninguna cartera la supera en retorno a ningún capital. La transversal se acerca, la núcleo-curva queda lejos.

## Cartera transversal

| Capital | Retorno | Sharpe | Costos pagados | Contra SPY |
|---|---|---|---|---|
| USD 1.000 | 64,27 % | 0,52 | USD 141 | -49,65 pts |
| USD 2.500 | 77,99 % | 0,51 | USD 420 | -35,94 pts |
| USD 5.000 | 80,47 % | 0,50 | USD 887 | -33,46 pts |

Posiciones que quedaron en cero alguna vez: 19 ETFs a USD 1.000, 7 a USD 2.500 (FXE, FXI, GLD, LQD, QQQ, TLT, USO) y 3 a USD 5.000 (FXE, GLD, USO). Posición más concentrada: 82 %, 91 % y 93 %.

Qué se rompe: con poco capital la cartera no se puede armar. A USD 1.000 no entran 19 de los ETFs que debería tener, y la cartera queda concentrada en pocas posiciones (hasta 82 % en una). Los costos son pesados: a USD 5.000 se pagan USD 887 en 12 años, 18 % del capital.

## Cartera núcleo-curva

| Capital | Retorno | Sharpe | Costos pagados | Contra SPY |
|---|---|---|---|---|
| USD 1.000 | 2,80 % | 0,28 | USD 26 | -111,12 pts |
| USD 2.500 | 9,97 % | 0,60 | USD 108 | -103,96 pts |
| USD 5.000 | 15,16 % | 0,67 | USD 267 | -98,76 pts |

Posiciones en cero alguna vez: 29 ETFs de 29 a USD 1.000, 28 a USD 2.500 y 24 a USD 5.000. Posición más concentrada: 60 %, 69 % y 81 %.

Qué se rompe: esta cartera reparte el capital entre tres piernas y 29 ETFs, así que cada ETF recibe una fracción de uno o dos por ciento (inferido de la regla de pesos, no medido). A USD 2.500 eso son pocas decenas de dólares por ETF, y casi todos cuestan más que eso. La cartera queda casi toda en efectivo, y el retorno de 2,8 % a 15 % es el de una cuenta que casi no invierte, no el de la estrategia. El Sharpe de 0,60 a 0,67 no dice nada del edge, y no se debe comparar con el de la cartera a USD 100.000.

## Lectura

1. Con USD 2.500 la núcleo-curva no se puede operar tal cual: 28 de 29 ETFs quedan en cero. Si sigue como candidata, necesita un universo mucho más corto, no más capital.
2. La transversal sí se arma desde USD 2.500, con siete huecos. Pierde 36 puntos contra SPY y paga USD 420 de costos.
3. Ninguna de las dos gana a SPY comprar y mantener con USD 2.500 en este período.
4. Todo esto es en el período de desarrollo. No dice cómo se comportarían en 2019-2022 ni después.

## Para reproducir y revisar

Cada tarjeta sale del comando de arriba, sin redondeos manuales. Cruce hecho por otro camino: el cierre ajustado de SPY da +126,1 % bruto entre 2007-01-03 y 2018-12-31; descontando la retención de 30 % sobre dividendos y los costos de entrada y salida queda en el 113,92 % de la tarjeta, diferencia esperable para esos costos. El cierre sin ajustar da +76,8 % solo por precio. La revisión del PM puede recalcular una celda de capital y costos a mano con la regla de `ordenes_rebalanceo`.

## Límites de esta corrida

- El período termina en 2018, con la misma regla de `--to` que la cartera. Los números no son de paper.
- "En cero" cuenta el ETF si alguna vez en el período no alcanzó para una acción; no dice cuánto tiempo.
- La retención de 30 % y el spread son supuestos del lote (CC-02 y CC-03), no mediciones.
