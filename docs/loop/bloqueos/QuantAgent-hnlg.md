# QuantAgent-hnlg: anualización contradictoria en V33.2

El ticket pide Sharpe con 252 períodos anuales en diario y, a la vez,
igualdad con `Backtest._calculate_sharpe_ratio(0.0)` al cierre.
El motor usa la frecuencia observada, no 252: `_get_periods_per_year`
en `quantagent/backtesting/backtest.py:1677` divide los intervalos de la
curva por los años transcurridos (365.25 días por año).

Intenté resolver la fórmula leyendo ambos métodos y ejecutando el cálculo
real del motor sobre cuatro puntos diarios de equity: 100, 102, 101, 104.
Frecuencia observada: 365.25; Sharpe del motor: 12.345833.
Con los mismos retornos y desviación estándar, anualizar con 252 da 10.254762.
Por lo tanto, ambos requisitos no pueden cumplirse juntos en esa curva.
El fixture `spy-2y-1d.csv` también incluye días consecutivos de calendario.

No se escribió código ni se cambió el motor. Falta decidir:
A. Replicar la frecuencia observada del motor, conservando la igualdad.
B. Usar 252, aceptando que el Sharpe del adaptador difiera del motor y
   ajustando el criterio de aceptación (cambiar el motor está fuera de alcance).
Recomiendo A para preservar la comparación con el número oficial.

Reproducción: el comando Probar del PR ejecuta los métodos reales del motor,
sin base de datos ni snapshot, e imprime las tres cifras anteriores.
El ticket queda `in_progress` hasta que el PM precise la anualización.
