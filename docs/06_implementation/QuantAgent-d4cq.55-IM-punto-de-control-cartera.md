# QuantAgent-d4cq.55: punto de control de L12

Corrida del PM el 2026-10-18 de `backtest hipotesis correr hipotesis/cartera-tendencia-momentum.yaml --snapshot etf-1d-2026-10-amplio` sobre `local-data`, con el código de `lote/cartera` en `f6a17d62` (después de C08). Fede autorizó abrir la prueba (2019–2022) para esta cartera el mismo día. Solo agregados, sin precios.

La cartera tiene dos piernas sobre los 29 ETFs del snapshot amplio, con 0 parámetros libres: `momentum-12m` (lookback 252) y `sma-cross` (fast 1, slow 210), con pesos por igual riesgo.

| Etapa | Período | Veredicto | Sharpe a tasa 0 | Trades | Ganancia (sobre 100.000) |
|---|---|---|---:|---:|---:|
| 0 admisión | | PASA | | | |
| 1 filtro, fuera de muestra | 2007–2018 | PASA (Sharpe de cartera 0,64) | 0,64 | 1.139 | |
| 2 control contra `backtesting.py` y V30 | 2007–2018 | PASA | 0,76 | 1.471 | 35.190 |
| 3 prueba, DSR | 2019–2022 | CORTA (DSR de cartera < 0,95) | 0,25 | 405 | 2.818 |

## Lectura

- La cartera pasó el filtro y el control, pero en la prueba el Sharpe cayó de 0,76 a 0,25 y el DSR quedó por debajo de 0,95: con el total de intentos registrados no se puede descartar que el resultado de 2007–2018 sea suerte.
- Fila por cartera y etapa en `intentos.csv` (D7): tres filas con `activo = cartera`, una por etapa.
- La prueba de `cartera-tendencia-momentum` quedó abierta en `hipotesis-candado.csv` (13:33 UTC); se abre una sola vez, esta hipótesis no se puede reintentar.
- En la etapa 2 el motivo lista fallas de celdas sueltas (DBC, EEM, EWZ, USO y otras con pnl <= 0, Sharpe < 0,10 o drawdown > 0,35) aunque el veredicto de la cartera es PASA, porque el juicio es de la cartera completa. El mensaje es confuso y conviene separar las notas por celda del motivo del veredicto.
- No se cambió ningún umbral ni peso según el resultado.
