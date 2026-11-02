# Retención sobre dividendos implícitos

El CLI `backtest run` acepta `--retencion-dividendos` y `--costo-fiscal-pct`. En modo `--cuenta-chica`, la retención predeterminada es 30%; fuera de ese modo es 0. El costo fiscal se suma a la tasa y queda como parámetro para incorporar ganancias/bienes personales.

Para snapshots que exponen `close_sin_dividendos`, el CLI deriva el rendimiento implícito diario de dividendos comparando el retorno ajustado con el retorno del cierre sin dividendos. El motor descuenta de caja la fracción efectiva sobre el valor de las posiciones abiertas al cierre previo. Los fixtures sin esa columna no generan descuentos. Con tasa efectiva cero, el flujo existente no cambia.

Validación: `pytest tests/test_backtest*.py` y los tests de snapshot/retención.
