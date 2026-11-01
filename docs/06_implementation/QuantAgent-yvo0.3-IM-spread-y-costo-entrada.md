# Spread y costo fijo de entrada

`backtest run` admite `--spread-bps` y `--costo-entrada-usd` solo con `--cuenta-chica`. El spread se aplica por lado como slippage adicional y se exporta en columnas separadas para entrada y salida. El costo fijo se descuenta del capital inicial y se informa aparte.

Los valores iniciales por clase están en `SPREAD_BPS_POR_CLASE` en `quantagent/settings.py`; son estimaciones, no mediciones de mercado. El spread explícito de CLI reemplaza el valor estimado para la corrida.

Validación: pruebas de CLI y snapshot de backtest.
