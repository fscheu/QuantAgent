# Orden OTO para entrada y protección M5

`AlpacaBroker.submit_oto` envía una sola `StopOrderRequest` con `order_class=oto` y `stop_loss` cuantizado. Persiste el identificador de la orden principal y el de la pierna stop para que el seguimiento y `replace_stop` usen la pierna remota correcta. `correr_diario` conserva el envío separado para brokers sin `submit_oto`.

El SDK Alpaca instalado documenta en `OrderRequest.qty` que cantidades fraccionarias solo se admiten para órdenes de mercado. Como una OTO M5 usa una entrada stop, `submit_oto` rechaza cantidades fraccionarias localmente con `oto_no_soporta_fracciones`; registra ambas órdenes locales y no envía una request.

Los errores de envío de entrada se persisten como intent rechazado. Un stop OTO rechazado no deja una entrada remota independiente pendiente.

Validación: `NUMBA_CACHE_DIR=/tmp/claude-1000/nd03/numba .venv/bin/python -m pytest tests/test_pipeline_m5.py tests/trading -q`.
