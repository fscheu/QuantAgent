# QuantAgent-y89w: cancelación de la entrada final

Se intentó agregar al inicio de `StrategyAdapter._close`:

```python
if not self.position:
    for order in list(self.orders):
        order.cancel()
    self.pos = None
    return
```

La reproducción con LONG en cada vela, posición 1.0001, comisión 0.001,
slippage 0.0005 y primeras 32 filas de spy-2y-1d deja de abrir el corto.
Antes: 0 trades cerrados y un corto abierto; después: 0 cerrados, sin
posiciones ni órdenes pendientes. Tests del adaptador: 12 passed / 188.85 s.

Pero `python -m pytest -q tests/test_compare_control.py -k control_cortos_al_cien_por_ciento`
da 2 failed / 1 passed: RSI con costos cripto y etf informa
`sin validar: distinta cantidad de trades (motor=46, port=45)`.
Retirado el guardado, el mismo comando da 3 passed / 41.58 s:
RSI vuelve a `coinciden 46 de 46`. Suite original del adaptador:
11 passed / 184.42 s. Smoke original: exit 0, 9 trades, PnL 542.36.

La condición también cancela una entrada pendiente válida al final;
no distingue una entrada que será rechazada de una que puede llenarse.
Decisión pendiente: definir cómo cerrar esa entrada válida sin mandar
una orden opuesta para una entrada rechazada, conservando los 46 trades,
o autorizar que el control cambie su conteo y ajustar el criterio del ticket.

Se retiraron implementación y test nuevos; entrega solo documental.
No hay rama WIP: este entorno deja Git en solo lectura y el wrapper
publica únicamente la entrega final. `try.sh` exacto falla en FETCH_HEAD
por ese mismo permiso; los comandos internos indicados sí se ejecutaron.
