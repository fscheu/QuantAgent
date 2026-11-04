# QuantAgent-nd03 · IM · Protección de entradas M5

Los precios de entrada y protección se cuantizan al tick de Alpaca antes de enviar la orden: centavos desde USD 1 y diezmilésimas por debajo, con redondeo `ROUND_HALF_UP`. Si el stop de protección falla o queda rechazado, el pipeline intenta cancelar la entrada, marca su estado y registra el intent como `rechazado_riesgo` con motivo `stop_rechazado`; el ciclo sigue con los demás símbolos.
