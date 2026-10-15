# QuantAgent-gfs1.1.2.3 — bloqueo del control en apertura

Se intentó agregar --llenado apertura|cierre en compare_control.py y seleccionar
next-open para el control de una corrida en apertura. El cambio no alcanza el criterio.
Se retiraron todos los cambios de código y tests; esta entrega solo documenta el bloqueo.

Evidencia: tests/test_compare_control.py con el cambio dio 12 passed / 1 failed
(162.26 s). El caso nuevo RSI/spy-2y-1d con costos etf produjo 45 trades en el
motor y 35 en el adaptador; hubo cancelaciones por margen en el adaptador.

El caso exigido RSI/SPY 2007–2018, snapshot etf-1d-2026-10, costos etf,
tamaño 1.0 y apertura produjo 187 trades en ambos, pero el control informó
sin validar: fila 1, columna qty. PnL motor -15619.09; adaptador -15620.14.
No se diagnosticó todavía la causa de la diferencia de cantidad de la primera fila.

Reproducción sin modificar archivos del repo: el parche en memoria fuerza next-open
en ambas llamadas del adaptador, conservando la comparación real de trades.

```bash
timeout 5m python -m quantagent.cli backtest compare --strategies rsi --symbols SPY --snapshot etf-1d-2026-10 --from 2007-01-01 --to 2018-12-31 --costos etf --out /tmp/gfs123.csv --trades-dir /tmp/gfs123-trades && python -c 'import sys,csv; sys.path.insert(0,"scripts"); import compare_control as c; run=c.bt_adapter.run_adapter; c.bt_adapter.run_adapter=lambda *a,**k: run(*a,**dict(k,fill="next-open")); c.main(["--tabla","/tmp/gfs123.csv","--trades-dir","/tmp/gfs123-trades","--snapshot","etf-1d-2026-10","--costos","etf"]); r=next(csv.DictReader(open("/tmp/gfs123.csv"))); print(r["control"]); assert r["control"] == "sin validar: fila 1, columna qty"'
```

Esperado: sin validar: fila 1, columna qty; exit 0 confirma el bloqueo.
Decidir: autorizar un ticket previo de diagnóstico y corrección del adaptador,
o retirar el requisito de coincidencia exacta. No se propone ampliar tolerancias.
El motor, el adaptador y los números de referencia quedan sin cambios.
No se publicó loop-wip: el aislamiento prohíbe git push; el intento se reproduce arriba.

