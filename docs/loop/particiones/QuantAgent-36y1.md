# V32: partición de walkforward

Estimación previa: 360 líneas agregadas+borradas, incluyendo tests y código
movido. Supera el objetivo de 100; esta entrega no implementa código.
Contrato funcional: `bd show QuantAgent-36y1`. Cola: `PLAN-CONTINUACION.md` §4.

| Hijo | Entrega | Estimación | Revisión |
|---|---|---:|---|
| QuantAgent-36y1.1 | Ventanas consecutivas y selección por Sharpe con desempate estable | 80 | leer |
| QuantAgent-36y1.2 | Extraer escritura append de sweep, conservando su comportamiento | 90 | probar |
| QuantAgent-36y1.3 | Comando aislado, candado y CSV por ventana | 100 | probar |
| QuantAgent-36y1.4 | Integración sintética con motor real y tabla VM de cuatro ventanas | 90 | probar |

Dependencias: `.4 → .3 → .2 → .1`. Cada hijo contiene en Beads objetivo,
archivos permitidos, criterio binario y comando de tests dirigidos.
El padre queda in_progress hasta completar los cuatro hijos.

Cobertura del padre: selección sin mirar prueba e integración real en `.4`;
ventanas sin solapamiento en `.1` y `.4`; rango insuficiente en `.1` y `.3`;
tabla rsi/SPY 2007–2018 con grilla [7,14,21], ajuste 4/prueba 2 en `.4`.
Registro solo de ajustes: extracción en `.2`, uso en `.3`, prueba en `.4`.
El candado y las opciones del CLI se cubren en `.3`; registrar el comando
requiere tocar `quantagent/cli/backtest.py` solo en su registro, como V31a.2.

Tests a correr en cada hijo:
`python -m pytest -q tests/test_backtest_sweep.py tests/test_backtest_walkforward.py`.
También smoke; suite completa y oráculos en CI.
Si un hijo se estima mayor a 100 líneas antes de implementarlo, se parte de nuevo.
