# V32.3: validación previa y comando walkforward

Estimación previa: 150 líneas entre opciones/imports (25), validación de
grilla, candado, snapshot y ventanas (35), ejecución y CSV (40), tests (50).
Supera 100; esta entrega no cambia código. Contrato: `bd show QuantAgent-36y1.3`.
Complementa la partición histórica `QuantAgent-36y1.md`, sin reescribirla.

| Hijo | Objetivo | Estimación | Revisión |
|---|---|---:|---|
| QuantAgent-36y1.3.1 | Preparar grilla, rango, configuración y ventanas antes de ejecutar | 75 | leer |
| QuantAgent-36y1.3.2 | Registrar CLI, ejecutar con aislamiento y escribir CSV | 90 | probar |

La suma incluye el paso de consumo de la preparación en el segundo hijo.
Cadena: `.3.1 → .2`, `.3.2 → .3.1`, `.4 → .3.2`.
Se conserva la dependencia original `.4 → .3`; el padre queda in_progress
hasta que se integren ambos hijos. La cola reemplaza su fila por los hijos.

Aceptación binaria de .3.1: rango insuficiente y fechas protegidas fallan
antes de ejecutar/escribir; default queda dentro del ajuste; el fin de
prueba se recorta a `--to`, incluido 2016-12-30 (comentario del PM).
Aceptación binaria de .3.2: comando registrado sin flags de apertura;
errores salen 1 sin CSV ni intentos; una ventana escribe todos los campos
del padre y la fecha recortada. Solo ajustes registran intentos.
Selección sin redondear, base aislada por corrida y JSON ordenado siguen
siendo requisitos de .3.2. Integración real y tabla VM permanecen en .4.

Tests para ambos: `python -m pytest -q tests/test_backtest_sweep.py tests/test_backtest_walkforward.py`.
También `scripts/loop/smoke.sh`; suite completa y oráculos en CI.
Archivos permitidos y criterios completos están en cada ticket de Beads.
