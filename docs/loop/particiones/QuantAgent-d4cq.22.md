# QuantAgent-d4cq.22: partición de E07

Estimación previa: 160–200 líneas agregadas+borradas entre reutilizar el callback
de `filtrar`, encadenar etapas y probar los tres caminos del comando.
El ticket exige como máximo 100; esta entrega no modifica código.

| Hijo | Objetivo | Aceptación binaria | Estimación | Revisión |
|---|---|---|---|---|
| QuantAgent-d4cq.22.1 | Exponer resultados de etapas 0 y 1 reutilizando el flujo actual | Tests de `filtrar` verdes y resultados reutilizables iguales al CSV, incluido rechazo real | 80–100 | leer |
| QuantAgent-d4cq.22.2 | Agregar `correr` hasta etapa 1 y registrar estado | Rechazo real: una fila; corte real del filtro: dos; ningún compare; salida y exit codes correctos | 80–100 | leer |
| QuantAgent-d4cq.22.3 | Encadenar etapa 2 con parámetros elegidos | Control real sobre snapshot sintético, tres filas y resumen/exit code según veredicto | 70–100 | leer |

Orden: .22.1 → .22.2 → .22.3. La cola de §4 reemplaza la fila del padre.
El padre queda in_progress hasta cerrar los tres hijos; E08 sigue dependiendo
del padre, para impedir que arranque con el comando incompleto.
Cada hijo debe respetar su propio límite de 100 líneas, incluyendo tests.
E07b entrega un comando temporalmente limitado a etapas 0 y 1; E07c completa E07.
No se agregan umbrales, candado ni etapa 3.

Tests dirigidos de cada hijo: `python -m pytest -q tests/test_cli_hipotesis.py`.
Los tests de etapa 2 usan `pytest.importorskip("backtesting")`, según el comentario
del PM; los de etapas 0 y 1 no se saltean.
