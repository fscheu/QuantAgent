# Partición I01: estado de intentos y N del DSR

El ticket abarca cinco módulos y seis archivos de pruebas dirigidas. Estimación total:
160–230 líneas agregadas+borradas, incluidas las pruebas de compatibilidad y consumidores.
No se escribió código. Se divide en tres entregas secuenciales de ≤100 líneas cada una.

1. **QuantAgent-1gw0.1.1 — CSV (45–70 líneas).** `sweep.py` y
   `test_backtest_sweep.py`: agregar `estado` vacío al final del esquema, aceptar el
   encabezado previo de 15 columnas y conservar filas anteriores al agregar registros.
   Mantener también compatibilidad con archivos anteriores a `sharpe_base`; corregir
   cortes relativos del esquema en fixtures. Aceptación: CSV nuevo con estado vacío,
   lectura y append del previo sin pérdida ni alteración de valores. Revisión: probar.
2. **QuantAgent-1gw0.1.2 — Filas válidas (65–90 líneas).** Depende de `.1`.
   `verdict.py`, `sweep.py`, `comparar.py`, `estado.py` y sus tests: una función
   `filas_validas` excluye `invalida_bug` y snapshot sintético. Se usa en DSR,
   `n_efectivo`, `_n_intentos`, `contar_intentos` y `resumen_tanda`.
   Aceptación: una fila invalidada no cambia DSR ni conteos; sin estado sigue válida.
   Mantener filtro numérico de Sharpes y N por filas hasta `.3`. Revisión: probar.
3. **QuantAgent-1gw0.1.3 — N distinto (50–70 líneas).** Depende de `.2`.
   `verdict.py`, `validacion.py`, `test_verdict.py`, `test_fabrica_validacion.py`:
   N por defecto usa la clave existente de `n_efectivo`; `dsr_crudo` recibe el número
   de filas válidas explícito. Aceptación: dos repeticiones cuentan N=1 en DSR normal
   y N=2 en `dsr_crudo` de etapa 3. Fixtures que representan configuraciones distintas
   deben identificarlas; conservar cálculo independiente. Revisión: probar.

Cada hijo incluye sus comandos en Beads y corre smoke. La lista completa heredada es:
`python -m pytest -q tests/test_verdict.py tests/test_backtest_sweep.py tests/test_hipotesis_comparar.py tests/test_estado.py tests/test_fabrica_validacion.py tests/test_informe.py`.
Si un hijo supera la estimación de 100 líneas, se subdivide antes de implementar.

El padre queda `in_progress` hasta integrar los tres hijos; I02 sigue dependiendo
del padre. No se invalidan corridas reales, no se toca `local-data/`, no se agrega
época limpia ni se cambia la fórmula de `sharpe_deflactado`.
