# QuantAgent-lwji.2 — Informe de Sharpe en exceso

S01b requiere reconstruir las curvas, conectar CLI/HTML y demostrar que no
hay escrituras ni fechas nuevas. Estimación conjunta: 140–170 líneas con tests;
supera el máximo de 100 que exige el ticket. No se implementó código.

| Hijo | Objetivo | Aceptación binaria | Revisión | Estimación |
|---|---|---|---|---|
| QuantAgent-lwji.2.1 | Diagnóstico desde equity guardada con curva_cartera, curva_benchmark y S01a | Sharpes coinciden con S01a; tres registros y DSR idénticos; ausencia explícita; motor no llamado y benchmark limitado a fechas guardadas | leer | 85–100 líneas |
| QuantAgent-lwji.2.2 | Mostrar diagnóstico en HTML y conectar CLI/tanda | CLI muestra ambos Sharpes; sin datos marca ausencia; registros y DSR idénticos; tanda pasa carpeta y snapshot | probar | 80–100 líneas |

El segundo hijo depende del primero; ambos incluyen sus tests y deben volver a
partirse si se estiman más de 100 líneas. La cola de L19 reemplaza al padre
por los dos hijos. El padre queda in_progress hasta su cierre.

Archivos autorizados y comandos dirigidos en los tickets hijos. Se mantiene
el corte y el registro a tasa cero. No se llama `_curva_y_v30_cartera`, no se
ejecuta el motor ni se abren ventanas. La integración no queda terminada con
el primer hijo: el segundo debe probar el comando completo.

Probar la partición: `NUMBA_CACHE_DIR=/tmp/quantagent-loop-numba python -m pytest -q tests/test_informe.py tests/test_criterio.py && NUMBA_CACHE_DIR=/tmp/quantagent-loop-numba scripts/loop/smoke.sh`.
