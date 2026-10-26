# Bloqueo: QuantAgent-g7kg.2

La lista dirigida incluye ocho archivos (93 tests). Los casos de resumen, tanda, informe y alarma pasaron,
pero `tests/test_cli_hipotesis.py` no terminó antes del timeout de 180 segundos; el comando finalizó con
exit 124. Por contrato no se entrega la integración sin que toda la lista quede verde.

El test focalizado del cambio y el test de señal existente pasaron (2 passed). El smoke pasó al dirigir las
cachés de Numba y Matplotlib a `/tmp`. Sin esas variables, el entorno falla al importar `vectorbt` porque
Numba intenta escribir en `site-packages`.

Para destrabar: correr la lista dirigida completa en la VM con `NUMBA_CACHE_DIR=/tmp/numba-cache` y
`MPLCONFIGDIR=/tmp/mpl-cache`; si sigue excediendo 180 segundos, identificar el test lento/frenado antes de
retomar la implementación. El código y su prueba se retiraron de esta entrega.
