# Normalización transversal en objetivos y sombra (T4)

La consulta de objetivos reutiliza `pesos_objetivo(..., seleccion=...)` y entrega esos pesos al rebalanceador; la sombra nueva conserva los objetivos y el valor de cartera recibidos. Esta entrega verifica helpers existentes; no implementa la integración de señales paper, que sigue en `av1p.4.7/.8`.

El trabajo acumulado en `.1`–`.6` transporta la selección ganadora fuera de muestra de etapa 1, lee los CSV guardados de etapas 2/3, informes y reconstrucción de criterio, y usa la misma selección en curva y benchmark. El benchmark conserva el efectivo cuando los pesos suman cero. Ver [decisión T4](../04_decisions/2026-10-17-DC-piernas-transversales.md#t4-pesos-q1-decidida-por-fede) y [partición 7v45](../loop/particiones/QuantAgent-7v45.md).

La prueba sintética combina una pierna transversal con una pierna por activo: el activo no seleccionado pesa cero, cada pierna conserva 1/S, y la curva de referencia termina en 110000. El mismo objetivo pasa por `rebalancear` y queda registrado en un archivo temporal nuevo con el valor de cartera de 100000.

Este cambio aplica a consultas futuras. La normalización no cambia el SR individual por activo; el agregado puede variar y no se cuantifica aquí su impacto real. No se recalculan ni corrigen resultados históricos.

## Validación

Evidencia sintética en `tests/trading/test_rebalanceador.py`. Comando dirigido ejecutado desde la raíz:

```sh
PYTHONPATH=. NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/quantagent-mpl QUANTAGENT_SNAPSHOT_DIR=/tmp/quantagent-synthetic-only DATABASE_URL=sqlite:////tmp/quantagent-synthetic-only/tests.sqlite /home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m pytest -q tests/trading/test_rebalanceador.py tests/test_cartera.py tests/test_hipotesis_carteras_familias.py tests/test_hipotesis_cartera_nucleo_curva.py
```
