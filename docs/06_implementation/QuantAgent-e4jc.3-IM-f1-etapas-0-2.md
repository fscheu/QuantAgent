# F1, etapas 0 a 2

Corrida única del 2026-10-26 sobre el snapshot `etf-1d-2026-10-amplio`, desde `local-data`.

## Resultado

- Etapa 0: PASA. La admisión aceptó `rebalanceo-fin-de-mes`; informó las advertencias `desvio_de_la_cita` y `overlay_solo_largo`, que no bloquean.
- Etapa 1: CORTA por menos de 30 trades en la cartera.
- La corrida se detuvo en etapa 1. No se evaluó la etapa 2 ni se abrió la reserva; no corresponde pedir a Fede el OK de reserva.
- El intento de etapa 1 quedó en `local-data/intentos.csv` y suma a N; no se creó salida por celda en `local-data/intentos/f1` porque la etapa 2 no corrió.

Comando reproducido:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/mpl-cache QUANTAGENT_SNAPSHOT_DIR=/home/azureuser/repos/projects/QuantAgent/local-data python -m quantagent.cli backtest hipotesis correr hipotesis/rebalanceo-fin-de-mes.yaml --snapshot etf-1d-2026-10-amplio --carpeta local-data/intentos/f1 --hasta-etapa 2
```

La propuesta fue aprobada por Fede con `hipotesis propuestas aprobar rebalanceo-fin-de-mes --quien fede` antes de la corrida. El archivo vive ahora en `hipotesis/`.
