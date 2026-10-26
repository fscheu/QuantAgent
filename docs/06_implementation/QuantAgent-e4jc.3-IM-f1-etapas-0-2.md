# F1, etapas 0 a 2

Corrida única del 2026-10-26 sobre `etf-1d-2026-10-amplio` (2007–2018). El intento de etapa 1 suma a N.

## Veredicto

- Etapa 0: **PASA**, admisión aceptada; avisos no bloqueantes `desvio_de_la_cita` y `overlay_solo_largo`.
- Etapa 1: **PASA**, Sharpe de cartera **1,1885**, con **462 trades** contabilizados.
- Etapa 2: **CORTA**. Para SPY e IEF el filtro registra 0 trades, PnL 0 y Sharpe tasa 0 de 0; corta por `trades < 2`, `pnl <= 0` y `sharpe_tasa_0 < 0.10` en ambas celdas.
- No se puede calcular una correlación informativa con 1/N: ambas curvas de etapa 2 quedan planas por cero trades. La etapa 1 solo aporta Sharpe de cartera y no genera esa comparación.
- La reserva 2023+ no se abrió y no corresponde pedir a Fede el OK para abrirla. No se añadió ninguna entrada al candado; su SHA-256 después de la corrida es `8cc215d8a57c73991d2c36513b0743e09752d4383b0f8304f21f99ae8910f7a5`.

El resumen de etapa 1 quedó en `local-data/intentos.csv`; las celdas y curvas de etapa 2 quedaron en `local-data/intentos/f1b/`. El intento fallido de etapa 1 del 07:16 y el actual del 10:53 están ambos en N.

Comando reproducido:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/mpl-cache QUANTAGENT_SNAPSHOT_DIR=/home/azureuser/repos/projects/QuantAgent/local-data python -m quantagent.cli backtest hipotesis correr hipotesis/rebalanceo-fin-de-mes.yaml --snapshot etf-1d-2026-10-amplio --carpeta local-data/intentos/f1b --hasta-etapa 2
```

Salida: etapa 0 PASA; etapa 1 PASA (Sharpe 1.19); etapa 2 CORTA por las tres condiciones indicadas para SPY e IEF (exit 1 esperado por el veredicto).

## Nota del PM

Entrega recuperada por el PM del log de la corrida de Codex (la rama `loop/QuantAgent-e4jc.3` ya existía por #526 y el wrapper no pudo publicar). Cifras copiadas del IM de Codex, no recalculadas. Salvedad: la etapa 2 muestra 0 trades por activo para una estrategia de pesos objetivo, así que el CORTA de etapa 2 es probablemente otro artefacto de conteo (el control de etapa 2 no ve los rebalanceos). Ticket de seguimiento: QuantAgent-tm5d.
