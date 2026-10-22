# Bloqueo QuantAgent-cq4u

Intenté recalcular M1–M3 desde las curvas guardadas con `NUMBA_DISABLE_JIT=1`:

```bash
NUMBA_DISABLE_JIT=1 QUANTAGENT_SNAPSHOT_DIR=local-data python scripts/criterio_paper_desde_curvas.py hipotesis/cartera-nucleo-curva.yaml local-data/curvas-l14-curva/etapa2 local-data/curvas-l14-curva/etapa3 etf-1d-2026-10-amplio
```

La corrida termina con exit 1: falta el snapshot `local-data/etf-1d-2026-10-amplio/manifest.json`. También faltan `local-data/curvas-l14-curva/etapa2` y `etapa3`. No hay datos para producir las cuatro filas ni el pedido en seco sin inventar resultados; no registré intentos ni abrí ventanas.

Destraba: correr S09 en la VM que tenga esas curvas, el snapshot y los intentos actuales, y guardar el IM allí.
