# QuantAgent-e4jc.8.3 — snapshot sin destino persistente autorizado

Entrega de bloqueo; el ticket permanece `in_progress`. No se seleccionaron
ETFs, no se congeló un snapshot y no hay manifiesto nuevo que aprobar.

## Intento y evidencia

- Ticket elegible en L20: D-2b cerrado; D-2a ya integrado en `0c7a0330`.
- El 2026-10-22, `download_daily("FXA", "2007-01-03")` recibió 4983 filas
  de Yahoo en memoria. El recorte hasta 2018 terminó en 2018-12-31.
  Esto prueba conectividad, no constituye selección ni snapshot congelado.
- `QUANTAGENT_SNAPSHOT_DIR=/home/baiscf/repos/QuantAgent/local-data` está
  fuera de las raíces de escritura autorizadas y del worktree permitido.
  No se intentó escribir allí ni modificar snapshots existentes.
- `snapshot verify --name etf-ampliados-1d-2018-20261022` salió 1:
  falta el manifiesto en esa carpeta. No se presenta como verificación verde.
- Guardar Parquet ignorados en este worktree no resuelve la entrega:
  `scripts/loop/run_nightly.sh`, función `cleanup`, elimina el worktree
  al salir; `try.sh` busca los datos en el checkout principal.

## Decisión que destraba

Relanzar este mismo ticket con escritura autorizada en la carpeta persistente
`QUANTAGENT_SNAPSHOT_DIR` y excepción explícita a «nada fuera de este worktree»
para crear únicamente el nuevo snapshot. El wrapper ya contempla
`next-local-data` → `--add-dir` en `scripts/loop/run_nightly.sh`.
No se modificó configuración externa ni se activó ese mecanismo acá.

Después se podrá completar la selección con fuentes históricas, descargar,
versionar el manifiesto y verificar los Parquet reales desde `try.sh`.
No se implementaron proxies ni se calcularon resultados de estrategias.
