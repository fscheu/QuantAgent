# QuantAgent-d4cq.71: prueba de truncamiento de vintage (M03)

`check_no_lookahead` corta las velas en T pero no las versiones de ALFRED: una estrategia que lee la última revisión de una serie pasaba la prueba. Ahora hay una variante que también recorta el snapshot macro.

- `data/macro.py::recortar_snapshot_macro(nombre, al, destino, base_dir)`: copia el snapshot a `destino/nombre` descartando las versiones con `realtime_start > al` y acotando `realtime_end` a `al` (extremos inclusivos, como FRED).
- `backtesting/signals.py::check_no_lookahead_macro(signal_fn, velas, cortes, snapshot, base_dir)`: usa `check_no_lookahead(..., truncated_fn=...)`; en cada corte `$QUANTAGENT_SNAPSHOT_DIR` apunta a la copia recortada a T mientras corre `signal_fn`.
- `fabrica/admision.py::admitir`: si la clase declara `requiere_macro = True`, corre la variante macro con el parámetro de valor único `snapshot` de cada combinación; sin `snapshot` o con un snapshot ilegible, rechaza con motivo.

**Contrato para estrategias macro (M04, d4cq.72).** Atributo de clase `requiere_macro = True`; parámetros `snapshot` (y `serie`) de valor único en el YAML; leer solo con `valor_conocido` / `serie_punto_en_el_tiempo`, que resuelven el snapshot por nombre desde `$QUANTAGENT_SNAPSHOT_DIR`. La estrategia no necesita saber nada del recorte.

**Demora de publicación.** El recorte usa el `realtime_start` que registra el snapshot: el valor de un viernes que ALFRED publica el lunes no existe para el viernes ni el fin de semana. Cubierto con un caso sintético (`tests/test_vintage_truncation.py`).

**Corrida en vivo (VM).** Los tests son sintéticos: no usan la key de FRED ni `local-data/`. Una corrida en vivo necesita `QUANTAGENT_SNAPSHOT_DIR=local-data` con `macro-2026-10/` (el parquet no está versionado; solo `tests/fixtures/snapshots/macro-2026-10.json`) y una hipótesis macro real con `snapshot: macro-2026-10` pasando por `admitir`, que llega con M04. La key solo hace falta para bajar un snapshot nuevo (`data macro create`).
