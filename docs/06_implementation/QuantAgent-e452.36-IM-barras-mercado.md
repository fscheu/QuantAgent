# EM-10 · Almacén de barras diarias de mercado (QuantAgent-e452.36)

**Qué hay:** tabla `market_bars` (símbolo, sesión, OHLCV, fuente), `market_bar_status` (última sesión NYSE cerrada, `stale`, sesiones faltantes), universo de 31 símbolos (los 29 ETFs de `marketConfig.universe` más XLC y XLRE) y el comando `quantagent mercado barras actualizar [--simbolos] [--desde] [--base] [--snapshot] [--dry-run]`. La API no descarga nada en vivo.

**Prueba con claves reales (2026-11-04, sobre una copia de `maestro.db`, no sobre la real):** `migrate-paper-db --apply` en la copia creó `market_bars`; `actualizar --desde 2019-01-01` bajó 31 símbolos x 1.971 barras (61.101 filas, 2019-01-02 a 2026-11-03, fuente alpaca) sin huecos; XLC y XLRE tienen la misma historia que el resto; la segunda pasada no cambió el conteo. La sesión del día aparecía como `stale=True, missing_sessions=1` porque la corrida fue después del cierre y Alpaca todavía no publicaba la barra del día: la lógica de `stale` funciona, y el timer de EM-11 tiene que correr con margen después del cierre.

**Orden para producción** (el esquema de `maestro.db` lo coordina el hilo del registro; no correr dos cambios de esquema a la vez):
1. Backup de `~/web-ops/maestro.db`.
2. `quantagent api migrate-paper-db --database-url sqlite:///<ruta>` sin `--apply` (informe); con OK de Fede, con `--apply`. Aparte de `market_bars`, la base hoy no tiene otras tablas del esquema ni `backtest_runs.hypothesis_id`; las completa la misma migración.
3. `quantagent mercado barras actualizar --desde 2019-01-01` (claves de Alpaca en el entorno), primero con `--dry-run`.
4. Timer diario después del cierre (EM-11).
