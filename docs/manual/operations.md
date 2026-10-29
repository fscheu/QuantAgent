# Operations

Commands below inspect the configured paper account or local data. They do not start a trading cycle or create orders.

## Reconcile paper positions

Run `python -m quantagent.cli trading reconciliar` with the paper database and Alpaca paper credentials configured. Matching positions print `Sin divergencias: posiciones locales y Alpaca coinciden.` Differences print one line per discrepancy. Investigate differences before relying on the digest.

## Daily digest

Run `python -m quantagent.cli trading digest --date YYYY-MM-DD` for the configured paper portfolio; output starts with `Resumen de paper trading — YYYY-MM-DD`, then `## Alertas y avisos`, `## Señal del scheduler`, and `## Actividad y P&L`. Position and reconciliation sections appear when data is available. For a dedicated portfolio, use `python -m quantagent.cli trading digest --date YYYY-MM-DD --portfolio <portfolio>`; it also reports account valuation, executions, and pending orders. Exit status is 1 when a dividend query fails, a reconciliation differs, or an unexplained rejected order exists.

## Local data and snapshots

`local-data/` holds ignored local databases and snapshot files. Set `QUANTAGENT_SNAPSHOT_DIR` to the directory holding snapshots; do not commit its data. Verify an existing snapshot with `python -m quantagent.cli data snapshot verify <name>`; success prints `OK <symbols> símbolos, <rows> filas`. Check its sessions and OHLC quality with `python -m quantagent.cli data snapshot check --name <name>`; expect one quality summary per symbol and exit status 1 only for missing sessions.

To create a snapshot, run `python -m quantagent.cli data snapshot create --name <name> --symbols SPY --start YYYY-MM-DD --end YYYY-MM-DD`. It downloads from Yahoo and prints `Snapshot <name> creado: <symbols> símbolos, <rows> filas`. Use a completed session as the end date; snapshots are immutable.

Paper deployment setup and its approval gates are described in the [paper operations runbook](../06_implementation/QuantAgent-av1p.5-IM-runbook-paper-ops.md). This manual does not activate services or run operational procedures.
