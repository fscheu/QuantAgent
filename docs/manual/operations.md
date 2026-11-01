# Operations

Commands below inspect the configured paper account or local data. They do not start a trading cycle. `trading snapshot` queries Alpaca without changing the broker, then writes a snapshot to the local database. `trading pre-apertura` can submit stop orders and cancel stale entry orders at the broker.

## Reconcile paper positions

Run `python -m quantagent.cli trading reconciliar` with the paper database and Alpaca paper credentials configured. Matching positions print `Sin divergencias: posiciones locales y Alpaca coinciden.` Differences print one line per discrepancy. Investigate differences before relying on the digest.

## Before market open

Run `python -m quantagent.cli trading pre-apertura` against the paper account before the session opens. It does three things. First, for each long position with a fractional quantity and no open stop at the broker, it submits a day stop at the position's stored stop price, because Alpaca only accepts whole-share stops as good-till-cancelled. Second, it cancels buy stop entry orders older than three NYSE sessions. Third, it checks that every position is covered by an open sell stop for at least its full quantity. The result lists `reenviados` (stops resubmitted), `entradas_vencidas` (cancelled entries), `sin_stop` (positions still unprotected) and `avisos`. An `Alerta PDT` warning appears when the account's day-trade count reaches 2; see PDT in the [glossary](glossary.md). Treat any symbol in `sin_stop` as unprotected and investigate before the open.

For the M5 bucket, run `python -m quantagent.cli trading pre-apertura --bucket m5` to use its dedicated Alpaca paper credentials and `local-data/m5.db`. Pass `--db <path>` to select another SQLite file. Any `sin_stop` symbol prints `ALARMA` and exits with status 1.

Run one M5 cycle with `python -m quantagent.cli trading m5-correr --modo semanal` after the weekly close, or `python -m quantagent.cli trading m5-correr --modo diario` for the daily cycle. `--as-of <ISO timestamp>` sets the cycle time (default: current UTC); `--db <path>` selects the M5 SQLite file (default: `local-data/m5.db`). Weekly output reports the regime; daily output reports intent and order counts. An exception exits with status 1. See the [M5 paper operations runbook](../06_implementation/QuantAgent-fnpr.6.5-IM-runbook-puesta-en-marcha-m5.md) for activation gates.

## Daily digest

Run `python -m quantagent.cli trading digest --date YYYY-MM-DD` for the configured paper portfolio; output starts with `Resumen de paper trading — YYYY-MM-DD`, then `## Alertas y avisos`, `## Señal del scheduler`, and `## Actividad y P&L`. Position and reconciliation sections appear when data is available. For a dedicated portfolio, use `python -m quantagent.cli trading digest --date YYYY-MM-DD --portfolio <portfolio>`; it also reports account valuation, executions, and pending orders. Exit status is 1 when a dividend query fails, a reconciliation differs, or an unexplained rejected order exists.

For the M5 bucket, run `python -m quantagent.cli trading digest --bucket m5 --date YYYY-MM-DD`; it reads `local-data/m5.db` by default and reports intents, current stops, and heat. It also lists weekly candidates by sector and rank, sectors falling back to their ETF with discard reasons, price/fundamentals discard counts (up to three examples each), and the current fundamentals snapshot date and age. Quantities use up to four decimals. No intents prints `Sin actividad de intents hoy.`

M5 reads the weekly regime, candidates, discards and shadow CSVs next to the selected database. The shadow comparison reports the latest weekly interval and compounded returns since the first shadow decision for H1, H3 and an equal allocation to the 14 M5 ETFs (1/N), using Alpaca closes through the last closed NYSE session. Friday decisions apply to the following interval; a decision at the final close is excluded. Returns use decision closes as references, so they include the next opening gap and are a close-to-close monitoring approximation, without costs. Missing endpoint prices produce `sin dato`, without carrying prices forward or reallocating missing legs. Zero weights remain cash. The pipeline uses its existing equity calculation (initial capital plus unrealized P&L) and existing equity snapshots at the reference dates; missing snapshots produce `sin dato`. No snapshots are written by the digest. Fundamentals come from `QUANTAGENT_SNAPSHOT_DIR`, or the database directory; the weekly CSVs do not record which snapshot was used, so the digest identifies the current verified snapshot, with a monthly fallback dated by month.

## Equity snapshots

Run `python -m quantagent.cli trading snapshot --db <file> --portfolio <portfolio>` to query the Alpaca paper account and upsert one row for the current New York date in `equity_snapshots`. Each row stores `equity`, `cash`, and `invested` (`equity - cash`). This is read-only at the broker; rerunning it on the same day overwrites that day's row. The command requires a deployed tag that includes it.

Schedule it after each digest; a snapshot failure should not fail the digest.

Before the API reads each paper database, run `python -m quantagent.cli api migrate-paper-db --database-url <url>` and review the dry run, then run it again with `--apply`. Back up each database first.

The read-only web app replaces the Streamlit QA interface.

## Local data and snapshots

`local-data/` holds ignored local databases and snapshot files. Set `QUANTAGENT_SNAPSHOT_DIR` to the directory holding snapshots; do not commit its data. Verify an existing snapshot with `python -m quantagent.cli data snapshot verify <name>`; success prints `OK <symbols> símbolos, <rows> filas`. Check its sessions and OHLC quality with `python -m quantagent.cli data snapshot check --name <name>`; expect one quality summary per symbol and exit status 1 only for missing sessions.

To create a snapshot, run `python -m quantagent.cli data snapshot create --name <name> --symbols SPY --start YYYY-MM-DD --end YYYY-MM-DD`. It downloads from Yahoo and prints `Snapshot <name> creado: <symbols> símbolos, <rows> filas`. Use a completed session as the end date; snapshots are immutable.

Paper trading setup is described in the [paper operations runbook](../06_implementation/QuantAgent-av1p.5-IM-runbook-paper-ops.md). This runbook does not activate services or run operational procedures.

## Small-account backtest

`quantagent backtest cuenta-chica` shows how a strategy or portfolio behaves with the real account (default USD 1,000, 2,500 and 5,000) instead of USD 100,000. It buys whole shares only, keeps 5 % of a portfolio in cash, charges spread, commission and a fixed entry cost, and withholds 30 % of implied dividends. Each capital gets one card: Sharpe, return, costs paid, symbols that never fit one share, largest position, and the difference against SPY buy and hold with the same costs.

- One strategy on one symbol: `quantagent backtest cuenta-chica --strategy rsi --snapshot <snapshot> --symbol SPY`.
- A portfolio of several legs: `quantagent backtest cuenta-chica --hipotesis hipotesis/<file>.yaml --snapshot <snapshot> --costos etf --to 2018-12-31`. Add `--macro-snapshot <snapshot>` when a leg uses the yield-curve slope. The portfolio path rejects `--to` from 2019-01-01 so it never opens the test period or the reserve.

Read the "positions at zero" line first: a symbol listed there could not buy one share at that capital, so the card describes a mostly-cash account, not the strategy. A large portfolio can take several minutes. The first report is in `docs/06_implementation/QuantAgent-yvo0.5-IM-cuenta-chica-m4.md`; the option list is in the [CLI reference](reference/cli/backtest.md).
