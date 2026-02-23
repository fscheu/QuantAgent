# Streamlit MVP UI Requirements

This document specifies the functional UI for the MVP in Streamlit. Focus is on functionality, not aesthetics. Target: configure the system, view agent outputs, schedule/run backtests, monitor progress, and inspect paper trading activity with provenance and environment separation.

## Goals & Scope
- Configure Portfolio/Risk profiles and model settings; persist and reuse them (snapshots per run).
- Explore agent analyses/signals with full provenance (order links, checkpoint/thread refs, model metadata).
- Run and monitor backtests; support “replay executions” reusing stored analyses with different Portfolio/Risk profiles.
- Monitor paper trading (scheduler status, orders, positions, metrics) with environment filtering.
- Minimal Streamlit-first interactions (forms, tables, charts, polling) with clear progress feedback.

Assumptions/Constraints (MVP):
- Single-user operation (no auth). Streamlit app runs locally for dev/test.
- Background work is handled by backend code (APScheduler/threads); UI polls DB state (st_autorefresh).
- Large artifacts (charts) referenced by path/id; no large blobs directly in DB.
- Replay sweeps run sequentially (queued), not concurrently (keeps MVP simple).
- Images saved to disk by default (“path-only”); state graph stores paths (not base64) wherever posible.
- Checkpoints persisted via Postgres saver; avoid embedding big payloads in checkpoint rows.
- Equity curves & metrics stored as JSON/rows (compact), not images.
- Logging favors summarized reasoning; full dumps only in debug modes.
- Environments: `backtest`, `paper` (prod is out of MVP scope for UI).
 - Universe of instruments is a fixed list stored in the Portfolio profile; sector exposure rules are out of scope for MVP.

## Global UI Structure
- Global environment filter (default: paper/backtest per tab context).
- Global refresh (auto every 5–10s on data-heavy tabs).
- Sidebar: quick actions (start/stop scheduler, run backtest, switch profile).
- Tabs:
  1) Dashboard
  2) Configuration
  3) Analyses
  4) Backtesting
  5) Replay
  6) Orders & Positions
  7) Logs

## Tab 1: Dashboard
Purpose: High-level monitoring for selected environment.

Components:
- Environment selector (paper/backtest) and timeframe selector.
- KPIs: Portfolio Value, Daily P&L, Win Rate, Open Positions, Open Orders.
- Equity curve (paper) or Backtest run equity (select latest run).
- Recent trades table (10–50 rows, filterable by symbol).
- Scheduler status box (enabled/disabled, next run, last run, errors).

Data sources:
- Positions/Trades/Orders filtered by `environment`.
- Metrics precomputed or computed on the fly from DB.

## Tab 2: Configuration
Purpose: Manage and persist Strategy profiles and model settings.

Features:
- StrategyConfig list (name, kind, version, created_at). Actions: View JSON, Duplicate, Activate for paper/backtest defaults.
- Create/Edit profile (raw JSON editor for MVP). Kinds: portfolio, risk, combined.
- Portfolio profile: manage Universe (multi-select symbols) and sizing/risk parameters (base_position_pct, max_position_pct, max_daily_loss_pct, slippage_pct).
- Model settings: provider, model_name, temperature; save as preset; select default for analyses.
- Display resolved snapshot preview for the currently selected profile (effective config after overrides).

Acceptance:
- Create, list, duplicate profiles. Set defaults per environment.
- Save model presets and set default.
- Backtest/Paper runs snapshot the resolved config.
 - Universe can be managed from the Portfolio profile and used as default assets for backtests.

## Tab 3: Analyses
Purpose: Explore agent outputs with provenance & checkpoints.

Filters:
- Date range, symbol(s), timeframe, environment, model provider/name, min confidence, has order link.

Table columns:
- generated_at, symbol, timeframe, signal, confidence, model_provider/name, agent_version, environment, thread_id, checkpoint_id, order_id.

Row Details (expand):
- Indicator/Pattern/Trend reports (structured → pretty text), charts (pattern/trend images), reasoning (if present), links:
  - “Open Order” (if order_id)
  - “Open Checkpoint” (thread_id/checkpoint_id)
  - Image source is a file path (path-only); optional inline thumbnail when configured

Actions:
- Run single analysis (symbol/timeframe/last N candles) using current model preset (MVP optional).

## Tab 4: Backtesting
Purpose: Create and monitor backtest runs.

Create Backtest Run form:
- Assets (optional; if empty, use Portfolio profile's Universe), timeframe, date range, model preset, strategy profile (for initial execution) and mode:
  - “Generate analyses only” (store signals/analyses + metadata + checkpoints)
  - “Generate + Execute” (run with selected profile immediately)
- Artifacts saving policy: `none | path-only (default) | path+thumbnail`

Runs table:
- id, created_at, status (pending/running/completed/failed), progress %, assets/timeframe/range, model preset, profile snapshot hash, metrics (when done).

Run details:
- Live progress panel (processed/total candles, ETA, current symbol), logs (last N lines), cancel button (if running).

## Tab 5: Replay
Purpose: Reuse stored analyses from a prior backtest run with different portfolio/risk profiles without re-calling LLMs.

Form:
- Select backtest_run_id; select one or multiple profiles (scenario sweep); execute replay.
  - Multiple profiles are executed sequentially (queued), not concurrently (MVP decision).

Output:
- Replay runs table with metrics. Compare two runs side-by-side (win_rate, profit_factor, sharpe, max_dd, total_pnl). Equity curves overlay (basic).

## Tab 6: Orders & Positions
Purpose: Inspect paper trading activity with provenance.

Orders table (environment=paper):
- id, symbol, side, qty, status, created_at, trigger_signal_id, model, profile name, PnL (if closed via trades).

Order details:
- Link to triggering analysis and list of related analyses during lifetime. Fills, average price, commission. Position impact.

Positions table (environment=paper):
- symbol, qty, avg_cost, current_price, unrealized_pnl/pct, side, opened_at.

## Tab 7: Logs
Purpose: Central place to inspect recent events.

Filters: environment, symbol, event type (analysis/order/fill/error), date range.
Columns: ts, level, event_type, symbol, ref_id (order_id/signal_id), message.

## MVP Interaction Model (Streamlit Constraints)
- Long operations executed in background (APScheduler/threads). UI triggers writes to DB to enqueue work, then polls status.
- Use `st.autorefresh` (5–10s) in Backtesting and Dashboard.
- Use lightweight JSON editors (textarea) for profiles; validate on save.
- Charts via matplotlib/plotly; tables via `st.dataframe` with pagination.
 - File storage layout for images (example): `data/artifacts/{environment}/{run_or_thread}/{symbol}/{ts}_{pattern|trend}.png`.
 - Retention: configurable (e.g., keep images for paper N days; in backtests, optional “save none/every Nth candle”).

## Artifacts Storage & Checkpoints (Notes)
- Images on disk with DB path references; default “path-only”.
- Replace any image payloads in agent state with path references before persisting state/checkpoints.
- Checkpoints on Postgres; do not embed large artifacts in checkpointed state.

## Open Questions
1) Where to store large artifacts (charts) for long-term? Local path for MVP; object storage in futuro si hace falta (decision: path-only por defecto).
2) Model presets scope: precedence = per-run override > environment default > global default (decision).
3) Access control (multi-user) out of scope; si surge necesidad, se agrega owner_id (nota).
4) Replay “scenario sweep”: ejecutar secuencialmente; límite configurable de perfiles por replay (decision: secuencial, sin concurrencia).

## Non-Goals (MVP)
- No authentication/multi-user.
- No real broker execution.
- No advanced custom theming.
- No live WebSocket feeds; polling only.

## Minimal Data Assumptions
- Entities: StrategyConfig, BacktestRun, (optionally) ExecutionRun; Signals/Analyses carry environment + checkpoint + model metadata; Orders/Trades/Positions carry environment.

## Acceptance Summary
- Can configure and persist profiles and model presets.
- Can create backtest runs, monitor progress, view metrics.
- Can replay runs using stored analyses with different profiles.
- Can browse analyses with provenance (orders, checkpoints) and artifacts.
- Can monitor paper trading (scheduler, orders, positions) with environment filters.
