# Streamlit App Architecture (MVP)

This document describes the structure and responsibilities of the Streamlit UI for QuantAgent, aligned with the MVP requirements and Phase 1 roadmap.

## Goals
- Provide a functional UI to configure profiles, inspect analyses, run backtests, perform sequential “replay sweeps”, and monitor paper trading.
- Keep the code modular, testable, and easy to extend without over‑engineering.

## High-Level Overview

```
apps/
└─ streamlit/
   ├─ app.py                 # Orchestrator (tabs, env select, passes db handle)
   ├─ services/
   │  └─ db.py               # get_db_handle() cached; DbHandle dataclass
   ├─ utils/
   │  └─ ui.py               # DataFrame helpers (e.g., df_from_query)
   └─ views/                 # One module per tab (pure render functions)
      ├─ dashboard.py        # KPIs, recent trades, scheduler status
      ├─ configuration.py    # Strategy profiles JSON editor; model presets
      ├─ analyses.py         # Explore signals (filters, provenance fields)
      ├─ backtesting.py      # Create runs; show status/metrics (placeholders)
      ├─ replay.py           # Replay sweeps (sequential)
      ├─ orders_positions.py # Orders & positions (environment filter)
      └─ logs.py             # Logging placeholder
```

- `app.py` wires the tabs and provides common state (environment select, session defaults). It does not hold heavy logic.
- Each `views/*.py` exposes `render(...)`, imports `streamlit as st`, and uses the shared db handle if needed.
- `services/db.py` centralizes DB connectivity via `get_db_handle()` with `@st.cache_resource` to avoid repeated engine/session creation.
- `utils/ui.py` contains small UI helpers (like safe DataFrame conversion from SQLAlchemy rows).

## Data Flow
- The UI reads from the same PostgreSQL database used by the backend (set via `DATABASE_URL`).
- Missing DB is handled gracefully (warnings, placeholders), so the app remains explorable.
- Session state keys:
  - `ui_profiles`: profiles JSON (portfolio/risk/combined) cached in session for MVP.
  - `model_presets`: model provider/name/temperature default.
  - `backtest_runs`: placeholder list until tables exist (to be replaced by real persistence).

## Environment Filtering
- The UI exposes a global environment selector (backtest, paper). For MVP, some queries do not yet filter by environment until schema fields exist; this will be tightened once migrations add the `environment` field.

## Replay Sweeps (Sequential)
- The Replay tab enqueues N executions (one per selected profile) and runs them sequentially.
- No concurrency is implemented in MVP (simpler; less load). Future: limited parallelism if needed.

## Artifacts Storage & Checkpoints
- Images are saved to disk (path-only by default), and state/analyses store paths instead of base64 to limit DB size.
- Suggested path layout: `data/artifacts/{environment}/{run_or_thread}/{symbol}/{ts}_{pattern|trend}.png`.
- Checkpoints persist via Postgres checkpointer; avoid embedding large payloads inside checkpoint rows.

## Extensibility Guidelines
- Keep `views/*.py` focused on rendering. Business logic and orchestration belong in services or backend modules.
- Prefer small, composable helpers in `utils/`. Avoid duplicating DB logic across views.
- When adding new tabs or features, align with docs/01_requirements/ui_streamlit_mvp_requirements.md and keep interactions Streamlit‑friendly (forms, polling, status).

## Running the App
- Install Streamlit: `pip install streamlit`
- Ensure `DATABASE_URL` is set and PostgreSQL is running (see docs/MIGRATIONS.md)
- Run: `streamlit run apps/streamlit/app.py`

## Next Steps
- Replace session placeholders with real persistence (`StrategyConfig`, `BacktestRun`, `ReplayRun`).
- Wire APScheduler jobs for backtest generation and replay execution.
- Add environment filtering once schema includes `environment` in operational tables.
- Show images from disk in Analyses (path-only by default) and Orders details.

