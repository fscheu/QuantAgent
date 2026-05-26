# QuantAgent-kkj.9 — RQ — Strategy Selector UI

**Beads issue:** QuantAgent-kkj.9  
**Depends on:** QuantAgent-kkj.8 (strategy registry — closed/merged)  
**Type:** Feature  
**Scope:** UI only — three Streamlit views

---

## Context

Four trading strategies are implemented in `quantagent/strategy/`:

| Key | Type | Params |
|---|---|---|
| `RSIMeanReversionStrategy` | deterministic | rsi_period, oversold_threshold, overbought_threshold, stop_loss_pct, take_profit_pct, trailing_stop_pct |
| `FiftyTwoWeekHighStrategy` | deterministic | lookback_days, proximity_threshold, trend_ma_period, volume_ma_period, volume_factor, stop_loss_pct, take_profit_pct, trailing_stop_pct |
| `TripleScreenStrategy` | deterministic | weekly_bars, trend_ema_period, stoch_k_period, stoch_d_period, stoch_oversold, stoch_overbought, stop_loss_pct, take_profit_pct, trailing_stop_pct |
| `LLMAgentStrategy` | llm | (none — wraps TradingGraph pipeline) |

A strategy registry (`quantagent/strategy/registry.py`) was delivered by kkj.8 and exposes `get_strategy_registry()`, `get_strategy_names()`, and `build_strategy(name, **kwargs)`. The registry is the authoritative source of truth; the UI must not hardcode strategy names or parameter lists.

No Streamlit view currently allows the operator to select which strategy runs. The backtesting form defaults silently to LLM model preset selection; the paper trading scheduler is hardcoded to `LLMAgentStrategy` in `apps/paper_trading.py`; the configuration view has no strategy defaults section.

---

## Functional Requirements

### FR-1 — Backtesting view strategy selector

`apps/streamlit/views/backtesting.py`

- **FR-1.1** The create-run form must include a strategy selector populated from `get_strategy_names()`.
- **FR-1.2** When a deterministic strategy is selected, the `model_preset` selectbox must be hidden or visually disabled.
- **FR-1.3** When `LLMAgentStrategy` is selected, the `model_preset` selectbox must remain visible and a `st.warning` must appear indicating LLM token cost.
- **FR-1.4** When a strategy with configurable params is selected, one input widget per param must render below the selector: `st.number_input` for `int`/`float` params, pre-filled with the registry default.
- **FR-1.5** The selected strategy name and its parameter values must be saved in `config_snapshot` under keys `strategy` and `strategy_params`.
- **FR-1.6** The selector must pre-select the backtest default strategy from session state key `default_strategy["backtest"]` when set.

### FR-2 — Paper Trading start form strategy selector

`apps/streamlit/views/paper_trading.py`

- **FR-2.1** The `_render_scheduler_controls` expander must include a strategy selector and dynamic param widgets above the Start button.
- **FR-2.2** The model-preset fields must only display if the selected strategy type is `"llm"`.
- **FR-2.3** On Start, the strategy name and params must be forwarded to the subprocess via new CLI args `--strategy` and `--strategy-params`.
- **FR-2.4** The UI must show the active strategy name (stored in session state when started) in the scheduler status area.
- **FR-2.5** The selector must pre-select the paper default strategy from session state key `default_strategy["paper"]` when set.

### FR-3 — Configuration view strategy defaults

`apps/streamlit/views/configuration.py`

- **FR-3.1** A "Strategy Defaults" section must appear in the configuration right column alongside the existing "Defaults per environment" portfolio selectors.
- **FR-3.2** The operator must be able to select a default strategy for `paper` and `backtest` environments independently.
- **FR-3.3** Selections are persisted to `st.session_state.default_strategy` with keys `"paper"` and `"backtest"`.
- **FR-3.4** A "Set default" button per environment confirms the selection.

### FR-4 — Backend subprocess args

`apps/paper_trading.py`

- **FR-4.1** The CLI must accept a `--strategy` arg (strategy name string, default: `LLMAgentStrategy`).
- **FR-4.2** The CLI must accept a `--strategy-params` arg (JSON-encoded dict string, default: empty `{}`).
- **FR-4.3** `_build_scheduler` must call `build_strategy(name, **params)` with the parsed args and pass the resulting instance to `TradingScheduler(strategy=...)`.

---

## Out of Scope

- Backend execution wiring for backtesting runs (separate ticket).
- Creating new strategies or modifying existing strategy logic.
- DB persistence for strategy defaults (session state is sufficient for this ticket).
- Redesigning the general layout of any view beyond adding the selector section.
- Range sliders for params (use `st.number_input` uniformly; slider is optional future improvement).
