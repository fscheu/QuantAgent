# QuantAgent-kkj.9 — PL — Strategy Selector UI

**Beads issue:** QuantAgent-kkj.9  
**Depends on:** QuantAgent-kkj.8 (closed)  
**Phase:** Implementation planning

---

## Overview

Four changes across four files, delivered in sequence. All changes are UI-only or CLI-arg additions; no DB schema changes, no new strategy logic.

---

## Step 1 — `apps/paper_trading.py` — Add `--strategy` / `--strategy-params` CLI args

**Why first:** The Streamlit subprocess call must pass these args. Implementing the CLI before the UI avoids a broken intermediate state.

### Changes

1. Add import: `from quantagent.strategy.registry import build_strategy`
2. In `_parse_args()`, add two new arguments:
   ```python
   parser.add_argument("--strategy", type=str, default="LLMAgentStrategy",
       help="Strategy name from registry (default: LLMAgentStrategy)")
   parser.add_argument("--strategy-params", type=str, default="{}",
       help="JSON-encoded dict of strategy constructor params")
   ```
3. In `_apply_overrides()`, parse `--strategy-params` JSON but do NOT push into `SchedulerSettings` (scheduler settings don't carry strategy). Instead, return the parsed values alongside the config, or use a wrapper namespace.

4. Refactor `_build_scheduler` to accept `strategy_name: str` and `strategy_params: dict` as arguments:
   ```python
   def _build_scheduler(config, strategy_name="LLMAgentStrategy", strategy_params=None):
       ...
       strategy = build_strategy(strategy_name, **(strategy_params or {}))
       scheduler = TradingScheduler(..., strategy=strategy)
   ```
5. In `main()`, extract `args.strategy` and `json.loads(args.strategy_params)`, pass to `_build_scheduler`.

**Validation:** `python apps/paper_trading.py --strategy RSIMeanReversionStrategy --strategy-params '{"rsi_period": 10}' --run-once --enable` should start without error (data fetch may fail in test env but arg parsing must succeed).

---

## Step 2 — `apps/streamlit/views/backtesting.py` — Strategy selector + dynamic params

**Pattern:** Strategy selector and dynamic param widgets are placed OUTSIDE `st.form("create_backtest")` so that selecting a strategy triggers a Streamlit rerun and immediately renders the correct param widgets. Values are stored in Streamlit session state via `key=` arguments and read on form submission.

### Changes

1. Add import at top:
   ```python
   from quantagent.strategy.registry import get_strategy_registry
   ```

2. In `render()`, after session state initialization and before the `portfolio_options` collection, add:
   ```python
   strategy_registry = get_strategy_registry()
   strategy_names = list(strategy_registry.keys())
   _default_bt = (st.session_state.get("default_strategy") or {}).get("backtest")
   _bt_idx = strategy_names.index(_default_bt) if _default_bt in strategy_names else 0
   st.session_state.setdefault("default_strategy", {"paper": None, "backtest": None})
   ```

3. Before the `with st.form("create_backtest"):` block, render the strategy UI section:
   ```python
   st.markdown("**Estrategia**")
   selected_strategy = st.selectbox(
       "Estrategia", strategy_names, index=_bt_idx, key="bt_strategy_key"
   )
   strategy_meta = strategy_registry[selected_strategy]
   if strategy_meta["type"] == "llm":
       st.warning("⚠️ Esta estrategia requiere LLM (costo de tokens).")
   # Dynamic params
   for param_name, param_meta in strategy_meta["params"].items():
       st.number_input(
           f"{param_name} — {param_meta['description']}",
           value=param_meta["default"],
           key=f"bt_param_{param_name}",
       )
   ```

4. Inside `st.form`, hide model_preset when strategy is deterministic:
   ```python
   _show_llm = st.session_state.get("bt_strategy_key", strategy_names[0])
   _show_llm = strategy_registry.get(_show_llm, {}).get("type") == "llm"
   if _show_llm:
       model_preset = st.selectbox("Model preset", model_presets, ...)
   else:
       model_preset = None
   ```

5. In the submit handler, build strategy params dict and extend snapshot:
   ```python
   selected_strat = st.session_state.get("bt_strategy_key", strategy_names[0])
   strat_params = {
       k: st.session_state.get(f"bt_param_{k}", v["default"])
       for k, v in strategy_registry[selected_strat]["params"].items()
   }
   snapshot = {
       ...,  # existing keys
       "strategy": selected_strat,
       "strategy_params": strat_params,
   }
   ```

---

## Step 3 — `apps/streamlit/views/paper_trading.py` — Strategy selector in start form

### Changes

1. Add import:
   ```python
   from quantagent.strategy.registry import get_strategy_registry
   ```

2. In `_render_scheduler_controls`, before the `st.expander(...)` block, build registry references:
   ```python
   strategy_registry = get_strategy_registry()
   strategy_names = list(strategy_registry.keys())
   _default_pt = (st.session_state.get("default_strategy") or {}).get("paper")
   _pt_idx = strategy_names.index(_default_pt) if _default_pt in strategy_names else 0
   st.session_state.setdefault("default_strategy", {"paper": None, "backtest": None})
   ```

3. Inside the expander, after the assets / mode / interval widgets, add:
   ```python
   selected_strategy = st.selectbox(
       "Estrategia", strategy_names, index=_pt_idx, key="sc_strategy_key"
   )
   strat_meta = strategy_registry[selected_strategy]
   for param_name, param_meta in strat_meta["params"].items():
       st.number_input(
           f"{param_name} — {param_meta['description']}",
           value=param_meta["default"],
           key=f"sc_param_{param_name}",
       )
   if strat_meta["type"] == "llm":
       st.info("Estrategia LLM — modelo LLM requerido.")
       # (model preset field or info about LLM config here)
   ```

4. Update `_launch_subprocess` signature to accept `strategy` and `strategy_params`:
   ```python
   def _launch_subprocess(assets_str, mode, interval_hours, environment, strategy, strategy_params):
       import json
       cmd = [...existing args..., "--strategy", strategy,
              "--strategy-params", json.dumps(strategy_params)]
   ```

5. In the Start button handler, collect strategy and params before launching:
   ```python
   if st.button("▶ Start", ...):
       sel_strat = st.session_state.get("sc_strategy_key", strategy_names[0])
       strat_params = {
           k: st.session_state.get(f"sc_param_{k}", v["default"])
           for k, v in strategy_registry[sel_strat]["params"].items()
       }
       st.session_state["sc_active_strategy"] = sel_strat
       _launch_subprocess(assets_input, mode, interval_hours, environment, sel_strat, strat_params)
   ```

6. In the status display area (after `_render_status_card`), show active strategy:
   ```python
   active_strat = st.session_state.get("sc_active_strategy")
   if active_strat:
       st.caption(f"Running strategy: **{active_strat}**")
   ```

---

## Step 4 — `apps/streamlit/views/configuration.py` — Strategy Defaults section

### Changes

1. Add import:
   ```python
   from quantagent.strategy.registry import get_strategy_names
   ```

2. In `render()`, add session state init:
   ```python
   st.session_state.setdefault("default_strategy", {"paper": None, "backtest": None})
   ```

3. In `colR`, below the "Model presets" section (or above it — designer preference), add:
   ```python
   st.markdown("**Strategy Defaults**")
   strategy_names = get_strategy_names()
   for env_key in ("paper", "backtest"):
       strat_opts = ["(none)"] + strategy_names
       current_strat = st.session_state.default_strategy.get(env_key) or "(none)"
       chosen_strat = st.selectbox(
           f"{env_key.title()} default strategy",
           strat_opts,
           index=strat_opts.index(current_strat) if current_strat in strat_opts else 0,
           key=f"default_strat_{env_key}",
       )
       if st.button(f"Set {env_key} strategy default", key=f"btn_strat_default_{env_key}"):
           st.session_state.default_strategy[env_key] = (
               None if chosen_strat == "(none)" else chosen_strat
           )
           st.success(f"Strategy default for {env_key} set to {st.session_state.default_strategy[env_key]}")
   ```

---

## Step 5 — Tests

File: `tests/streamlit/` (new or existing)

Required test coverage (see AC document for exact cases):
1. `test_backtesting_strategy_selector` — verify selector renders, LLM warning shown for LLMAgentStrategy, params rendered for RSI.
2. `test_backtesting_snapshot_contains_strategy` — verify `config_snapshot` dict contains `strategy` and `strategy_params` keys after form submit.
3. `test_paper_trading_subprocess_args` — verify `_launch_subprocess` builds cmd with `--strategy` and `--strategy-params`.
4. `test_paper_trading_cli_strategy_arg` — verify `apps/paper_trading.py` parses `--strategy RSIMeanReversionStrategy` without error.
5. `test_configuration_strategy_defaults` — verify session state `default_strategy` is set when "Set default" is clicked.

---

## Risks & Dependencies

| Risk | Mitigation |
|---|---|
| Streamlit session state key collisions between views | Use view-prefixed keys (`bt_`, `sc_`) consistently |
| LLMAgentStrategy import triggers LangGraph/LLM provider init at module level | Registry already imports at module load; verify no side-effects in test env |
| `_launch_subprocess` cmd list construction — JSON with spaces | Always use `json.dumps()` (compact, no spaces by default) |
| `default_strategy` key missing in views that load before Configuration | Each view calls `setdefault("default_strategy", ...)` defensively |

---

## Commit Plan

Single commit on feature branch:

```
feat(QuantAgent-kkj.9): add strategy selector to backtesting, paper trading, and configuration views

- apps/paper_trading.py: add --strategy / --strategy-params CLI args
- apps/streamlit/views/backtesting.py: strategy selector + dynamic params outside form
- apps/streamlit/views/paper_trading.py: strategy selector + subprocess arg forwarding
- apps/streamlit/views/configuration.py: Strategy Defaults section
- tests: 5 new test cases covering selector render, snapshot, subprocess args, CLI parsing, session state
```
