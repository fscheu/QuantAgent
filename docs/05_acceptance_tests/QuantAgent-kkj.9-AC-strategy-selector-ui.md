# QuantAgent-kkj.9 — AC — Strategy Selector UI

**Beads issue:** QuantAgent-kkj.9  
**Format:** Given / When / Then

---

## AC-1 — Backtesting form shows strategy selector with all 4 strategies

**Given** the operator opens the Backtesting view  
**When** the create-run form is displayed  
**Then** a `Estrategia` selectbox is visible with all 4 strategy names from `get_strategy_names()` as options

**Test:** `test_backtesting_strategy_selector_renders_all_strategies`  
```python
# render backtesting view with mocked db
# assert st.selectbox was called with label "Estrategia"
# assert option list == list(get_strategy_registry().keys())
```

---

## AC-2 — Selecting a deterministic strategy hides model_preset

**Given** the backtesting form is open  
**When** the operator selects `RSIMeanReversionStrategy` (or any strategy with `type != "llm"`)  
**Then** the `model_preset` selectbox is not rendered

**Test:** `test_backtesting_deterministic_strategy_hides_model_preset`  
```python
# set session_state["bt_strategy_key"] = "RSIMeanReversionStrategy"
# render view
# assert "Model preset" label NOT in rendered widgets
```

---

## AC-3 — Selecting a deterministic strategy shows its configurable params

**Given** the backtesting form is open and `RSIMeanReversionStrategy` is selected  
**When** the param widgets render  
**Then** a `st.number_input` widget exists for each param key in `STRATEGY_REGISTRY["RSIMeanReversionStrategy"]["params"]` (6 params: rsi_period, oversold_threshold, overbought_threshold, stop_loss_pct, take_profit_pct, trailing_stop_pct), pre-filled with registry defaults

**Test:** `test_backtesting_rsi_params_rendered_with_defaults`  
```python
# set session_state["bt_strategy_key"] = "RSIMeanReversionStrategy"
# render view
# assert st.number_input called for each param key
# assert default values match STRATEGY_REGISTRY["RSIMeanReversionStrategy"]["params"]
```

---

## AC-4 — Selecting LLMAgentStrategy shows LLM cost warning

**Given** the backtesting form is open  
**When** the operator selects `LLMAgentStrategy`  
**Then** a `st.warning` message is displayed containing "LLM" and "token"

**Test:** `test_backtesting_llm_strategy_shows_warning`  
```python
# set session_state["bt_strategy_key"] = "LLMAgentStrategy"
# render view
# assert st.warning called with message containing "LLM"
```

---

## AC-5 — Strategy and params saved in config_snapshot

**Given** the operator selects `RSIMeanReversionStrategy`, sets `rsi_period=10`, and submits the create-run form  
**When** the BacktestRun is created (or session state fallback is used)  
**Then** `config_snapshot["strategy"] == "RSIMeanReversionStrategy"` and `config_snapshot["strategy_params"]["rsi_period"] == 10`

**Test:** `test_backtesting_snapshot_contains_strategy_and_params`  
```python
# set session_state keys for strategy + params
# simulate form submit
# capture BacktestRun.config_snapshot or session_state.backtest_runs[-1]
# assert snapshot["strategy"] == "RSIMeanReversionStrategy"
# assert snapshot["strategy_params"]["rsi_period"] == 10
```

---

## AC-6 — Paper Trading start form includes strategy selector

**Given** the operator opens the Paper Trading view  
**When** the Start Scheduler expander is visible  
**Then** a strategy `st.selectbox` with all 4 strategy names is present inside the expander

**Test:** `test_paper_trading_start_form_has_strategy_selector`  
```python
# render paper_trading view with mocked db
# assert "Estrategia" selectbox present in _render_scheduler_controls output
```

---

## AC-7 — Subprocess receives --strategy and --strategy-params args

**Given** the operator selects `FiftyTwoWeekHighStrategy` with `lookback_days=200` and clicks Start  
**When** `_launch_subprocess` is called  
**Then** the subprocess `cmd` list contains `"--strategy"`, `"FiftyTwoWeekHighStrategy"`, `"--strategy-params"`, and a JSON string containing `lookback_days: 200`

**Test:** `test_launch_subprocess_includes_strategy_args`  
```python
# mock subprocess.Popen, _write_pid
# call _launch_subprocess(..., strategy="FiftyTwoWeekHighStrategy", strategy_params={"lookback_days": 200})
# assert "--strategy" in cmd
# assert "FiftyTwoWeekHighStrategy" in cmd
# assert "--strategy-params" in cmd
# assert json.loads(strategy_params_arg)["lookback_days"] == 200
```

---

## AC-8 — CLI apps/paper_trading.py accepts --strategy arg

**Given** `apps/paper_trading.py` is invoked with `--strategy RSIMeanReversionStrategy --strategy-params '{"rsi_period": 10}' --enable`  
**When** arg parsing runs  
**Then** `args.strategy == "RSIMeanReversionStrategy"` and `json.loads(args.strategy_params)["rsi_period"] == 10`

**Test:** `test_paper_trading_cli_parses_strategy_args`  
```python
# call _parse_args() with sys.argv patched
# assert args.strategy == "RSIMeanReversionStrategy"
# assert json.loads(args.strategy_params) == {"rsi_period": 10}
```

---

## AC-9 — Configuration view lets operator set strategy defaults

**Given** the Configuration view is open  
**When** the operator selects `TripleScreenStrategy` for paper and clicks "Set paper strategy default"  
**Then** `st.session_state.default_strategy["paper"] == "TripleScreenStrategy"`

**Test:** `test_configuration_strategy_default_set_on_button_click`  
```python
# render configuration view
# simulate selectbox to TripleScreenStrategy, click "Set paper strategy default"
# assert st.session_state.default_strategy["paper"] == "TripleScreenStrategy"
```

---

## AC-10 — Strategy selector pre-selects the configured default

**Given** `st.session_state.default_strategy["backtest"] == "TripleScreenStrategy"`  
**When** the Backtesting view is rendered  
**Then** the strategy selector defaults to `TripleScreenStrategy` (index matches its position in `get_strategy_names()`)

**Test:** `test_backtesting_strategy_selector_uses_default_from_session`  
```python
# pre-set session_state["default_strategy"]["backtest"] = "TripleScreenStrategy"
# render backtesting view
# assert selectbox index corresponds to "TripleScreenStrategy"
```

---

## AC-11 — Selector is built from registry, not hardcoded

**Given** a new strategy is added to `STRATEGY_REGISTRY`  
**When** any of the three views renders the strategy selector  
**Then** the new strategy appears in the selectbox without changes to view code

**Test:** `test_selector_reflects_registry_contents`  
```python
# temporarily patch STRATEGY_REGISTRY to add a dummy entry
# render backtesting view
# assert new strategy name appears in selectbox options
```
