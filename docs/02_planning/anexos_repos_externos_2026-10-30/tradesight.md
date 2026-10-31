**TradeSight is useful as a reference for evidence-driven monitoring and paper-trading operations. I would not adopt its research engine or strategy-promotion logic.** Its strongest ideas are explicit data provenance, broker/local reconciliation, trade explanations, and visible rejection reasons. Several implementation defects undermine its backtest metrics and its advertised qualification process.

This is a static, read-only review of commit `746b964` dated October 8, 2026. I did not execute or import repository code, install dependencies, or run tests. TradeSight paths below are relative to the inspected repository.

The QuantAgent comparison follows the [October 28 post-M5 roadmap][roadmap]. That document clarifies two points beyond the prompt’s shorthand: operational storage is **Postgres**, with SQLite used in tests; **FastAPI + React** is already decided for the replacement web application.

## 1) What it is and maturity

TradeSight is a self-hosted Python application combining technical stock scanning, strategy backtesting and parameter optimization, Alpaca paper trading, and a local Flask dashboard. Its current positioning emphasizes research evidence and paper-only operation.

| Aspect | Finding |
|---|---|
| Public adoption | GitHub displayed **169 stars, 25 forks, and 134 commits** when checked. These indicate interest, not validated trading performance. [Repository](https://github.com/rmbell09-lang/tradesight) |
| Version | `1.2.0`, with “Beta” development status in `pyproject.toml`. |
| Latest local commit | October 8, 2026: optional packaged-download documentation. The checkout is shallow, so local history cannot establish a development cadence. |
| Size | **282 tracked files**, **83 Python files**, approximately **23,837 Python lines including tests**, and **6 MB excluding `.git`**. |
| Repository hygiene | **32 tracked backup files**, including copies of trading, dashboard, and optimization code. |
| Tests | **24 test modules**, with roughly **336 statically identifiable test functions/methods**. These are not executed-test results. |
| CI | The tracked `.github` content contains `FUNDING.yml`; I found no CI workflow in this snapshot. |
| License | README and package metadata declare **MIT**. I found no standalone tracked `LICENSE` or `COPYING` file. |
| Commercial packaging | README advertises an optional $49 source package. Packaging is separate from research maturity. |

**Maturity assessment:** a substantial personal research application with meaningful operational hardening, but inconsistent integration between newer safety claims and older execution paths.

The repository contains reconciliation services, runtime guards, release verification, and failure-drill tooling. Those are positive signs. However, it also contains duplicated implementations, machine-specific deployment metadata, runtime schema migrations, and important contradictions between documentation and code.

Historical audits are useful context, but not current verification. For example, `tradesight_bias_audit.md` describes several backtesting defects; inspection shows that next-open entries and trade-sequence Monte Carlo were subsequently implemented, while other problems remain.

## 2) Features

| Area | Implemented functionality | Material limitation |
|---|---|---|
| Stock scanning | Fixed stock universes, historical bars, volume filters, technical indicators, confluence and opportunity scores. | These are technical rankings, not evidence of predictive edge. |
| Indicators | RSI, MACD, Bollinger Bands, moving averages, ATR, volume measures, and others through TA-Lib and custom calculations. | Research and trading paths do not consistently use identical formulas. |
| Strategy library | Moving-average crossover, RSI mean reversion, VWAP reversion, opening-range breakout, MACD, Bollinger, momentum, and trend variants. | Some names imply more faithful market/session modeling than the implementation provides. |
| Strategy tournaments | Multiple rounds, elimination, surviving strategies, rankings, and SQLite history. | The tournament’s “test” round contributes to selecting the winner. |
| Optimization | Family-specific parameter grids, champion-centered searches, broader rescoring of finalists, cross-symbol evaluation. | Mostly technical parameter search rather than hypothesis generation. |
| Paper execution | Alpaca market orders, fractional quantities, position synchronization, order/fill reconciliation, and trade-update WebSocket reconnection. | Qualification and strategy selection are not consistently isolated from execution. |
| Risk management | Position caps, exposure groups, stop/take-profit management, trailing stops, cooldowns, daily-loss controls, and drawdown suspension. | Defaults and rules are distributed across several modules. |
| Evidence UI | Broker positions/orders/fills, local records, reconciliation status, provenance labels, and “why this trade?” explanations. | Some supposedly read-only views perform schema or state writes. |
| Alerts | Email, webhooks, local safety records, and alert history. | Configuration and successful delivery still require operational proof. |
| Live readiness | Policy-driven readiness checklist and blocker display. | Some checks measure weaker evidence than their labels suggest. |
| Polymarket | Scanner and retained dashboard paths. | README explicitly describes this scanner as unmaintained and stale records as archived. |

Important source locations include `src/scanners/`, `src/indicators/`, `src/strategy_lab/`, `src/trading/`, and [web/dashboard.py][dashboard].

An optional earnings filter is referenced in `paper_trader.py`, but its imported module, `data.earnings_calendar`, is absent from the inspected snapshot. I would therefore not count that protection as an available feature.

## 3) Architecture

### Modules and data flow

| Layer | Main files | Responsibility |
|---|---|---|
| Market access | `src/data/alpaca_client.py` | Custom Alpaca HTTP client, market data, orders, account information, demo generation. |
| Analysis | `src/indicators/`, `src/scanners/` | Indicators, regime classification, scanning, opportunity scoring. |
| Research engine | `src/strategy_lab/backtest.py`, `backtester.py`, `tournament.py`, `ai_engine.py` | Simulation, validation helpers, tournaments, source-code mutation. |
| Research orchestration | `src/automation/strategy_automation.py`, `scripts/overnight_strategy_evolution.py` | Dataset construction, tournaments, parameter search, reports, champion evaluation. |
| Paper operation | `src/trading/paper_trader.py`, `position_manager.py` | Strategy selection, signals, sizing, execution, exits, synchronization. |
| Evidence and controls | `accounting_truth.py`, `trade_evidence.py`, `runtime_risk.py`, `live_readiness.py` | Broker proof, reconciliation, operational entry gates, readiness projections. |
| Presentation | `web/dashboard.py`, `web/templates/unified_dashboard.html` | Flask endpoints, server-rendered HTML, browser polling. |
| Launching | `tradesight/cli.py`, launcher scripts | Resolve installation, launch dashboard, run operational jobs. |

The main flows are:

```text
Alpaca daily bars
    → strategy-family tournament
    → tournament_history.db
    → parameter optimizer using hourly history
    → JSON reports / champion.json / symbol_performance.json

Recent tournament winners + champion/feedback state
    → paper trader
    → Alpaca paper orders
    → local positions and fill records
    → reconciliation and evidence services
    → dashboard
```

A significant mismatch is visible here: **family selection uses daily bars, optimization uses hourly bars, and paper trading adds further execution rules and feedback adjustments.** The complete deployed behavior is not necessarily the strategy evaluated in a particular research report.

### Simple infrastructure, complicated application state

The infrastructure is relatively small: Python processes, Flask, SQLite, JSON files, and scheduled scripts. There is no distributed compute system or elaborate service platform.

Application complexity is much higher:

- `paper_trader.py`: approximately **2,915 lines**.
- Overnight optimizer: approximately **2,209 lines**.
- Main dashboard backend: approximately **1,336 lines**.
- Unified dashboard template: approximately **1,591 lines**.

State is split among multiple SQLite databases, JSON reports, champion files, feedback records, and readiness receipts. `OptimizerRegistry` projects a unified view over these sources; it does not establish a single authoritative lifecycle model.

Other architectural weaknesses:

- **Duplicate dashboards:** `web/dashboard.py` and `src/web/dashboard.py`, plus many backups.
- **Duplicate indicators:** TA-Lib in the scanner, pandas formulas in the backtester, and another RSI implementation in `signals/signal_service.py`.
- **Import side effects:** configuration creates directories and loads credentials; some scripts initialize logs at import.
- **Runtime migrations:** services inspect tables and issue `ALTER TABLE`, including from evidence-view code.
- **Machine coupling:** `ops/canonical_source.json` contains a particular Mac installation path and launch-agent names.

The architectural lesson for QuantAgent is to keep a small deployment while making ownership of state and lifecycle transitions explicit. Low service count alone does not guarantee simplicity.

## 4) Historical and live data sources

### Providers, access, and costs

| Source | Actual repository use | Credentials/cost | Caveats |
|---|---|---|---|
| **Alpaca historical equities** | Primary daily tournament data and hourly optimization data. | Alpaca credentials; code explicitly requests `feed='iex'`. | IEX coverage differs from consolidated market coverage. |
| **Alpaca current quotes** | Current prices for scanning, position updates, and trading decisions. | Alpaca credentials. | Quote normalization and fallback behavior are problematic. |
| **Alpaca paper Trading API** | Orders, positions, fills, account state, and reconciliation. | Paper credentials; paper trading is free. [Official SDK documentation](https://alpaca.markets/sdks/python/) | Broker simulation is a separate evidence category from real-money execution. |
| **Yahoo through yfinance** | Hourly-history fallback and daily VIX retrieval. | No key passed by these functions. | Adjusted history can change; access is not an institutional data guarantee. |
| **Polymarket Gamma API** | Prediction-market scanner. | No authentication in the inspected fetch path. | Unmaintained; not relevant to QuantAgent’s equity roadmap. |
| **Synthetic data** | Demo clients, research examples, and several fallback branches. | None. | Must remain categorically separate from market evidence. |

Alpaca’s current documentation lists **Basic as free**, with IEX real-time equity coverage, 30 WebSocket symbols, and 200 historical requests per minute. Algo Trader Plus is listed at **$99/month** with all-exchange equity coverage and higher limits. TradeSight hardcodes IEX for historical requests, so purchasing a subscription would not automatically change that path. [Alpaca market-data plans](https://docs.alpaca.markets/us/docs/about-market-data-api)

For QuantAgent’s $2,500 pilot, $99/month would be $1,188/year—approximately 47.5% of starting capital. TradeSight provides no evidence that this expense would be justified.

`src/config.py` also declares Polygon, Yahoo, and OpenAI credentials. **Declarations are not integrations:** I found no active Polygon research pipeline or OpenAI strategy-generation client in the inspected implementation.

### Storage and caching

TradeSight stores **operational and derived evidence** more thoroughly than underlying research data:

- SQLite for positions, portfolio state, tournaments, and feedback.
- JSON for optimizer reports, champion parameters, and per-symbol scores.
- A persisted balance cache.
- A dashboard reconciliation cache.
- No implemented immutable OHLCV snapshot store with manifests and hashes.
- No point-in-time fundamental or macroeconomic warehouse.
- No equivalent to QuantAgent’s planned Parquet/DuckDB data layer.

Historical research downloads depend on the current date. Reports preserve some source and bar-count metadata, but not enough underlying data to guarantee reproduction of a previous run.

### Concrete quality issues

**1. Synthetic fallback remains in the shared data client.**  
`get_historical_data()` generates tagged demo data after several error conditions. Newer scanners and optimizer paths reject these tags, which is useful, but the client still returns synthetic prices instead of a typed failure.

More seriously, `get_quote()` returns generated quotes on non-200 responses and exceptions, without equivalent provenance fields on `StockQuote`. Thus the README’s strong fallback claim is not uniformly true. See [src/data/alpaca_client.py][broker].

**2. Quote fields misrepresent market meaning.**  
`get_quote()` puts the ask price in `last` and the ask size in `volume`. Neither is the corresponding last-trade or aggregate-volume measure.

**3. Corporate-action handling is inconsistent.**  
Yahoo fallback uses `auto_adjust=True`; the Alpaca historical request does not explicitly specify adjustment. There is no normalization contract ensuring comparable price bases across providers.

**4. Universes are current, fixed selections.**  
The client’s “S&P 500” universe is a 50-symbol subset. Research uses selected current large caps and ETFs, without historical membership or delisted securities. This is insufficient for M5’s point-in-time stock-selection requirements.

**5. Fundamental and macro pipelines are absent.**  
No implemented Sharadar/EDGAR point-in-time fundamentals, ALFRED vintages, or daily EPS-consensus archive was found.

Yahoo’s own library documentation also distinguishes software availability from rights to the downloaded data and describes yfinance as an unofficial research tool. [yfinance documentation](https://ranaroussi.github.io/yfinance/)

## 5) Hypothesis/strategy generation and testing loop

### How ideas are created

There are two distinct mechanisms.

**The operational optimizer** searches parameters within predefined families: RSI mean reversion, MACD crossover, momentum breakout, and Bollinger bounce. Parameters include entry thresholds, stops, profit targets, holding periods, and filters.

**The “AI strategy engine”** performs random mutations of Python source using regular expressions, then reconstructs functions with `exec()`. It evaluates several mutations and retains improvements on the same training data.

There is **no LLM hypothesis researcher** in that engine. It does not generate an economic rationale, enforce a hypothesis schema, or learn through structural postmortems. The OpenAI configuration entry does not change that. Source: `src/strategy_lab/ai_engine.py`.

For QuantAgent, this is substantially weaker than the roadmap’s YAML hypotheses, admission checks, attempt accounting, and restricted feedback to the generator.

### What validation actually exists

TradeSight does implement several useful mechanisms:

- Chronological train/validation/test slicing in tournament preparation.
- First-70% training and last-30% evaluation in the optimizer.
- Cross-symbol evaluation.
- Window-based stability checks.
- Multiple-testing penalty heuristics.
- Monte Carlo/bootstrap helpers.
- Persisted rejection reasons.
- Evidence-triggered full searches, separated from daily monitoring.

Full-search triggers include new bars across several symbols, credible closed paper trades, a regime change, or a 14-day maximum age. These are concrete controls, although new bars across correlated stocks are not independent experiments.

Sources: `src/automation/strategy_automation.py` and [scripts/overnight_strategy_evolution.py][optimizer].

### Why I would not trust the resulting qualification

**A. The backtest equity curve omits invested principal.**

Opening a position deducts its notional and entry fee from cash. Equity is then recorded as:

```text
cash + unrealized P&L
```

The position’s principal is missing.

Ignoring costs, a $500 account buying a $75 position would immediately display approximately $425 equity despite an unchanged market price. Closing the position returns the principal and creates an artificial jump.

This corrupts drawdown and return-series-based Sharpe, both of which influence optimization and qualification. This finding follows directly from `_open_position()` and `run_backtest()` in [src/strategy_lab/backtest.py][engine].

**B. The optimizer’s “walk-forward” check reuses training history.**

`walk_forward_validate()` receives parameters already selected using the first 70% of history. It then tests those fixed parameters on four overlapping windows.

With the default construction, the windows end at approximately 25%, 37.5%, 50%, and 62.5% of the complete history. Their alleged out-of-sample slices therefore lie **inside the original 70% training region**.

There is no per-fold fitting followed by an untouched future test. The separate final-30% evaluation is a real chronological split within one run; this window check is not independent walk-forward evidence.

**C. Holdout reuse is uncontrolled.**

The same moving historical tail is evaluated repeatedly across searches. Per-symbol “OOS” outcomes are also written to `symbol_performance.json` and used to decide which symbols may trade.

That makes those outcomes selection inputs. There is no equivalent to QuantAgent’s reserve-opening ledger, finite holdout budget, or global accounting of repeated research attempts.

The tournament similarly uses its “test” round to rank survivors and choose a winner.

**D. “Deflated Sharpe” is a heuristic proxy.**

The optimizer calculates:

```text
Sharpe improvement − sqrt(log(current-search trials)) / sqrt(trade count)
```

It explicitly calls this a proxy. It is not the roadmap’s statistically specified deflated-Sharpe procedure, and its trial count does not account for the complete history of searches.

It should not replace QuantAgent’s attempt accounting.

**E. Monte Carlo covers narrower questions than the labels imply.**

There are two implementations:

- `backtester.py` bootstraps realized trade returns.
- The optimizer bootstraps aggregate OOS returns across symbols.

The first improves on shuffling OHLCV bars. The second asks whether resampled symbol averages remain positive. Neither proves resilience to new regimes, serial dependence, different execution, or correlated portfolio losses.

**F. Execution and accounting details remain inconsistent.**

In `backtest.py`:

- Entries correctly execute at the next bar’s open.
- Stops and targets still trigger from the **close**, rather than intrabar extremes.
- Stop fills do not model adverse opening gaps adequately.
- Test slices restart indicators without preceding training history.
- The loop starts after 50 bars despite also calculating a 100-bar moving average.
- Strategies receive the full DataFrame; no admission boundary prevents arbitrary future-row access.
- Strategy exceptions are silently skipped.
- Closed-trade P&L omits the separately deducted entry fee.
- Some exit timestamps use wall-clock time or remain unset.
- Cross-symbol summaries average independent account metrics, rather than simulate a shared portfolio.

These are reasons to reject the engine as a reusable dependency.

### Promotion and incubation are not consistently enforced

The README says automatic promotion is disabled. However:

- Optimizer `main()` can call `ChampionTracker.evaluate_challenger()` after its quality gate passes.
- That tracker can promote on first run, after a blended-score comparison, or through a forced-replacement circuit breaker.
- `PaperTrader.get_latest_tournament_winners()` selects recent winners directly.
- When no winners are available, paper trading falls back to a predefined strategy list.

Thus the documented frozen-candidate lifecycle is not enforced end to end.

Readiness counts also need caution. `live_readiness.py` uses recorded verified accounting dates for forward-session evidence; these are not necessarily contiguous trading sessions for one immutable strategy version. Account-level trade counts are likewise weaker than candidate-specific incubation evidence.

**Absent or incomplete:** economically grounded hypothesis generation, robust global multiple-testing accounting, protected final holdouts, independent fold-wise walk-forward optimization, enforced candidate-specific shadow incubation, and a small-account benchmark qualification.

## 6) Reusable community components and libraries

TradeSight’s main reusable dependencies are conventional. Much of its complexity comes from custom code surrounding them.

| Component | TradeSight use | Implication for QuantAgent |
|---|---|---|
| **TA-Lib** | Scanner indicators. | QuantAgent already declares it. Standardize formulas and warm-up behavior instead of maintaining parallel RSI/MACD implementations. [Official documentation](https://ta-lib.github.io/ta-lib-python/) |
| **pandas / NumPy** | Indicators, data manipulation, scoring, bootstrap calculations. | Already available in QuantAgent; sufficient for the initial ETF scale. |
| **yfinance** | Historical fallback and VIX. | Already available. Put it behind QuantAgent’s snapshot/provenance contract rather than copying fallback logic. |
| **Flask** | Dashboard and JSON endpoints. | Useful evidence that a modest UI can serve operational needs. Keep QuantAgent’s decided FastAPI architecture rather than adding another web stack. |
| **websocket-client** | Alpaca trade-update stream and reconnect supervision. | Reuse the reconnect concept and failure scenarios. QuantAgent already has `alpaca-py`; prefer its trading/data interfaces where suitable. [Official SDK](https://alpaca.markets/sdks/python/) |
| **Chart.js** | Charts in `src/web/dashboard.py`. | An available standard chart library, but QuantAgent already plans ECharts and Lightweight Charts. Adding a third system has little value. [Chart.js documentation](https://www.chartjs.org/docs/latest/) |
| **SQLite / JSON** | Operational records and receipts. | Good for isolated local tools. They do not justify reversing QuantAgent’s existing Postgres decision. |
| **pytest** | Regression and guard tests. | Reuse the operational scenarios conceptually: stale accounting, orphan positions, partial fills, reconnects, and release isolation. |

TradeSight **does not provide an established third-party backtesting engine underneath its research workflow**. Its engine, tournament, optimization, scoring, and promotion machinery are custom.

QuantAgent already has more relevant reuse choices: `vectorbt` for screening, `alpaca-py` for brokerage access, SQLAlchemy/Alembic for state, and the `backtesting` development dependency for independent comparisons. Adopting another custom engine would increase the burden of proving equivalence.

## 7) Comparison against QuantAgent’s post-M5 roadmap

### M-ciclos: continuous research and candidate maturation

**Adopt:** separate cheap daily observation from expensive research searches. TradeSight’s persisted “search skipped because…” receipt is a useful interface idea. So are visible candidate/champion distinctions and explicit rejection reasons.

**Retain QuantAgent’s stronger controls:** monthly cycles, five hypotheses, cumulative attempt accounting, structural postmortems, reserve-use tracking, and a six-cycle stopping rule. These directly address problems TradeSight’s repeated parameter search leaves unresolved.

**Avoid in TradeSight:** automatic replacement after losses. A failing champion is a reason to suspend or retire it; it is not evidence that an unproven challenger should trade.

**Avoid in QuantAgent:** building a general tournament platform, adaptive scoring layer, and elaborate family analytics before the monthly process produces useful candidates.

A minimal authoritative record needs:

```text
hypothesis/version + cycle + data snapshot + attempts
+ stage + verdict/reasons + evidence links
+ incubation dates + approval/retirement history
```

The roadmap also contains an unresolved inconsistency: its simplified incubation proposal puts the three-month period in local shadow, while the recorded C3 decision still says three months in Alpaca. Resolve that criterion before implementing counters or readiness screens.

### M-cuenta: proving viability at $2,500

**Adopt:** fractional quantity handling, cash/equity caps, recorded sizing reasons, and broker/local reconciliation.

**Do not mistake TradeSight’s $500 default for small-account validation.** Its independent per-symbol backtests do not establish that a combined portfolio can fund simultaneous positions.

TradeSight does not implement the roadmap’s complete test:

- Actual fractionability and whole-share rounding.
- Shared cash and settlement availability.
- Dividend withholding.
- Funding and account-specific fiscal costs.
- Sensitivity at $1,000/$2,500/$5,000.
- Comparison with SPY under the same capital and cost assumptions.

**Avoid in QuantAgent:** sophisticated market-impact or adaptive-size models before implementing those basic constraints. At this scale, unopenable positions, spreads, turnover, and fixed costs deserve attention first.

### M-web: replacing Streamlit

**Adopt the evidence presentation.** The useful examples are:

- Separate market-data provenance from trading mode and accounting verification.
- Show freshness timestamps and concrete errors.
- Distinguish broker facts from local strategy attribution.
- Provide a “why this trade?” view.
- Display blockers with current value, required value, and supporting evidence.

`src/trading/trade_evidence.py` is especially relevant: it returns recorded signal and risk context and explicitly says when historical explanations were not captured. It does not invent a rationale afterward.

**Keep the roadmap’s first delivery small:** Account and Funnel, read-only, with phone-friendly layouts. Ordinary polling is sufficient initially.

**Avoid in TradeSight:** a large template, large route module, duplicate dashboard implementations, and request-time schema changes.

**Avoid in QuantAgent:** bespoke chart infrastructure, real-time browser streaming, a built-in chat product, or comprehensive Market/Hypothesis screens before the initial views are used. FastAPI + React does not require a large frontend platform.

### M-datos: market and fundamental storage

TradeSight reinforces the need for QuantAgent’s snapshot approach. Its reports preserve results without preserving enough input data to reproduce them.

**Adopt:** explicit source, observation time, availability, and verification fields.

**Retain:**

- Immutable Parquet snapshots and manifests.
- Named dataset versions attached to research.
- Postgres for operational and research-lifecycle state.
- DuckDB when the stock universe makes it useful.
- A catalog describing source, license, units, location, and “known at” time.

**Avoid in QuantAgent:** implementing the roadmap’s entire reference architecture for a live data layer and corporate-action engine immediately. Its simplified monthly full-refresh approach is proportionate for the current ETF scope. TradeSight offers no reason to bring forward a larger ingestion platform.

### M-manual: current operating knowledge

TradeSight has substantial documentation, but its current claims and execution code disagree. More prose alone would not solve that.

**Adopt:** short explanations next to evidence states and readiness blockers.

**Retain QuantAgent’s plan:** English Markdown, generated CLI reference, linked decisions, current criteria, and operational procedures.

**Avoid in QuantAgent:** building a separate documentation application or elaborate synchronization system. Start with the shared Markdown source. Also, the proposed test rejecting Spanish accented characters is not a reliable English-language check; help completeness and generated-reference consistency are more useful automated checks.

### Concrete quick wins

| Priority | Action | Benefit |
|---|---|---|
| 1 | Define separate fields for `source`, `trading_mode`, `verification_status`, `observed_at`, and `data_version`. | Prevents “real market data” from being confused with real execution or verified accounting. |
| 2 | Add a read-only trade-explanation response using persisted decision, sizing, order, and fill evidence. | Makes Account useful without an LLM explanation layer. |
| 3 | Show reconciliation freshness and entry-suspension reasons prominently. | Makes operational failures understandable. |
| 4 | Produce one cycle receipt, including no-search/no-promotion outcomes. | Makes M-ciclos observable before the web exists. |
| 5 | Check that qualification and incubation evidence refers to one frozen candidate version. | Prevents account activity from being mistaken for strategy validation. |
| 6 | Add focused accounting oracles where missing: unchanged-price entry/exit, fees, gaps, and shared cash. | Targets the kinds of defects visible in TradeSight. |
| 7 | Complete the $2,500 feasibility comparison before expanding research subscriptions or optimization scope. | Aligns engineering effort with the roadmap’s actual pilot objective. |

**Adoption decision:** borrow TradeSight’s evidence vocabulary and operational UI patterns. Keep QuantAgent’s research funnel, snapshot discipline, explicit approvals, and small-account gate. The most valuable simplification for both projects is one authoritative candidate lifecycle and fewer adaptive layers between the strategy tested and the strategy operated.

[roadmap]: docs/02_planning/2026-10-28_roadmap_post_m5.md
[dashboard]: ext/tradesight/repo/web/dashboard.py
[broker]: ext/tradesight/repo/src/data/alpaca_client.py:124
[optimizer]: ext/tradesight/repo/scripts/overnight_strategy_evolution.py:958
[engine]: ext/tradesight/repo/src/strategy_lab/backtest.py:86