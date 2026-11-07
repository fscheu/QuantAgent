# Funnel, hypothesis detail, and market

The Embudo, hypothesis detail, cycle receipt, and Mercado screens are read-only views. The web app sends GET requests to the API; it never changes the registry, downloads market data, submits orders, or operates a broker account. Treat every displayed value as context from the latest data available, not as a live price or trading instruction.

## Embudo

Choose **Todos**, a monthly cycle (`YYYY-MM`), or **Sin ciclo**. The period control offers 30 days, 90 days, or all history. It filters hypotheses with a stage event in that window and applies to the funnel and hypothesis list only. Selecting a specific cycle disables and ignores the period because that cycle is its own time window. The current-cycle card always shows all its hypotheses. **Sin ciclo** shows imported history as stage events and attempts, not as hypotheses; imported attempt labels are shown as stored.

The review counter counts consecutive closed cycles with no promoted hypothesis. Promotion means passing E3; six such cycles reach the review rule. With no closed cycles, the counter says there is no data to count. It does not change with the cycle or period filter. The feed can also show `cycle_closed` and `review_rule_reached` events.

Cycle milestones show proposal, approval/funnel start, and closure. Proposals are on day one; approval has no fixed date and starts the funnel. A cycle closes when its hypotheses finish or at month end. Paper is outside the cycle. The cycle card summarizes entered, active, cut, held, and promoted hypotheses; choose a past cycle to inspect its milestones and receipt.

The funnel chart shows E0–E3 and, for each stage, how many hypotheses reached it and how many passed, were cut, ended in ERROR, or triggered ALARMA. ERROR and ALARMA are shown as **detained**, separate from cuts. A small sample may replace the chart with a notice. Later phases such as shadow, paper, and candidate are shown separately. Drop reasons identify the stage and cause; where a numeric measure exists, the example shows its value and comparator/threshold. The hypothesis list can be filtered by state or stage. A card may show its cycle, current stage/verdict, idea, discard reason and measure, and latest metric when present.

## Hypothesis detail

Open a hypothesis from its card. The timeline shows the result and date for each reached stage, its reason, and any measured value versus threshold; stages not reached are marked as such. Later phases are listed separately. The attempts list includes date, asset or portfolio, snapshot hash, Sharpe, trades, PnL, and stage when available. An invalid attempt is dimmed and labeled invalid, with its reason; it remains visible for audit. **Used reserve 2023+** marks use of the reserved period.

Backtests linked to the hypothesis appear in the linked-backtests section. **“No linked backtests yet”** means this detail has no linked backtest records; it does not claim that no separate backtest exists. A paper account is shown only when one is linked. Paper is not part of a monthly cycle.

## Cycle receipt

An open cycle has a **partial** receipt: its counts, stage results, drops, attempts, and snapshots are a current view and its hash is not fixed yet. After closure, the receipt is immutable and displays its hash; snapshot rows show their hash; the as-of date shows a dash (—) because the receipt does not store it yet. The hash identifies the closed receipt content. Pending stages in a partial receipt have not started; an empty snapshots list means no data has been used yet. Older v1 receipts may be degraded: fields unavailable in that version can be absent, so do not interpret a missing field as a zero or as evidence that an event did not happen.

## Mercado

Mercado is a close-of-session view built from stored daily bars. The header gives the session date and update time and warns when sessions are missing. The SPY regime card reports one of four combinations of trend (price above/below its 200-session average) and volatility (20-session volatility versus its one-year median), when available, plus its history strip. Missing inputs are labeled rather than inferred.

The **Motor M5** card shows the latest weekly motor state, calculation time, next calculation, and sector eligibility. Risk-off means no new purchases; neutral means elevated macro stress while SPY is not risk-off; risk-on is the remaining state. This is the paper portfolios' motor rule, not a recommendation. If the motor has not calculated, the card says so; it does not imply that the market is risk-off.

The sector table or heatmap shows the 11 ranked sectors and SPY, EEM, and EWZ as reference rows. It includes returns by period, the M5 score/rank, enabled or blocked status and reason, and holdings in paper accounts when present. Macro currently includes the T10Y2Y yield spread with observation/publication dates and recession periods on its chart; the other indicator cards are marked **Coming soon**. The ETF section contains the 29 ETF universe grouped by asset class. Select up to five to compare historical series. Open an ETF for daily candles, choose a 3-, 6-, or 12-month range and linear or log scale. The upcoming-events section lists the next 14 days and links back to the complete event list on Home.

## Reading missing or old data

- **No data** means the required stored observations or history are absent or insufficient. For example, SPY needs 200 sessions for its moving average, and an ETF chart needs at least two candles. Check that the daily-bar update job has run and inspect the relevant data source; a missing value is not zero.
- **Stale data** means expected sessions or a macro release are missing. Read the missing-session list or expected release date, then check the daily market-bar updater/timer or the macro source before relying on the figures. For macro, the stale notice appears after two business days past the expected release (or the cadence-based rule when there is no release calendar). The page dims stale values and identifies the section.
- **The motor has not calculated** means no weekly M5 state is available. Check the M5 state output configured by `QUANTAGENT_M5_DIR` (default `local-data`) and the process that produces it. This is distinct from a calculated risk-off state.
- If an API section reports an error, use its retry action; other market sections can still load. If registry-backed Embudo data cannot load, check that `QUANTAGENT_REGISTRO_URL` points to the reachable read-only registry database.

The market-bar updater is `quantagent mercado barras actualizar`; its scheduled job maintains stored daily bars. The API does not fetch live market data. See [market bars implementation](../06_implementation/QuantAgent-e452.36-IM-barras-mercado.md) and [registry operation](../06_implementation/QuantAgent-e452.27-IM-registro-postgres.md).
