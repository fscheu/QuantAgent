# Glossary

Each entry gives a plain definition, a quick example, and its source.

## Research and workflow

### Hypothesis — A strategy idea recorded with its rules; for example, a YAML file says to buy when a moving average crosses upward. [Roadmap §0](../02_planning/2026-10-28_roadmap_post_m5.md#glosario-corto)
### Funnel — The screening stages for a hypothesis; for example, a strategy advances from stage 1 to stage 2. [Roadmap §0](../02_planning/2026-10-28_roadmap_post_m5.md#glosario-corto)
### Sharpe deflated — The Sharpe criterion adjusted for the number of attempts N, so more attempts raise the bar; for example, 100 tested ideas require stronger evidence than one. [Roadmap §0](../02_planning/2026-10-28_roadmap_post_m5.md#glosario-corto)
### Sharpe ratio — Annualized excess return per unit of daily return volatility; the Performance report convention and sample threshold are defined in [Performance](performance.md#conventions).
### Reserve — Untouched data held for a final check; for example, each admitted hypothesis opens 2023+ once. [Reservation protocol](../04_decisions/QuantAgent-e4jc.14-DC-protocolo-reserva-2023.md)
### Reserva 2023+ with cap 3 — The final data period is limited to three prelisted hypotheses; for example, F1 enters only after passing stage 2. [R01, decision 5](../04_decisions/QuantAgent-R01-DC-decisiones-investigacion-edge.md#decisiones)
### C0 — The stage 3 verdict criterion, with measurement corrections; it is not a cycle budget and has no C1 or C2 path; for example, the reserve is assessed with C0 plus M1. [R01, decisions 1 and 5](../04_decisions/QuantAgent-R01-DC-decisiones-investigacion-edge.md#decisiones)
### M1 — Research reporting Sharpe in excess of the short rate using DGS3MO excess returns; the Performance screen uses its separately documented backtest convention. [R01, decision 2](../04_decisions/QuantAgent-R01-DC-decisiones-investigacion-edge.md#decisiones), [Performance](performance.md#conventions)
### N (attempts) — The count of strategy tests used in the deflated Sharpe correction; for example, opening the reserve adds an attempt to N even when it fails. [Reservation protocol](../04_decisions/QuantAgent-e4jc.14-DC-protocolo-reserva-2023.md)
### Stages 0–3 — Admission, quick vectorbt screen, candle-by-candle engine check, then stage 3 verdict; for example, a hypothesis must clear stage 2 before stage 3. [Roadmap §0](../02_planning/2026-10-28_roadmap_post_m5.md#glosario-corto)
### Candidate — A strategy that survives the funnel and matures through paper checks; for example, a strategy that passes the stage 3 gate enters local shadow. [Roadmap §1](../02_planning/2026-10-28_roadmap_post_m5.md#1-m-ciclos-generar-filtrar-y-madurar-candidatas-de-forma-continua)
### 1/N — The equal-weight benchmark across N choices; for example, 1/N across 10 strategies assigns 10% to each. [R01, decision 3](../04_decisions/QuantAgent-R01-DC-decisiones-investigacion-edge.md#decisiones)
### Shadow (sombra local) — Simulated trading run by our code without a broker; for example, compare its fills with backtest fills. [Roadmap §0](../02_planning/2026-10-28_roadmap_post_m5.md#glosario-corto)
### Paper — Trading with simulated money at Alpaca; for example, submit a simulated order to check execution. [Roadmap §0](../02_planning/2026-10-28_roadmap_post_m5.md#glosario-corto)
### Fraccionable — An asset Alpaca allows buying in fractional shares; for example, buy 0.3 shares when the asset is fractionable. [Roadmap §0](../02_planning/2026-10-28_roadmap_post_m5.md#glosario-corto)
### Cuenta cash — A brokerage account without borrowing or short selling; for example, a purchase must fit available cash. [Roadmap §2](../02_planning/2026-10-28_roadmap_post_m5.md#2-m-cuenta-que-la-candidata-funcione-en-una-cuenta-de-usd-20003000)
### Clean epoch (época limpia, `invalida_bug`) — The start date from which attempts count toward N. Earlier attempts affected by a bug are kept in `intentos.csv` but marked `invalida_bug` and left out of the deflated Sharpe; for example, rows invalidated on 2026-10-28 stay on file but no longer raise N. [Attempts epoch plan](../loop/particiones/QuantAgent-1gw0.1.md)
### PDT — Pattern day trader rule for certain small margin accounts; for example, four same-day round trips in five business days trigger it. [Roadmap §0](../02_planning/2026-10-28_roadmap_post_m5.md#glosario-corto)
### Snapshot — A frozen data download; for example, two runs use the same ETF snapshot. [Roadmap §4](../02_planning/2026-10-28_roadmap_post_m5.md#4-m-datos-dnde-guardar-los-datos-de-mercado-y-fundamentales)
### Equity snapshot — A daily record of a paper account's equity, cash, and invested value; for example, the API uses it to show portfolio equity history. [API operations](api.md), [reconcile.py](../../quantagent/cli/reconcile.py)
### Master accounts database — The database that registers paper portfolios and points to their own databases; for example, the API reads its account registry. [API operations](api.md), [accounts.py](../../apps/api/accounts.py)
### Manifest — A file listing data hashes; for example, matching hashes show both runs used the same files. [Roadmap §4](../02_planning/2026-10-28_roadmap_post_m5.md#4-m-datos-dnde-guardar-los-datos-de-mercado-y-fundamentales)
### Train, test, and reserve — Data periods for fitting, judging, and a final untouched check; for example, train on 2007–2018 and test on 2019–2022. [Roadmap §0](../02_planning/2026-10-28_roadmap_post_m5.md#glosario-corto)
### Buy and hold — Buy at the start and sell at the end; for example, compare a strategy with SPY over the same dates. [Roadmap §2](../02_planning/2026-10-28_roadmap_post_m5.md#2-m-cuenta-que-la-candidata-funcione-en-una-cuenta-de-usd-20003000)
### Intrabar — Within a candle's high and low; for example, check whether a stop was hit before the close. [Roadmap §0](../02_planning/2026-10-28_roadmap_post_m5.md#glosario-corto)
### Control (backtesting.py) — An independent backtest implementation; for example, matching trades provide a cross-check. [Edge research §3](../02_planning/2026-10-20_investigacion_edge_y_generacion.md)
### Golden test — A saved expected result; for example, fail when a trade or metric changes. [Edge research §3](../02_planning/2026-10-20_investigacion_edge_y_generacion.md)
### Walk-forward — Fit on one period and measure on the next; for example, roll the training window forward yearly. [Edge research §3](../02_planning/2026-10-20_investigacion_edge_y_generacion.md)
