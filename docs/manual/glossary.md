# Glossary

Each entry gives a plain definition and a quick example.

## Research and workflow

### Hypothesis — A testable strategy idea; for example, buy when a moving average crosses upward.
### Funnel — The stages that screen a hypothesis; for example, a candidate moves from stage 1 to stage 2.
### Sharpe deflated — A Sharpe score raised against the number of attempts; for example, 100 trials need stronger evidence than one.
### Reserve — Untouched data for a final check; for example, 2023+ is opened once per hypothesis.
### C0 — The initial cycle budget; for example, five hypotheses enter one monthly cycle.
### 1/N — The equal-weight benchmark across N choices; for example, 1/N across 10 strategies assigns 10% to each.
### Shadow — Local simulated trading without a broker; for example, compare its fills with backtest fills.
### Paper — Simulated broker trading; for example, submit an order to an Alpaca paper account.
### PDT — Pattern day trader rule for certain small margin accounts; for example, four same-day round trips in five business days trigger it.
### Snapshot — A frozen data download; for example, two runs use the same ETF snapshot.
### Manifest — A file listing data hashes; for example, matching hashes show both runs used the same files.
### Train, test, and reserve — Data periods for fitting, judging, and a final untouched check; for example, train on 2007–2018 and test on 2019–2022.
### Buy and hold — Buy at the start and sell at the end; for example, compare a strategy with SPY over the same dates.
### Intrabar — Within a candle's high and low; for example, check whether a stop was hit before the close.
### Control (backtesting.py) — An independent backtest implementation; for example, matching trades provide a cross-check.
### Golden test — A saved expected result; for example, fail when a trade or metric changes.
### Walk-forward — Fit on one period and measure on the next; for example, roll the training window forward yearly.

## Bonds and markets

### Sovereign bond — A loan to a government; for example, an Argentine bond pays coupons and principal.
### Yield to maturity (YTM, TIR) — Annualized return if held to maturity; for example, a lower purchase price raises its yield.
### Duration — Approximate price sensitivity to rates; for example, duration 5 implies about a 5% loss for a 1-point rate rise.
### Convexity — The adjustment for changing duration as rates move; for example, gains and losses are not symmetric.
### Yield curve — Yields plotted against maturities; for example, higher long yields make an upward curve.
### VWAP — Volume-weighted average price; for example, 100 shares at $10 and 10 at $20 give a VWAP near $10.91.
### TNA and TEM — Nominal annual and effective monthly rates; for example, report a loan rate in either unit.
### Peso instruments — Local rate products such as CER and BADLAR; for example, CER adjusts with inflation.
### MEP, CCL, official, and swap dollar — Ways to price or obtain dollars; for example, the gap between MEP and official rates is a spread.
### Implied rate — A rate inferred from a market price; for example, a dollar future price implies a peso rate.
### Open interest — Outstanding contracts; for example, it counts futures positions still open.
### Implied forward — A future rate inferred from today's curve; for example, 1-year and 2-year yields imply a rate for year two.
### Cointegration — Prices that tend to return to a stable relationship; for example, trade a pair when its spread is unusually wide.
### Nelson–Siegel / Diebold–Li — Formulas that fit a smooth yield curve; for example, compare a bond yield with the fitted curve.
### Breakeven — The point where two choices have equal value; for example, inflation that makes CER and fixed-rate bonds equivalent.
### Country risk — Extra yield demanded over US Treasuries; for example, compare Argentine dollar bonds with Treasury yields.
### DV01 — Dollar price change for a one-basis-point rate move; for example, DV01 $20 means about $20 lost if rates rise 0.01%.
### Butterfly — A comparison of a middle maturity with the two ends; for example, compare 5-year yield with 2- and 10-year yields.
### Real yield — Yield after accounting for inflation; for example, TIPS report US real yields.
### Basis point — One hundredth of a percentage point; for example, 50 basis points equal 0.50%.
