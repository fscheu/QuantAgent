# Performance metrics

The Performance screen is a read-only report. The backend calculates every value; the web client displays it. Its metric set and provisional sample thresholds follow [the Performance screen design](../03_design/web-design-system/RENDIMIENTO.md#43-métricas-clave-3--metricgroup-13-filas).

## Conventions

The report uses simple daily returns derived from equity, a 252-trading-day annualization, and the backtest engine's default annual risk-free rate of 2%, exposed as a parameter; benchmark returns use SPY with dividends, and monthly metrics use complete calendar months only.

This is deliberately different from `sharpes_en_exceso()`, which uses point-in-time DGS3MO. That function is a separate research diagnostic for decision M1; the performance report follows the backtest's default convention so its Sharpe agrees with the engine unless the report parameter is explicitly changed.

Let `r_t = equity_t / equity_(t-1) - 1`, `R = Π(1 + r_t) - 1`, and `N` be the number of daily return observations in the selected period. Let `rf` be the annual risk-free parameter and `rf_d = rf / 252`. For benchmark-relative metrics, `b_t` is the corresponding SPY daily return and `R_SPY` is its cumulative return over the same dates.

## Metrics

Thresholds are provisional backend parameters. `short sample` marks the point below which the value may be shown with a warning; a dash means there is no separate short-sample threshold.

| Group | Metric | Definition and formula | Minimum to calculate | Short sample below |
|---|---|---|---:|---:|
| Return | Cumulative return | Compounded change in equity: `R = Π(1 + r_t) - 1`. For portfolios, returns are time-weighted so registered deposits and withdrawals are excluded. | 21 trading days | — |
| Return | CAGR | Annualized compounded return: `(1 + R)^(252 / N) - 1`. | 252 trading days | — |
| Return | Positive months | Share of complete calendar months with a positive compounded return: `count(month_return > 0) / count(complete_months)`. | 3 complete months | 12 complete months |
| Return | Best month | Highest compounded return among complete calendar months, reported with its month. | 2 complete months | 12 complete months |
| Return | Worst month | Lowest compounded return among complete calendar months, reported with its month. | 2 complete months | 12 complete months |
| Risk | Volatility | Sample standard deviation of daily returns, annualized: `s(r) × √252`. | 63 trading days | 252 trading days |
| Risk | Sharpe | Annualized mean daily excess return divided by daily return sample standard deviation: `(mean(r_t - rf_d) / s(r)) × √252`. | 63 trading days | 252 trading days |
| Risk | Sortino | Annualized mean daily excess return divided by downside deviation, where downside deviation is `√mean(min(r_t - rf_d, 0)^2)`: `mean(r_t - rf_d) / downside_deviation × √252`. | 63 trading days | 252 trading days |
| Risk | Maximum drawdown | Largest peak-to-trough equity decline: `min(equity_t / running_peak_t - 1)`. The report also gives peak, trough, recovery, and duration. | 21 trading days | — |
| Risk | Calmar | CAGR divided by the absolute maximum drawdown: `CAGR / |max_drawdown|`. | 252 trading days | — |
| Versus SPY | Alpha | Annualized intercept of the daily excess-return regression `r_t - rf_d = alpha_d + beta × (b_t - rf_d) + error`: `252 × alpha_d`. | 252 trading days | — |
| Versus SPY | Beta | Sensitivity of daily returns to SPY: `cov(r, b) / var(b)`. | 63 trading days | 252 trading days |
| Versus SPY | Relative return | Strategy cumulative growth relative to SPY: `(1 + R) / (1 + R_SPY) - 1`. | 21 trading days | — |

“Trading days” means daily observations available for both the subject and SPY over the common comparison period. A month is complete only when the selected data covers the full calendar month. At fewer than 12 rebalance events, the report also flags a low rebalance count; this threshold is provisional and applies in addition to the metric-specific day or month thresholds.

## Statuses and display

The backend assigns each metric `ok`, `short_sample`, or `not_computable` and supplies the reason. A non-computable metric is shown as a dash with its reason, never as a fabricated value. A short-sample metric may display its value with a short-sample label. The backend returns the active thresholds so the UI can describe them accurately.

Portfolio returns are time-weighted between daily equity snapshots. Register deposits and withdrawals with `quantagent api accounts flows register`; withdrawals are recorded as negative amounts. Reports expose registered and unreconciled flows. An unreconciled flow sets `status: returns_not_computable` with `not_computable_reason: unregistered_flows`. A portfolio database without the `cash_flows` table has no flow registry and its return is not calculable.
