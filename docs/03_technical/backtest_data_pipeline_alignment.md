# Backtest OHLCV Pipeline Alignment

## Problem Overview
- `Backtest._df_to_kline_data()` emits lowercase keys (`"timestamps"`, `"opens"`, etc.), but downstream tooling expects `"Datetime"`, `"Open"`, `"High"`, `"Low"`, `"Close"`.
- `apps/flask/web_interface.py` duplicates data acquisition via `fetch_yfinance_data_with_datetime`, renaming columns manually and bypassing the cached `DataProvider`.
- Agents and image-generation utilities rely on `static_util.read_and_format_ohlcv()` as the canonical formatter, so the divergent pipelines cause runtime `KeyError: 'Datetime'` when running backtests.

## Remediation Plan
- Add a shared helper in `quantagent/static_util.py` to (a) standardize OHLCV DataFrame columns from the `DataProvider` schema and (b) hand back the agent-ready dict via `read_and_format_ohlcv()`.
- Refactor the backtest engine to call the shared helper and delete `_df_to_kline_data()` to avoid duplicated dict construction logic.
- Update the Flask web interface to source market data through the same `DataProvider` + helper pipeline, removing `fetch_yfinance_data_with_datetime()`.
- Run focused tests (static util, backtest integration) to ensure both CLI and web flows deliver identical, agent-compatible payloads.
