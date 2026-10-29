# `quantagent data consenso` command reference

```text
Usage: quantagent data consenso [OPTIONS] COMMAND [ARGS]...

  Manage daily earnings and revenue estimates.

Options:
  --help  Show this message and exit.

Commands:
  foto  Save a daily EPS consensus snapshot in Parquet format.
```

## `quantagent data consenso foto`

```text
Usage: quantagent data consenso foto [OPTIONS]

  Save a daily EPS consensus snapshot in Parquet format.

Options:
  --simbolos TEXT       Comma-separated ticker symbols, for example
                        AAPL,MSFT,NVDA.
  --archivo FILE        File with one ticker symbol per line.
  --fecha TEXT          Snapshot date in YYYY-MM-DD format (defaults to
                        today).
  --fuente [yahoo|fmp]  Data source: yahoo or fmp.
  --help                Show this message and exit.
```

Options and defaults:

- `--simbolos` — default: `None`
- `--archivo` — default: `None`
- `--fecha` — default: `None`
- `--fuente` — default: `'yahoo'`
