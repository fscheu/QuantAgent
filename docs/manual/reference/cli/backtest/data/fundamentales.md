# `quantagent backtest data fundamentales` command reference

```text
Usage: quantagent backtest data fundamentales [OPTIONS] COMMAND [ARGS]...

  Manage SEC EDGAR (XBRL) fundamentals snapshots.

Options:
  --help  Show this message and exit.

Commands:
  cobertura  Report per-metric coverage since 2010 and tickers without a...
  create     Download companyfacts from SEC EDGAR and create the snapshot.
  refrescar  Refresh current index fundamentals and retain the previous...
  verify     Recompute the file hashes against the manifest.
```

## `quantagent backtest data fundamentales cobertura`

```text
Usage: quantagent backtest data fundamentales cobertura [OPTIONS]

  Report per-metric coverage since 2010 and tickers without a CIK.

Options:
  --name TEXT         Snapshot name to report.  [required]
  --composicion TEXT  Composition snapshot used to define index member-days.
  --help              Show this message and exit.
```

Options and defaults:

- `--name` — default: `None`
- `--composicion` — default: `'sp500-2026-10'`

## `quantagent backtest data fundamentales create`

```text
Usage: quantagent backtest data fundamentales create [OPTIONS]

  Download companyfacts from SEC EDGAR and create the snapshot.

Options:
  --name TEXT        Snapshot name, for example fundamentales-sp500-2026-10.
                     [required]
  --tickers-de TEXT  Composition snapshot providing the 2007-to-date index
                     members.
  --tickers TEXT     Comma-separated ticker symbols to use instead of the
                     composition snapshot.
  --help             Show this message and exit.
```

Options and defaults:

- `--name` — default: `None`
- `--tickers-de` — default: `'sp500-2026-10'`
- `--tickers` — default: `None`

## `quantagent backtest data fundamentales refrescar`

```text
Usage: quantagent backtest data fundamentales refrescar [OPTIONS]

  Refresh current index fundamentals and retain the previous valid snapshot on
  failure.

Options:
  --composicion TEXT  Composition snapshot providing the latest index members.
                      [default: sp500-2026-10]
  --help              Show this message and exit.

  Example: quantagent data fundamentales refrescar --composicion sp500-2026-10
```

Options and defaults:

- `--composicion` — default: `'sp500-2026-10'`

## `quantagent backtest data fundamentales verify`

```text
Usage: quantagent backtest data fundamentales verify [OPTIONS] [NAME_ARG]

  Recompute the file hashes against the manifest.

Options:
  --name TEXT  Snapshot name to verify (alternative to the argument).
  --help       Show this message and exit.
```

Options and defaults:

- `--name` — default: `None`
