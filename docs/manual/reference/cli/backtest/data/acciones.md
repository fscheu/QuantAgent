# `quantagent backtest data acciones` command reference

```text
Usage: quantagent backtest data acciones [OPTIONS] COMMAND [ARGS]...

  Manage S&P 500 stock snapshots for 2007-2018 using Yahoo or Quandl WIKI.

Options:
  --help  Show this message and exit.

Commands:
  cobertura  Report price coverage for 2007-2018 index member-days in the...
  create     Create a snapshot using Yahoo when index membership is...
```

## `quantagent backtest data acciones cobertura`

```text
Usage: quantagent backtest data acciones cobertura [OPTIONS]

  Report price coverage for 2007-2018 index member-days in the snapshot
  directory.

Options:
  --name TEXT  Snapshot name to report.  [required]
  --help       Show this message and exit.
```

Options and defaults:

- `--name` — default: `None`

## `quantagent backtest data acciones create`

```text
Usage: quantagent backtest data acciones create [OPTIONS]

  Create a snapshot using Yahoo when index membership is covered, otherwise
  WIKI_PRICES.csv.

Options:
  --name TEXT      Snapshot name, for example acciones-sp500-2026-10.
                   [required]
  --wiki-csv FILE  Path to the Quandl WIKI_PRICES.csv file.  [default:
                   /tmp/wiki/WIKI_PRICES.csv]
  --help           Show this message and exit.
```

Options and defaults:

- `--name` — default: `None`
- `--wiki-csv` — default: `'/tmp/wiki/WIKI_PRICES.csv'`
