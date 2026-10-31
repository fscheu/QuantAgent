# `quantagent data` command reference

```text
Usage: quantagent data [OPTIONS] COMMAND [ARGS]...

  Manage frozen market data.

Options:
  --help  Show this message and exit.

Commands:
  acciones       Manage S&P 500 stock snapshots for 2007-2018 using Yahoo...
  consenso       Manage daily earnings and revenue estimates.
  fundamentales  Manage SEC EDGAR (XBRL) fundamentals snapshots.
  macro          Manage ALFRED macro snapshots (all vintages) in Parquet.
  snapshot       Manage Parquet snapshots outside the repository...
```

## `quantagent data acciones`

```text
Usage: quantagent data acciones [OPTIONS] COMMAND [ARGS]...

  Manage S&P 500 stock snapshots for 2007-2018 using Yahoo or Quandl WIKI.

Options:
  --help  Show this message and exit.

Commands:
  cobertura  Report price coverage for 2007-2018 index member-days in the...
  create     Create a snapshot using Yahoo when index membership is...
```

## `quantagent data consenso`

```text
Usage: quantagent data consenso [OPTIONS] COMMAND [ARGS]...

  Manage daily earnings and revenue estimates.

Options:
  --help  Show this message and exit.

Commands:
  foto  Save a daily EPS consensus snapshot in Parquet format.
```

## `quantagent data fundamentales`

```text
Usage: quantagent data fundamentales [OPTIONS] COMMAND [ARGS]...

  Manage SEC EDGAR (XBRL) fundamentals snapshots.

Options:
  --help  Show this message and exit.

Commands:
  cobertura  Report per-metric coverage since 2010 and tickers without a...
  create     Download companyfacts from SEC EDGAR and create the snapshot.
  refrescar  Refresh current index fundamentals and retain the previous...
  verify     Recompute the file hashes against the manifest.
```

## `quantagent data macro`

```text
Usage: quantagent data macro [OPTIONS] COMMAND [ARGS]...

  Manage ALFRED macro snapshots (all vintages) in Parquet.

Options:
  --help  Show this message and exit.

Commands:
  create  Download all ALFRED series vintages and freeze the snapshot...
  verify  Recalculate hashes and compare them with the manifest.
```

## `quantagent data snapshot`

```text
Usage: quantagent data snapshot [OPTIONS] COMMAND [ARGS]...

  Manage Parquet snapshots outside the repository ($QUANTAGENT_SNAPSHOT_DIR).

Options:
  --help  Show this message and exit.

Commands:
  check   Check data quality against the NYSE calendar.
  create  Download Yahoo symbols (split-adjusted, not dividend-adjusted)...
  verify  Recalculate hashes and compare them with the manifest.
```
