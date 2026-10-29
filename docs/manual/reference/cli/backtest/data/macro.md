# `quantagent backtest data macro` command reference

```text
Usage: quantagent backtest data macro [OPTIONS] COMMAND [ARGS]...

  Manage ALFRED macro snapshots (all vintages) in Parquet.

Options:
  --help  Show this message and exit.

Commands:
  create  Download all ALFRED series vintages and freeze the snapshot...
  verify  Recalculate hashes and compare them with the manifest.
```

## `quantagent backtest data macro create`

```text
Usage: quantagent backtest data macro create [OPTIONS]

  Download all ALFRED series vintages and freeze the snapshot (requires
  FRED_API_KEY).

Options:
  --name TEXT    Snapshot name, for example macro-2026-10.  [required]
  --series TEXT  Comma-separated FRED series, for example UNRATE,CPIAUCSL.
                 [required]
  --help         Show this message and exit.
```

Options and defaults:

- `--name` — default: `None`
- `--series` — default: `None`

## `quantagent backtest data macro verify`

```text
Usage: quantagent backtest data macro verify [OPTIONS] [NAME_ARG]

  Recalculate hashes and compare them with the manifest.

Options:
  --name TEXT  Macro snapshot name to verify.
  --help       Show this message and exit.
```

Options and defaults:

- `--name` — default: `None`
