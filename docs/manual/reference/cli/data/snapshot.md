# `quantagent data snapshot` command reference

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

## `quantagent data snapshot check`

```text
Usage: quantagent data snapshot check [OPTIONS]

  Check data quality against the NYSE calendar. Exit 1 only when sessions are
  missing.

Options:
  --name TEXT  Snapshot name to check.  [required]
  --help       Show this message and exit.
```

Options and defaults:

- `--name` — default: `None`

## `quantagent data snapshot create`

```text
Usage: quantagent data snapshot create [OPTIONS]

  Download Yahoo symbols (split-adjusted, not dividend-adjusted) and freeze
  the snapshot.

Options:
  --name TEXT     Snapshot name, for example etf-1d-2026-10.  [required]
  --symbols TEXT  Comma-separated ticker symbols.  [required]
  --start TEXT    First date, YYYY-MM-DD.  [required]
  --end TEXT      Last included session, YYYY-MM-DD. Must be before today
                  because today's candle is not closed.  [required]
  --help          Show this message and exit.
```

Options and defaults:

- `--name` — default: `None`
- `--symbols` — default: `None`
- `--start` — default: `None`
- `--end` — default: `None`

## `quantagent data snapshot verify`

```text
Usage: quantagent data snapshot verify [OPTIONS] [NAME_ARG]

  Recalculate hashes and compare them with the manifest.

Options:
  --name TEXT  Snapshot name to verify.
  --help       Show this message and exit.
```

Options and defaults:

- `--name` — default: `None`
