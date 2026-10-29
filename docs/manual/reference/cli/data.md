# `quantagent data` command reference

```text
Usage: quantagent data [OPTIONS] COMMAND [ARGS]...

  Manage frozen market data.

Options:
  --help  Show this message and exit.

Commands:
  macro     Manage ALFRED macro snapshots (all vintages) in Parquet.
  snapshot  Manage Parquet snapshots outside the repository...
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
