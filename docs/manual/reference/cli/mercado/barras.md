# `quantagent mercado barras` command reference

```text
Usage: quantagent mercado barras [OPTIONS] COMMAND [ARGS]...

  Update stored daily market bars.

Options:
  --help  Show this message and exit.

Commands:
  actualizar  Update daily bars from Alpaca or a frozen snapshot.
```

## `quantagent mercado barras actualizar`

```text
Usage: quantagent mercado barras actualizar [OPTIONS]

  Update daily bars from Alpaca or a frozen snapshot.

Options:
  --simbolos TEXT  Comma-separated symbols; default: the 31-symbol universe.
  --desde TEXT     First session date in YYYY-MM-DD format; default: first
                   missing session.
  --base TEXT      SQLAlchemy database URL; default: QUANTAGENT_DATABASE_URL.
  --snapshot TEXT  Snapshot name to backfill instead of downloading from
                   Alpaca.
  --dry-run        Report changes without saving them; default: save changes.
  --help           Show this message and exit.

  Example: quantagent mercado barras actualizar --simbolos SPY,QQQ --desde
  2024-01-01
```

Options and defaults:

- `--simbolos` — default: `None`
- `--desde` — default: `None`
- `--base` — default: `None`
- `--snapshot` — default: `None`
- `--dry-run` — default: `False`
