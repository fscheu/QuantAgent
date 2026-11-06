# `quantagent mercado macro` command reference

```text
Usage: quantagent mercado macro [OPTIONS] COMMAND [ARGS]...

  Manage stored FRED macroeconomic observations.

Options:
  --help  Show this message and exit.

Commands:
  actualizar  Update macroeconomic observations from FRED.
```

## `quantagent mercado macro actualizar`

```text
Usage: quantagent mercado macro actualizar [OPTIONS]

  Update macroeconomic observations from FRED.

Options:
  --series TEXT  Comma-separated FRED series; default: T10Y2Y,USREC.
  --desde TEXT   First observation date in YYYY-MM-DD format; default:
                 2000-01-01 or last stored date minus 7 days.
  --base TEXT    SQLAlchemy database URL; default: QUANTAGENT_DATABASE_URL.
  --dry-run      Report rows without saving them; default: save changes.
  --help         Show this message and exit.

  Example: quantagent mercado macro actualizar --series T10Y2Y,USREC
```

Options and defaults:

- `--series` — default: `'T10Y2Y,USREC'`
- `--desde` — default: `None`
- `--base` — default: `None`
- `--dry-run` — default: `False`
