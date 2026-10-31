# `quantagent api` command reference

```text
Usage: quantagent api [OPTIONS] COMMAND [ARGS]...

  Run the QuantAgent HTTP API.

Options:
  --help  Show this message and exit.

Commands:
  accounts            Manage configured trading accounts.
  migrate-paper-db    Repair a legacy paper database and stamp it at the...
  schema              Export or check the live OpenAPI contract.
  serve               Serve the read-only QuantAgent API.
  sync-calendar       Synchronize Alpaca market holidays and early closes.
  sync-fred-calendar  Synchronize FRED releases and manual calendar events.
```

## `quantagent api accounts`

```text
Usage: quantagent api accounts [OPTIONS] COMMAND [ARGS]...

  Manage configured trading accounts.

Options:
  --help  Show this message and exit.

Commands:
  add      Register an account and its strategy.
  archive  Archive an account so it is hidden from the API.
  list     List active accounts.
  show     Show one account by slug.
```

## `quantagent api migrate-paper-db`

```text
Usage: quantagent api migrate-paper-db [OPTIONS]

  Repair a legacy paper database and stamp it at the current Alembic head.

Options:
  --database-url TEXT  SQLAlchemy database URL to inspect and repair.
                       [required]
  --apply              Apply the schema repair; default: dry-run only.
  --help               Show this message and exit.

  Example: quantagent api migrate-paper-db --database-url
  sqlite:////srv/quantagent/paper.db --apply
```

Options and defaults:

- `--database-url` — default: `None`
- `--apply` — default: `False`

## `quantagent api schema`

```text
Usage: quantagent api schema [OPTIONS]

  Export or check the live OpenAPI contract.

Options:
  --output PATH  JSON contract output path.  [default: docs/api/openapi.json]
  --check        Compare with the output file without writing; default:
                 export.
  --help         Show this message and exit.

  Example: quantagent api schema --check
```

Options and defaults:

- `--output` — default: `PosixPath('docs/api/openapi.json')`
- `--check` — default: `False`

## `quantagent api serve`

```text
Usage: quantagent api serve [OPTIONS]

  Serve the read-only QuantAgent API.

Options:
  --host TEXT     Interface address to bind.  [default: 127.0.0.1]
  --port INTEGER  TCP port to bind.  [default: 8000]
  --help          Show this message and exit.

  Example: quantagent api serve --port 8000
```

Options and defaults:

- `--host` — default: `'127.0.0.1'`
- `--port` — default: `8000`

## `quantagent api sync-calendar`

```text
Usage: quantagent api sync-calendar [OPTIONS]

  Synchronize Alpaca market holidays and early closes.

Options:
  --help  Show this message and exit.

  Example: quantagent api sync-calendar
```

## `quantagent api sync-fred-calendar`

```text
Usage: quantagent api sync-fred-calendar [OPTIONS]

  Synchronize FRED releases and manual calendar events.

Options:
  --help  Show this message and exit.

  Example: quantagent api sync-fred-calendar
```
