# `quantagent api` command reference

```text
Usage: quantagent api [OPTIONS] COMMAND [ARGS]...

  Run the QuantAgent HTTP API.

Options:
  --help  Show this message and exit.

Commands:
  accounts       Manage configured trading accounts.
  schema         Export or check the live OpenAPI contract.
  serve          Serve the read-only QuantAgent API.
  sync-calendar  Synchronize Alpaca market holidays and early closes.
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
