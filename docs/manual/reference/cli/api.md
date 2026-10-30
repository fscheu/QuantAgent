# `quantagent api` command reference

```text
Usage: quantagent api [OPTIONS] COMMAND [ARGS]...

  Run the QuantAgent HTTP API.

Options:
  --help  Show this message and exit.

Commands:
  serve          Serve the read-only QuantAgent API.
  sync-calendar  Synchronize Alpaca market holidays and early closes.
```

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
