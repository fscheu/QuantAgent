# `quantagent trading` command reference

```text
Usage: quantagent trading [OPTIONS] COMMAND [ARGS]...

  Paper trading operations.

Options:
  --help  Show this message and exit.

Commands:
  digest        Show the daily paper trading digest.
  pre-apertura  Prepare paper stops and expire stale entry orders before...
  reconciliar   Compare local PAPER positions with Alpaca.
  snapshot      Save today's Alpaca account equity and cash snapshot.
```

## `quantagent trading digest`

```text
Usage: quantagent trading digest [OPTIONS] [ARCHIVO]

  Show the daily paper trading digest.

Options:
  --date [%Y-%m-%d]    Digest date in YYYY-MM-DD format.
  --db FILE            SQLite file; defaults to DATABASE_URL.
  --shadow FILE        JSON file with shadow weights.
  --portfolio TEXT     Dedicated portfolio for positions, executions, and
                       current Alpaca valuation.
  --dividendos-alpaca  Read Alpaca dividends for the selected date.
  --sin-dividendos     Disable the Alpaca dividend query.
  --help               Show this message and exit.
```

Options and defaults:

- `--date` — default: `None`
- `--db` — default: `None`
- `--shadow` — default: `None`
- `--portfolio` — default: `None`
- `--dividendos-alpaca` — default: `False`
- `--sin-dividendos` — default: `False`

## `quantagent trading pre-apertura`

```text
Usage: quantagent trading pre-apertura [OPTIONS]

  Prepare paper stops and expire stale entry orders before market open.

Options:
  --help  Show this message and exit.

  Example: quantagent trading pre-apertura
```

## `quantagent trading reconciliar`

```text
Usage: quantagent trading reconciliar [OPTIONS]

  Compare local PAPER positions with Alpaca.

Options:
  --help  Show this message and exit.
```

## `quantagent trading snapshot`

```text
Usage: quantagent trading snapshot [OPTIONS]

  Save today's Alpaca account equity and cash snapshot.

Options:
  --db FILE         SQLite database file to update.  [required]
  --portfolio TEXT  Portfolio profile used to connect to Alpaca.  [required]
  --help            Show this message and exit.

  Example: quantagent trading snapshot --db paper.db --portfolio paper-m4
```

Options and defaults:

- `--db` — default: `None`
- `--portfolio` — default: `None`
