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
```

## `quantagent trading digest`

```text
Usage: quantagent trading digest [OPTIONS] [ARCHIVO]

  Show the daily paper trading digest.

Options:
  --date [%Y-%m-%d]    Digest date in YYYY-MM-DD format.
  --bucket [m5]        Trading bucket for the digest.
  --db FILE            SQLite file; M5 defaults to local-data/m5.db, otherwise
                       DATABASE_URL.
  --shadow FILE        JSON file with shadow weights.
  --portfolio TEXT     Dedicated portfolio for positions, executions, and
                       current Alpaca valuation.
  --dividendos-alpaca  Read Alpaca dividends for the selected date.
  --sin-dividendos     Disable the Alpaca dividend query.
  --help               Show this message and exit.

  Example: quantagent trading digest --bucket m5 --date 2026-10-29
```

Options and defaults:

- `--date` — default: `None`
- `--bucket` — default: `None`
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
