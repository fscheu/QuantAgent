# `quantagent api accounts` command reference

```text
Usage: quantagent api accounts [OPTIONS] COMMAND [ARGS]...

  Manage configured trading accounts.

Options:
  --help  Show this message and exit.

Commands:
  add      Register an account and its strategy.
  archive  Archive an account so it is hidden from the API.
  flows    Register and reconcile manual deposits and withdrawals.
  list     List active accounts.
  show     Show one account by slug.
```

## `quantagent api accounts add`

```text
Usage: quantagent api accounts add [OPTIONS]

  Register an account and its strategy.

Options:
  --id TEXT                  Unique account slug.  [required]
  --name TEXT                Display name for the account.  [required]
  --env [paper|live]         Trading environment: paper or live.  [required]
  --kind [alpaca|simulated]  Account source: Alpaca or simulated.  [required]
  --alpaca-account-ref TEXT  Informational Alpaca account reference.
  --database-url TEXT        Readable SQLite database URL for this account.
                             [required]
  --bucket-id TEXT           Informational trade-intent bucket identifier.
  --strategy TEXT            Single strategy assigned to this account.
                             [required]
  --help                     Show this message and exit.

  Example: quantagent api accounts add --id paper-nucleo-curva --name 'Paper
  Nucleo Curva' --env paper --kind alpaca --database-url
  sqlite:////srv/quantagent/paper.db --strategy nucleo-curva
```

Options and defaults:

- `--id` — default: `None`
- `--name` — default: `None`
- `--env` — default: `None`
- `--kind` — default: `None`
- `--alpaca-account-ref` — default: `None`
- `--database-url` — default: `None`
- `--bucket-id` — default: `None`
- `--strategy` — default: `None`

## `quantagent api accounts archive`

```text
Usage: quantagent api accounts archive [OPTIONS] ACCOUNT_ID

  Archive an account so it is hidden from the API.

Options:
  --help  Show this message and exit.

  Example: quantagent api accounts archive paper-nucleo-curva
```

## `quantagent api accounts list`

```text
Usage: quantagent api accounts list [OPTIONS]

  List active accounts.

Options:
  --help  Show this message and exit.

  Example: quantagent api accounts list
```

## `quantagent api accounts show`

```text
Usage: quantagent api accounts show [OPTIONS] ACCOUNT_ID

  Show one account by slug.

Options:
  --help  Show this message and exit.

  Example: quantagent api accounts show paper-nucleo-curva
```

## `quantagent api accounts flows`

```text
Usage: quantagent api accounts flows [OPTIONS] COMMAND [ARGS]...

  Register and reconcile manual deposits and withdrawals.

Commands:
  list       List manual flows for an account.
  reconcile  Mark a registered flow as reconciled.
  register   Register a manual account deposit or withdrawal.
```

## `quantagent api accounts flows register`

```text
Usage: quantagent api accounts flows register [OPTIONS] ACCOUNT_ID

Options:
  --date TEXT                     Flow date in YYYY-MM-DD format. [required]
  --amount FLOAT                  Positive amount in the account currency. [required]
  --type [deposit|withdrawal]     Flow type; withdrawal is stored with a negative sign. [required]
  --help                          Show this message and exit.

  Example: quantagent api accounts flows register paper-a --date 2026-11-01 --amount 1000 --type deposit
```

## `quantagent api accounts flows list`

```text
Usage: quantagent api accounts flows list [OPTIONS] ACCOUNT_ID

  List manual flows for an account.

  Example: quantagent api accounts flows list paper-a
```

## `quantagent api accounts flows reconcile`

```text
Usage: quantagent api accounts flows reconcile [OPTIONS] ACCOUNT_ID FLOW_ID

  Mark a registered flow as reconciled.

  Example: quantagent api accounts flows reconcile paper-a 12
```
