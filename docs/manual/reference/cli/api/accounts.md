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

## `quantagent api accounts flows`

```text
Usage: quantagent api accounts flows [OPTIONS] COMMAND [ARGS]...

  Register and reconcile manual deposits and withdrawals.

Options:
  --help  Show this message and exit.

Commands:
  list       List manual flows for an account.
  reconcile  Mark a registered flow as reconciled.
  register   Register a manual account deposit or withdrawal.
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
