# `quantagent api accounts flows` command reference

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

## `quantagent api accounts flows list`

```text
Usage: quantagent api accounts flows list [OPTIONS] ACCOUNT_ID

  List manual flows for an account.

Options:
  --help  Show this message and exit.

  Example: quantagent api accounts flows list paper-a
```

## `quantagent api accounts flows reconcile`

```text
Usage: quantagent api accounts flows reconcile [OPTIONS] ACCOUNT_ID FLOW_ID

  Mark a registered flow as reconciled.

Options:
  --help  Show this message and exit.

  Example: quantagent api accounts flows reconcile paper-a 12
```

## `quantagent api accounts flows register`

```text
Usage: quantagent api accounts flows register [OPTIONS] ACCOUNT_ID

  Register a manual account deposit or withdrawal.

Options:
  --date [%Y-%m-%d]            Flow date in YYYY-MM-DD format.  [required]
  --amount FLOAT               Positive amount in the account currency.
                               [required]
  --type [deposit|withdrawal]  Flow type; withdrawal is stored with a negative
                               sign.  [required]
  --help                       Show this message and exit.

  Example: quantagent api accounts flows register paper-a --date 2026-11-01
  --amount 1000 --type deposit
```

Options and defaults:

- `--date` — default: `None`
- `--amount` — default: `None`
- `--type` — default: `None`
