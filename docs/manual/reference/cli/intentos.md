# `quantagent intentos` command reference

```text
Usage: quantagent intentos [OPTIONS] COMMAND [ARGS]...

  Manage the run attempts log.

Options:
  --help  Show this message and exit.

Commands:
  epoca-limpia  Record the start of the clean period.
  invalidar     List matching rows; invalidate them with a backup only...
```

## `quantagent intentos epoca-limpia`

```text
Usage: quantagent intentos epoca-limpia [OPTIONS]

  Record the start of the clean period.

Options:
  --desde TEXT   Start timestamp in ISO 8601 UTC format.  [required]
  --motivo TEXT  Reason for recording the clean period.  [required]
  --help         Show this message and exit.
```

Options and defaults:

- `--desde` — default: `None`
- `--motivo` — default: `None`

## `quantagent intentos invalidar`

```text
Usage: quantagent intentos invalidar [OPTIONS]

  List matching rows; invalidate them with a backup only when --aplicar is
  set.

Options:
  --hipotesis TEXT  Hypothesis ID whose attempts should be selected.
                    [required]
  --desde TEXT      Inclusive start timestamp in ISO 8601 UTC format.
  --hasta TEXT      Inclusive end timestamp in ISO 8601 UTC format.
  --motivo TEXT     Reason for invalidating the selected attempts.  [required]
  --aplicar         Write invalidation marks and create a backup; otherwise
                    show a dry run.
  --help            Show this message and exit.
```

Options and defaults:

- `--hipotesis` — default: `None`
- `--desde` — default: `None`
- `--hasta` — default: `None`
- `--motivo` — default: `None`
- `--aplicar` — default: `False`
