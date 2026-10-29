# `quantagent backtest hipotesis propuestas` command reference

```text
Usage: quantagent backtest hipotesis propuestas [OPTIONS] COMMAND [ARGS]...

  Review catalog proposals by approving or rejecting them.

Options:
  --help  Show this message and exit.

Commands:
  aprobar   Move a proposal from catalog/proposals to hypotheses and...
  rechazar  Archive a proposal with a reason.
```

## `quantagent backtest hipotesis propuestas aprobar`

```text
Usage: quantagent backtest hipotesis propuestas aprobar [OPTIONS] PROPUESTA_ID

  Move a proposal from catalog/proposals to hypotheses and record approval.

Options:
  --quien, --autor TEXT       Name of the approver.  [required]
  --propuestas-dir DIRECTORY  Proposal directory (defaults to the catalog
                              directory).
  --hipotesis-dir DIRECTORY   Hypothesis directory (defaults to the configured
                              directory).
  --help                      Show this message and exit.
```

Options and defaults:

- `--quien, --autor` — default: `None`
- `--propuestas-dir` — default: `None`
- `--hipotesis-dir` — default: `None`

## `quantagent backtest hipotesis propuestas rechazar`

```text
Usage: quantagent backtest hipotesis propuestas rechazar [OPTIONS]
                                                         PROPUESTA_ID

  Archive a proposal with a reason.

Options:
  --motivo TEXT               Reason for rejection.  [required]
  --quien, --autor TEXT       Name of the reviewer.  [required]
  --propuestas-dir DIRECTORY  Proposal directory (defaults to the catalog
                              directory).
  --help                      Show this message and exit.
```

Options and defaults:

- `--motivo` — default: `None`
- `--quien, --autor` — default: `None`
- `--propuestas-dir` — default: `None`
