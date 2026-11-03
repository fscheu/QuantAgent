# `quantagent registro` command reference

```text
Usage: quantagent registro [OPTIONS] COMMAND [ARGS]...

  Manage hypothesis evaluation cycles.

Options:
  --help  Show this message and exit.

Commands:
  calendario  Show monthly hypothesis cycle dates and registry status.
  ciclo       Open or close a monthly hypothesis cycle.
  corte       Show consecutive closed cycles without a stage 3 pass.
  importar    Import hypothesis stages and attempts from snapshot CSV files.
  ver         List hypothesis registry events in chronological order.

  Examples: quantagent registro ciclo abrir 2026-11 H1 H2 quantagent registro
  ciclo cerrar 2026-11
```

## `quantagent registro calendario`

```text
Usage: quantagent registro calendario [OPTIONS] [DESDE]

  Show monthly hypothesis cycle dates and registry status.

Options:
  --help  Show this message and exit.

  Example: quantagent registro calendario 2026-11
```

## `quantagent registro ciclo`

```text
Usage: quantagent registro ciclo [OPTIONS] COMMAND [ARGS]...

  Open or close a monthly hypothesis cycle.

Options:
  --help  Show this message and exit.

Commands:
  abrir   Open a cycle with up to five hypothesis IDs.
  cerrar  Close an open cycle and save its immutable receipt.

  Example: quantagent registro ciclo abrir 2026-11 H1 H2
```

## `quantagent registro corte`

```text
Usage: quantagent registro corte [OPTIONS]

  Show consecutive closed cycles without a stage 3 pass.

Options:
  --help  Show this message and exit.

  Example: quantagent registro corte
```

## `quantagent registro importar`

```text
Usage: quantagent registro importar [OPTIONS]

  Import hypothesis stages and attempts from snapshot CSV files.

Options:
  --dry-run  Preview counts without writing to the hypothesis registry.
  --help     Show this message and exit.

  Example: quantagent registro importar --dry-run
```

Options and defaults:

- `--dry-run` — default: `False`

## `quantagent registro ver`

```text
Usage: quantagent registro ver [OPTIONS]

  List hypothesis registry events in chronological order.

Options:
  --hipotesis TEXT  Show events for this hypothesis ID.
  --help            Show this message and exit.

  Examples: quantagent registro ver quantagent registro ver --hipotesis H1
```

Options and defaults:

- `--hipotesis` — default: `None`
