# `quantagent registro` command reference

```text
Usage: quantagent registro [OPTIONS] COMMAND [ARGS]...

  Manage hypothesis evaluation cycles.

Options:
  --help  Show this message and exit.

Commands:
  ciclo  Open or close a monthly hypothesis cycle.

  Examples: quantagent registro ciclo abrir 2026-11 H1 H2 quantagent registro
  ciclo cerrar 2026-11
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
