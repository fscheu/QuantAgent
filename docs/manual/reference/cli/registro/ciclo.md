# `quantagent registro ciclo` command reference

```text
Usage: quantagent registro ciclo [OPTIONS] COMMAND [ARGS]...

  Open or close a monthly hypothesis cycle.

Options:
  --help  Show this message and exit.

Commands:
  abrir    Open a cycle with up to five hypothesis IDs.
  cerrar   Close an open cycle and save its immutable receipt.
  informe  Show a phone-friendly report for a monthly cycle.

  Example: quantagent registro ciclo abrir 2026-11 H1 H2
```

## `quantagent registro ciclo abrir`

```text
Usage: quantagent registro ciclo abrir [OPTIONS] CICLO HIPOTESIS...

  Open a cycle with up to five hypothesis IDs.

Options:
  --help  Show this message and exit.

  Example: quantagent registro ciclo abrir 2026-11 H1 H2
```

## `quantagent registro ciclo cerrar`

```text
Usage: quantagent registro ciclo cerrar [OPTIONS] CICLO

  Close an open cycle and save its immutable receipt.

Options:
  --help  Show this message and exit.

  Example: quantagent registro ciclo cerrar 2026-11
```

## `quantagent registro ciclo informe`

```text
Usage: quantagent registro ciclo informe [OPTIONS] CICLO

  Show a phone-friendly report for a monthly cycle.

Options:
  --help  Show this message and exit.

  Example: quantagent registro ciclo informe 2026-11
```
