# `quantagent mercado` command reference

```text
Usage: quantagent mercado [OPTIONS] COMMAND [ARGS]...

  Manage stored market data.

Options:
  --help  Show this message and exit.

Commands:
  barras  Update stored daily market bars.
  macro   Manage stored FRED macroeconomic observations.
```

## `quantagent mercado barras`

```text
Usage: quantagent mercado barras [OPTIONS] COMMAND [ARGS]...

  Update stored daily market bars.

Options:
  --help  Show this message and exit.

Commands:
  actualizar  Update daily bars from Alpaca or a frozen snapshot.
```

## `quantagent mercado macro`

```text
Usage: quantagent mercado macro [OPTIONS] COMMAND [ARGS]...

  Manage stored FRED macroeconomic observations.

Options:
  --help  Show this message and exit.

Commands:
  actualizar  Update macroeconomic observations from FRED.
```
