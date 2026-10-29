# `quantagent backtest hipotesis` command reference

```text
Usage: quantagent backtest hipotesis [OPTIONS] COMMAND [ARGS]...

  Evaluate strategy factory hypotheses.

Options:
  --help  Show this message and exit.

Commands:
  comparar           Compare a core portfolio with the same portfolio...
  correr             Run stages 0 through 3 with early stopping and...
  filtrar            Load and admit a hypothesis over 2007–2018.
  informe            Generate an HTML page with a hypothesis verdict and...
  proponer           Request proposals from the LLM via...
  propuestas         Review catalog proposals by approving or rejecting...
  reserva            Check eligibility and date range for the 2023+...
  reserva-veredicto  Recover the verdict from saved equity curves.
  resumen            Summarize recent statuses and batch attempts.
  tanda              Process YAML files in order and save reports, logs,...
```

## `quantagent backtest hipotesis comparar`

```text
Usage: quantagent backtest hipotesis comparar [OPTIONS] NUCLEO CON_CAPA

  Compare a core portfolio with the same portfolio plus one overlay (ΔSharpe,
  ΔDSR).

Options:
  --snapshot TEXT      Snapshot used for evaluation.  [required]
  --carpeta DIRECTORY  Directory for comparison outputs.  [required]
  --margen FLOAT       Minimum ΔDSR for accepting the overlay; set before
                       running.  [required]
  --desde TEXT         First date in the evaluation period.  [default:
                       2007-01-01]
  --hasta TEXT         Last date in the evaluation period.  [default:
                       2018-12-31]
  --out FILE           Write the comparison report to this HTML file.
  --abrir-prueba       Run the test period from 2019-01-01 and open its lock
                       once for the overlaid portfolio.
  --help               Show this message and exit.
```

Options and defaults:

- `--snapshot` — default: `None`
- `--carpeta` — default: `None`
- `--margen` — default: `None`
- `--desde` — default: `'2007-01-01'`
- `--hasta` — default: `'2018-12-31'`
- `--out` — default: `None`
- `--abrir-prueba` — default: `False`

## `quantagent backtest hipotesis correr`

```text
Usage: quantagent backtest hipotesis correr [OPTIONS] ARCHIVO

  Run stages 0 through 3 with early stopping and status tracking.

Options:
  --snapshot TEXT              Snapshot used for evaluation.  [required]
  --carpeta DIRECTORY          Directory for stage outputs.  [required]
  --hasta-etapa INTEGER RANGE  Stop after this stage (0 through 3).  [default:
                               3; 0<=x<=3]
  --help                       Show this message and exit.
```

Options and defaults:

- `--snapshot` — default: `None`
- `--carpeta` — default: `None`
- `--hasta-etapa` — default: `3`

## `quantagent backtest hipotesis filtrar`

```text
Usage: quantagent backtest hipotesis filtrar [OPTIONS] ARCHIVO

  Load and admit a hypothesis over 2007–2018.

Options:
  --snapshot TEXT  Snapshot used for evaluation.  [required]
  --out FILE       Write results and per-symbol parameters to this CSV.
  --help           Show this message and exit.
```

Options and defaults:

- `--snapshot` — default: `None`
- `--out` — default: `None`

## `quantagent backtest hipotesis informe`

```text
Usage: quantagent backtest hipotesis informe [OPTIONS] HIPOTESIS_ID

  Generate an HTML page with a hypothesis verdict and its reasons.

Options:
  --out FILE                  Output HTML file.  [required]
  --archivo FILE              Hypothesis YAML file.
  --carpeta-equity DIRECTORY  Directory containing saved equity curves.
  --snapshot TEXT             Snapshot used to rebuild the benchmark.
  --help                      Show this message and exit.
```

Options and defaults:

- `--out` — default: `None`
- `--archivo` — default: `None`
- `--carpeta-equity` — default: `None`
- `--snapshot` — default: `None`

## `quantagent backtest hipotesis proponer`

```text
Usage: quantagent backtest hipotesis proponer [OPTIONS]

  Request proposals from the LLM via QUANTAGENT_LLM_CMD and validate them.

Options:
  --carpeta DIRECTORY  Directory for accepted proposals.
  --help               Show this message and exit.
```

Options and defaults:

- `--carpeta` — default: `None`

## `quantagent backtest hipotesis propuestas`

```text
Usage: quantagent backtest hipotesis propuestas [OPTIONS] COMMAND [ARGS]...

  Review catalog proposals by approving or rejecting them.

Options:
  --help  Show this message and exit.

Commands:
  aprobar   Move a proposal from catalog/proposals to hypotheses and...
  rechazar  Archive a proposal with a reason.
```

## `quantagent backtest hipotesis reserva`

```text
Usage: quantagent backtest hipotesis reserva [OPTIONS] ARCHIVO

  Check eligibility and date range for the 2023+ reserve period.

Options:
  --snapshot TEXT  Snapshot used to determine the available date range.
                   [required]
  --abrir          Register and open the reserve period once.
  --help           Show this message and exit.
```

Options and defaults:

- `--snapshot` — default: `None`
- `--abrir` — default: `False`

## `quantagent backtest hipotesis reserva-veredicto`

```text
Usage: quantagent backtest hipotesis reserva-veredicto [OPTIONS] ARCHIVO

  Recover the verdict from saved equity curves.

Options:
  --snapshot TEXT  Snapshot used for the evaluation.  [required]
  --help           Show this message and exit.
```

Options and defaults:

- `--snapshot` — default: `None`

## `quantagent backtest hipotesis resumen`

```text
Usage: quantagent backtest hipotesis resumen [OPTIONS]

  Summarize recent statuses and batch attempts.

Options:
  --desde [%Y-%m-%d|%Y-%m-%dT%H:%M:%S%z]
                                  Inclusive UTC start date or timezone-aware
                                  datetime.
  --out FILE                      Write the summary to this file.
  --help                          Show this message and exit.
```

Options and defaults:

- `--desde` — default: `None`
- `--out` — default: `None`

## `quantagent backtest hipotesis tanda`

```text
Usage: quantagent backtest hipotesis tanda [OPTIONS] CARPETA_YAML

  Process YAML files in order and save reports, logs, and a summary.

Options:
  --snapshot TEXT      Snapshot to evaluate against.  [required]
  --carpeta DIRECTORY  Directory for reports and run logs.  [required]
  --help               Show this message and exit.
```

Options and defaults:

- `--snapshot` — default: `None`
- `--carpeta` — default: `None`
