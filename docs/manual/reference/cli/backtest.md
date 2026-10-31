# `quantagent backtest` command reference

```text
Usage: quantagent backtest [OPTIONS] COMMAND [ARGS]...

  Run deterministic backtests against versioned fixtures.

Options:
  --help  Show this message and exit.

Commands:
  attempts     Count runs and parameter combinations recorded by sweep.
  compare      Compare multiple strategies across symbols against a...
  data         Manage frozen market data.
  hipotesis    Evaluate strategy factory hypotheses.
  run          Run a backtest against a versioned fixture.
  sweep        Run a strategy parameter grid against an asset.
  verdict      Evaluate strategies against a comparison table.
  verify       Verify backtest reproducibility across two runs.
  walkforward  Select parameters on training data and evaluate the winner...
```

## `quantagent backtest attempts`

```text
Usage: quantagent backtest attempts [OPTIONS]

  Count runs and parameter combinations recorded by sweep.

Options:
  --strategy TEXT  Strategy whose attempts to count.  [required]
  --symbol TEXT    Limit the count to this symbol.
  --help           Show this message and exit.
```

Options and defaults:

- `--strategy` — default: `None`
- `--symbol` — default: `None`

## `quantagent backtest compare`

```text
Usage: quantagent backtest compare [OPTIONS]

  Compare multiple strategies across symbols against a snapshot.

Options:
  --strategies TEXT               Comma-separated strategies (e.g. rsi,sma-
                                  cross).  [required]
  --symbols TEXT                  Comma-separated symbols (e.g. SPY,IWM).
                                  [required]
  --snapshot TEXT                 Snapshot in $QUANTAGENT_SNAPSHOT_DIR
                                  (adjusted bars).  [required]
  --from [%Y-%m-%d]               First session.
  --to [%Y-%m-%d]                 Last session.
  --costos [cripto|etf|etf-conservador]
                                  Cost profile.
  --tamano-posicion FLOAT RANGE   Capital fraction per position, in (0, 1].
                                  Sets the run's base and maximum size.
                                  [0.0<x<=1.0]
  --regla-tamano [confianza|fijo|riesgo]
                                  Position sizing rule: confidence or fixed.
  --orden-minima FLOAT RANGE      Minimum USD amount per entry order.  [x>0.0]
  --abrir-reserva                 Allow access to the reserve period.
  --abrir-prueba                  Allow access to the test period.
  --out FILE                      Output CSV file.  [required]
  --trades-dir DIRECTORY          Directory for per-strategy trade CSV files.
  --equity-dir DIRECTORY          Engine date,equity CSV per bar (equity
                                  rounded to 2 decimals).
  --md FILE                       Output Markdown document.
  --crisis                        Add crisis_2008/2020/2022: annual return
                                  minus buy and hold (informational).
  --llenado [apertura|cierre]     Fill price: opening or closing price of the
                                  candle.  [default: apertura]
  --params FILE                   JSON {"<strategy>": {"<parameter>": value}}
                                  with strategy parameters (default: registry
                                  values).
  --help                          Show this message and exit.
```

Options and defaults:

- `--strategies` — default: `None`
- `--symbols` — default: `None`
- `--snapshot` — default: `None`
- `--from` — default: `None`
- `--to` — default: `None`
- `--costos` — default: `None`
- `--tamano-posicion` — default: `1.0`
- `--regla-tamano` — default: `None`
- `--orden-minima` — default: `None`
- `--abrir-reserva` — default: `False`
- `--abrir-prueba` — default: `False`
- `--out` — default: `None`
- `--trades-dir` — default: `None`
- `--equity-dir` — default: `None`
- `--md` — default: `None`
- `--crisis` — default: `False`
- `--llenado` — default: `'apertura'`
- `--params` — default: `None`

## `quantagent backtest data`

```text
Usage: quantagent backtest data [OPTIONS] COMMAND [ARGS]...

  Manage frozen market data.

Options:
  --help  Show this message and exit.

Commands:
  acciones       Manage S&P 500 stock snapshots for 2007-2018 using Yahoo...
  consenso       Manage daily earnings and revenue estimates.
  fundamentales  Manage SEC EDGAR (XBRL) fundamentals snapshots.
  macro          Manage ALFRED macro snapshots (all vintages) in Parquet.
  snapshot       Manage Parquet snapshots outside the repository...
```

## `quantagent backtest hipotesis`

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

## `quantagent backtest run`

```text
Usage: quantagent backtest run [OPTIONS]

  Run a backtest against a versioned fixture.

Options:
  --strategy [arquetipo-a|arquetipo-b|curva-pendiente|fifty-two-week-high|halloween|momentum-12m|rebalanceo-fin-de-mes|rsi|rsi-regimen|senal-externa|sma-cross|triple-screen|tsmom-largo-corto|turn-of-the-month]
                                  Strategy to run.  [required]
  --fixture TEXT                  Fixture name under tests/fixtures/ (without
                                  the .csv extension). Mutually exclusive with
                                  --snapshot.
  --snapshot TEXT                 Snapshot in $QUANTAGENT_SNAPSHOT_DIR
                                  (adjusted bars). Requires --symbol.
  --symbol TEXT                   Snapshot symbol (one per run).
  --from [%Y-%m-%d]               First session, YYYY-MM-DD (default: first in
                                  the snapshot).
  --to [%Y-%m-%d]                 Last session, YYYY-MM-DD (default: last in
                                  the snapshot).
  --abrir-reserva                 Allow --to to reach the reserve period (from
                                  $QUANTAGENT_RESERVA_DESDE, default
                                  2023-01-01). Logged.
  --abrir-prueba                  Allow --to to reach the test period (from
                                  $QUANTAGENT_PRUEBA_DESDE, default
                                  2019-01-01). Logged.
  --out FILE                      Write the trade log CSV to this path.
  --equity-out FILE               Write the equity curve CSV (one row per
                                  candle) to this path.
  --verbose                       Also show the engine's per-candle
                                  'Insufficient data' messages on stderr.
  --commission-pct FLOAT RANGE    Commission per side on notional (0.001 =
                                  0.10%). Default: TRADING_COMMISSION_PCT.
                                  [x>=0.0]
  --costos [cripto|etf|etf-conservador]
                                  Cost profile (slippage and commission per
                                  side). Mutually exclusive with --commission-
                                  pct.
  --llenado [apertura|cierre]     Fill price: opening or closing price of the
                                  candle.  [default: apertura]
  --cuenta-chica FLOAT RANGE      Initial backtest capital in USD; enables
                                  whole-share entries. Default: disabled
                                  ($100,000 capital).  [x>0.0]
  --tamano-posicion FLOAT RANGE   Capital fraction per position, in (0, 1]
                                  (1.0 = 100%). Sets the run's base and
                                  maximum size.  [0.0<x<=1.0]
  --riesgo-por-trade FLOAT RANGE  Capital fraction at risk per trade.  [x>0.0]
  --regla-tamano [confianza|fijo|riesgo]
                                  Position sizing rule: confidence (capital ×
                                  fraction × signal confidence) or fixed.
                                  Default: confidence.
  --orden-minima FLOAT RANGE      Minimum USD amount per entry order.  [x>0.0]
  --params FILE                   JSON {"<strategy>": {"<parameter>": value}}
                                  with strategy parameters (default: registry
                                  values).
  --intrabar-stops / --no-intrabar-stops
                                  Evaluate stop loss and take profit against
                                  intrabar high/low prices.
  --help                          Show this message and exit.

  Example: quantagent backtest run --strategy rsi --fixture spy-90d --cuenta-
  chica 2500
```

Options and defaults:

- `--strategy` — default: `None`
- `--fixture` — default: `None`
- `--snapshot` — default: `None`
- `--symbol` — default: `None`
- `--from` — default: `None`
- `--to` — default: `None`
- `--abrir-reserva` — default: `False`
- `--abrir-prueba` — default: `False`
- `--out` — default: `None`
- `--equity-out` — default: `None`
- `--verbose` — default: `False`
- `--commission-pct` — default: `None`
- `--costos` — default: `None`
- `--llenado` — default: `'apertura'`
- `--cuenta-chica` — default: `None`
- `--tamano-posicion` — default: `None`
- `--riesgo-por-trade` — default: `None`
- `--regla-tamano` — default: `None`
- `--orden-minima` — default: `None`
- `--params` — default: `None`
- `--intrabar-stops, --no-intrabar-stops` — default: `True`

## `quantagent backtest sweep`

```text
Usage: quantagent backtest sweep [OPTIONS]

  Run a strategy parameter grid against an asset.

Options:
  --llenado [apertura|cierre]     Fill orders at the open or close.  [default:
                                  apertura]
  --strategy TEXT                 Strategy to optimize.  [required]
  --symbol TEXT                   Asset symbol (e.g. SPY).  [required]
  --snapshot TEXT                 Snapshot in $QUANTAGENT_SNAPSHOT_DIR
                                  (adjusted candles).  [required]
  --grid FILE                     JSON file containing the parameter grid.
  --vecinos-20                    Default and ±20% per parameter, one at a
                                  time; excludes --grid.
  --out FILE                      Output CSV file.  [required]
  --from [%Y-%m-%d]               First session.
  --to [%Y-%m-%d]                 Last session.
  --costos [cripto|etf|etf-conservador]
                                  Cost profile.
  --tamano-posicion FLOAT RANGE   Position size as a fraction of equity.
                                  [0.0<x<=1.0]
  --regla-tamano [confianza|fijo|riesgo]
                                  Position sizing rule.
  --riesgo-por-trade FLOAT RANGE  Risk per trade as a fraction of equity.
                                  [x>0.0]
  --orden-minima FLOAT RANGE      Minimum USD amount per entry order.  [x>0.0]
  --help                          Show this message and exit.
```

Options and defaults:

- `--llenado` — default: `'apertura'`
- `--strategy` — default: `None`
- `--symbol` — default: `None`
- `--snapshot` — default: `None`
- `--grid` — default: `None`
- `--vecinos-20` — default: `False`
- `--out` — default: `None`
- `--from` — default: `None`
- `--to` — default: `None`
- `--costos` — default: `None`
- `--tamano-posicion` — default: `1.0`
- `--regla-tamano` — default: `None`
- `--riesgo-por-trade` — default: `None`
- `--orden-minima` — default: `None`

## `quantagent backtest verdict`

```text
Usage: quantagent backtest verdict [OPTIONS]

  Evaluate strategies against a comparison table.

Options:
  --tabla FILE            Input comparison CSV file.  [required]
  --out FILE              Output CSV file.  [required]
  --trades-dir DIRECTORY  Add PnL concentration and yearly PnL from trade CSV
                          files.
  --equity-dir DIRECTORY  Add the p-value, Monte Carlo final percentile, and
                          informational p95 drawdown.
  --help                  Show this message and exit.
```

Options and defaults:

- `--tabla` — default: `None`
- `--out` — default: `None`
- `--trades-dir` — default: `None`
- `--equity-dir` — default: `None`

## `quantagent backtest verify`

```text
Usage: quantagent backtest verify [OPTIONS]

  Verify backtest reproducibility across two runs.

Options:
  --strategy [arquetipo-a|arquetipo-b|curva-pendiente|fifty-two-week-high|halloween|momentum-12m|rebalanceo-fin-de-mes|rsi|rsi-regimen|senal-externa|sma-cross|triple-screen|tsmom-largo-corto|turn-of-the-month]
                                  Strategy to run.  [required]
  --fixture TEXT                  Fixture name under tests/fixtures/ (without
                                  the .csv extension). Mutually exclusive with
                                  --snapshot.
  --snapshot TEXT                 Snapshot in $QUANTAGENT_SNAPSHOT_DIR
                                  (adjusted bars). Requires --symbol.
  --symbol TEXT                   Snapshot symbol (one per run).
  --from [%Y-%m-%d]               First session, YYYY-MM-DD (default: first in
                                  the snapshot).
  --to [%Y-%m-%d]                 Last session, YYYY-MM-DD (default: last in
                                  the snapshot).
  --abrir-reserva                 Allow --to to reach the reserve period (from
                                  $QUANTAGENT_RESERVA_DESDE, default
                                  2023-01-01). Logged.
  --abrir-prueba                  Allow --to to reach the test period (from
                                  $QUANTAGENT_PRUEBA_DESDE, default
                                  2019-01-01). Logged.
  --verbose                       Also show engine warnings on stderr.
  --commission-pct FLOAT RANGE    Commission per side on notional (0.001 =
                                  0.10%). Default: TRADING_COMMISSION_PCT.
                                  [x>=0.0]
  --costos [cripto|etf|etf-conservador]
                                  Cost profile (slippage and commission per
                                  side). Mutually exclusive with --commission-
                                  pct.
  --tamano-posicion FLOAT RANGE   Capital fraction per position, in (0, 1]
                                  (1.0 = 100%). Sets the run's base and
                                  maximum size.  [0.0<x<=1.0]
  --riesgo-por-trade FLOAT RANGE  Capital fraction at risk per trade.  [x>0.0]
  --regla-tamano [confianza|fijo|riesgo]
                                  Position sizing rule: confidence (capital ×
                                  fraction × signal confidence) or fixed.
                                  Default: confidence.
  --orden-minima FLOAT RANGE      Minimum USD amount per entry order.  [x>0.0]
  --help                          Show this message and exit.
```

Options and defaults:

- `--strategy` — default: `None`
- `--fixture` — default: `None`
- `--snapshot` — default: `None`
- `--symbol` — default: `None`
- `--from` — default: `None`
- `--to` — default: `None`
- `--abrir-reserva` — default: `False`
- `--abrir-prueba` — default: `False`
- `--verbose` — default: `False`
- `--commission-pct` — default: `None`
- `--costos` — default: `None`
- `--tamano-posicion` — default: `None`
- `--riesgo-por-trade` — default: `None`
- `--regla-tamano` — default: `None`
- `--orden-minima` — default: `None`

## `quantagent backtest walkforward`

```text
Usage: quantagent backtest walkforward [OPTIONS]

  Select parameters on training data and evaluate the winner on the next
  window.

Options:
  --llenado [apertura|cierre]     Fill orders at the open or close.  [default:
                                  apertura]
  --strategy TEXT                 Strategy to evaluate.  [required]
  --symbol TEXT                   Market symbol to evaluate.  [required]
  --snapshot TEXT                 Market data snapshot name.  [required]
  --grid FILE                     Parameter grid YAML file.  [required]
  --out FILE                      Output CSV file.  [required]
  --from [%Y-%m-%d]               First date in YYYY-MM-DD format.
  --to [%Y-%m-%d]                 Last date in YYYY-MM-DD format.
  --train-years INTEGER RANGE     Training window length in years.  [x>=1]
  --test-years INTEGER RANGE      Test window length in years.  [x>=1]
  --costos [cripto|etf|etf-conservador]
                                  Trading cost profile.
  --tamano-posicion FLOAT RANGE   Position size as a fraction of equity.
                                  [0.0<x<=1.0]
  --regla-tamano [confianza|fijo|riesgo]
                                  Position sizing rule.
  --riesgo-por-trade FLOAT RANGE  Risk per trade as a fraction of equity.
                                  [x>0.0]
  --orden-minima FLOAT RANGE      Minimum order value.  [x>0.0]
  --help                          Show this message and exit.
```

Options and defaults:

- `--llenado` — default: `'apertura'`
- `--strategy` — default: `None`
- `--symbol` — default: `None`
- `--snapshot` — default: `None`
- `--grid` — default: `None`
- `--out` — default: `None`
- `--from` — default: `None`
- `--to` — default: `None`
- `--train-years` — default: `4`
- `--test-years` — default: `2`
- `--costos` — default: `None`
- `--tamano-posicion` — default: `1.0`
- `--regla-tamano` — default: `None`
- `--riesgo-por-trade` — default: `None`
- `--orden-minima` — default: `None`
