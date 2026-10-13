# Spike de LEAN (QuantAgent-ryaw)

Segunda corrida, 2026-10-13: Python inicializa y ambos backtests terminan hasta 2018-12-31.
Contra next-open: SMA 3/5, RSI 140/187 (176 trades LEAN). Las diferencias están clasificadas;
SMA no cumple el criterio 5/5. Los JSON tienen Status Completed, pero el Launcher aborta
con salida 134 durante PythonInitializer.Shutdown, después de guardar resultados.

Resultados, causas y limitaciones: [reporte](../../docs/04_decisions/QuantAgent-ryaw-DC-spike-lean.md).
Contexto: [decisión, §§7–10](../../docs/04_decisions/2026-10-12-DC-quantconnect-y-pivot-lean.md).

| Archivo | Estado validado |
|---|---|
| write_lean_data.py | 3020 velas; LEAN aceptó precios escalados decimales, factor 1 y map sin cambios |
| lean_algos.py | SMA cash / RSI margen leverage 1; brokerage y fill model predeterminados; selección explícita del módulo |
| config.json | Handlers locales; runner sustituye rutas y parámetros; sin cambios en segunda corrida |
| run_lean_local.sh | Launcher Release + tiempo + strace; backtests Completed, aborto de cierre visible |
| run_lean.sh | Runner Docker histórico; no usado |
| closed_trades_to_csv.py | Validado con symbols, direction, orderIds y sesiones NY; rechaza RuntimeError |
| compare_trades.py | Cuatro comparaciones LEAN/control ejecutadas; sin cambios |

Se tocaron lean_algos.py, run_lean_local.sh, closed_trades_to_csv.py, write_lean_data.py,
este README y el reporte. Todo nuevo resultado y diagnóstico queda en W. No hubo commits;
el PM debe actualizar Beads e integrar tras revisar.

## Entorno y controles: comandos que funcionaron

```bash
export W=/home/azureuser/spike-lean-work
export DOTNET_ROOT=$W/dotnet DOTNET_CLI_HOME=$W/dotnet-home NUGET_PACKAGES=$W/nuget
export UV_CACHE_DIR=$W/uv-cache PIP_CACHE_DIR=$W/pip-cache TMPDIR=$W/tmp
export PATH="$DOTNET_ROOT:$PATH"
mkdir -p "$W/tmp" "$W/control"
source /home/azureuser/repos/projects/QuantAgent/.venv/bin/activate
export PYTHONPATH=/home/azureuser/repos/projects/QuantAgent-spike-lean
export PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=$W/matplotlib
export QUANTAGENT_SNAPSHOT_DIR=/home/azureuser/repos/projects/QuantAgent/local-data
export DATABASE_URL=sqlite:///$W/try.db
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 DOTNET_PROCESSOR_COUNT=2
python -c 'import quantagent; print(quantagent.__file__); from quantagent.database import init_db; init_db()'

for strategy in sma-cross rsi; do
  TRADING_BASE_POSITION_PCT=1 python scripts/bt_adapter.py --strategy "$strategy" \
    --snapshot etf-1d-2026-10 --symbol SPY --from 2007-01-03 --to 2018-12-31 \
    --fill next-open --costos etf --out "$W/control/adapter_$strategy.csv"
  python -m quantagent.cli backtest run --strategy "$strategy" \
    --snapshot etf-1d-2026-10 --symbol SPY --from 2007-01-03 --to 2018-12-31 \
    --costos etf --tamano-posicion 1.0 --out "$W/control/motor_$strategy.csv"
done
```

Resultado: 5 trades de SMA y 187 de RSI en cada control. Ambos controles tienen fechas distintas por su supuesto
de llenado; no se deben intercambiar. Perfil etf: slippage 0.0005 por lado, comisión 0; 100% del capital.

## Fuente y SDK: comandos que funcionaron

Se descargó un archivo de fuente en lugar de un clon superficial para respetar la prohibición de usar git.
La revisión usada fue `80e7843f645673bcbeaab963049f76f20f6785e1`. No hay global.json; el csproj pide net10.0.
Antes y después de cada descarga/build se ejecutó `df -h /`; detenerse si quedan menos de 30G.

```bash
df -h /
curl -fL https://codeload.github.com/QuantConnect/Lean/tar.gz/80e7843f645673bcbeaab963049f76f20f6785e1 \
  -o "$W/lean-source.tar.gz"
mkdir -p "$W/Lean"
tar -xzf "$W/lean-source.tar.gz" --strip-components=1 -C "$W/Lean"
df -h /
curl -fL https://dot.net/v1/dotnet-install.sh -o "$W/dotnet-install.sh"
/usr/bin/time -f '%e seconds' bash "$W/dotnet-install.sh" --channel 10.0 --install-dir "$DOTNET_ROOT"
df -h /
/usr/bin/time -f '%e seconds' dotnet build "$W/Lean/Launcher/QuantConnect.Lean.Launcher.csproj" \
  -c Release -m:2 -p:BuildInParallel=true
df -h /
```

SDK instalado: 10.0.401. Build: 0 errores, 4446 advertencias upstream. Logs completos en W/build-1.log.

## Python y datos: estado reutilizado y arreglo validado

No repetir instalación, build, controles ni exportación en este entorno. Los comandos anteriores
son el registro de la primera corrida. En W/py311 ya están Python 3.11.14, pandas 2.3.3 y numpy 1.26.4.
No instalar el paquete pip pythonnet: eclipsa el clr embebido de QuantConnect.pythonnet (Python.Runtime 2.0.66).
El arreglo que destrabó PandasConverter fue:

```bash
export W=/home/azureuser/spike-lean-work
export UV_CACHE_DIR=$W/uv-cache TMPDIR=$W/tmp
df -h /
uv pip uninstall --python "$W/py311/bin/python" pythonnet clr-loader
```

Datos ya exportados en W/data: daily/spy.zip, factor_files/spy.csv y map_files/spy.csv bajo equity/usa.
OHLC * 10000 con cuatro decimales aceptados; no se usó --enteros. Los archivos market-hours y
symbol-properties vienen de la fuente local. Nada de precios después de 2018-12-31.

## Corridas completas y comparaciones: comandos validados

El runner configura PYTHONHOME, PYTHONNET_PYDLL, PYTHONPATH y LEAN_SPIKE_ALGORITHM.
Esta última variable deja una sola clase QCAlgorithm visible al cargador Python. No usa un fill model propio.
Requiere strace y /usr/bin/time ya presentes. Ejecutar secuencialmente y esperar cada salida.
Detenerse si / tiene menos de 30G libres.

```bash
export W=/home/azureuser/spike-lean-work
df -h /
scripts/lean_spike/run_lean_local.sh SmaCrossSpike cash "$W" > "$W/sma-final.log" 2>&1
# Salida 134 en Shutdown; verificar Completed y RuntimeError vacío antes de convertir.
scripts/lean_spike/run_lean_local.sh RsiSpike margin "$W" > "$W/rsi-final.log" 2>&1
# Mismo aborto posterior a guardar Completed.

export PYTHONPATH=/home/azureuser/repos/projects/QuantAgent-spike-lean
export PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=$W/matplotlib
export DATABASE_URL=sqlite:///$W/try.db
PY=/home/azureuser/repos/projects/QuantAgent/.venv/bin/python
"$PY" scripts/lean_spike/closed_trades_to_csv.py "$W/results/SmaCrossSpike/SmaCrossSpike.json" --out "$W/lean_sma-cross.csv"
"$PY" scripts/lean_spike/closed_trades_to_csv.py "$W/results/RsiSpike/RsiSpike.json" --out "$W/lean_rsi.csv"
for strategy in sma-cross rsi; do
  for control in adapter motor; do
    "$PY" scripts/lean_spike/compare_trades.py "$W/control/${control}_$strategy.csv" "$W/lean_$strategy.csv"       > "$W/compare_${control}_$strategy.log"
    # Salida 1 esperada: hay diferencias reales.
  done
done
"$W/py311/bin/python" "$W/classify_differences.py"
```

Resultados: SMA 3/5 next-open, 0/5 motor; RSI 140/187 next-open, 12/187 motor.
Tiempos de pared instrumentados: SMA 17.74 s, RSI 24.03 s, incluido el aborto de cierre.
El parser usa symbols[0] y orderIds para tags; convierte UTC a sesión NY. No depende de lastFillTime.
Clasificación por trade del control: W/differences_{sma-cross,rsi}.csv. Las órdenes/eventos y trazas
completas están en W/results/<algoritmo>/. La clasificación es una auditoría de estas corridas.

Verificación de red y estado, sin publicar precios ni trades:

```bash
"$W/py311/bin/python" - <<'CHECK'
import json, os, re
from pathlib import Path
w = Path(os.environ['W'])
for algo in ('SmaCrossSpike', 'RsiSpike'):
    folder = w / 'results' / algo
    state = json.loads((folder / f'{algo}.json').read_text())['state']
    assert state['Status'] == 'Completed' and not state['RuntimeError'], state
    calls = len(re.findall(r'AF_INET6?\b', (folder / 'network.log').read_text()))
    print(algo, state['Status'], 'llamadas IP:', calls)
    assert calls == 0
CHECK
bash -n scripts/lean_spike/run_lean_local.sh
df -h /
```

Las dos trazas finales tienen cero llamadas AF_INET/AF_INET6 desde el arranque hasta la terminación.
RSI registra una lectura local fallida de alternative/interest-rate/usa/interest-rate.csv; no se descargó
ese archivo ni se verificaron intereses. El cierre limpio de Python sigue pendiente. No interpretar
el aborto de Shutdown como RuntimeError del algoritmo, ni un JSON de un arranque fallido como cero trades válidos.
