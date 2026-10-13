# Spike de LEAN (QuantAgent-ryaw)

Estado al 2026-10-13: **resultado negativo por bloqueo de Python.NET**. LEAN se compiló desde fuente sin Docker,
CLI, cuenta ni login. Los tres arranques fallaron al inicializar PandasConverter y al cerrar el GIL; no se
completaron backtests. Se detuvo el trabajo al tercer intento. No hay resultados de trades de LEAN válidos.

Evidencia, mediciones y criterios: [reporte del spike](../../docs/04_decisions/QuantAgent-ryaw-DC-spike-lean.md).
Contexto: [decisión, §§7–10](../../docs/04_decisions/2026-10-12-DC-quantconnect-y-pivot-lean.md).

| Archivo | Estado real |
|---|---|
| write_lean_data.py | Exportó 3020 velas; lectura por LEAN pendiente |
| lean_algos.py | Port SMA/RSI; MarketOnOpen y fill model por defecto; inicialización bloqueada |
| config.json | Handlers locales; el runner sustituye rutas y parámetros |
| run_lean_local.sh | Ejecuta Launcher Release y registra tiempo/red; reproduce el fallo, no completa backtests |
| run_lean.sh | Runner Docker anterior, no usado ni validado en esta ejecución |
| closed_trades_to_csv.py | Rechaza state.RuntimeError; conversión de trades reales pendiente |
| compare_trades.py | Probado entre los controles; paridad con LEAN pendiente |

Todo lo generado está en W = `/home/azureuser/spike-lean-work`. Se modificaron README.md, lean_algos.py,
config.json, closed_trades_to_csv.py y se agregó run_lean_local.sh, además del reporte enlazado.
No se usó git ni se modificaron motor, CLI o tests. Beads no pudo abrir su base; el PM debe actualizar el issue.

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

## Python y datos: preparación que funcionó

```bash
df -h /
uv venv --python /home/azureuser/.local/bin/python3.11 "$W/py311"
uv pip install --python "$W/py311/bin/python" pandas==2.3.3 numpy==1.26.4 pythonnet==3.2.1
df -h /
python scripts/lean_spike/write_lean_data.py --out "$W/data"
cp -r "$W/Lean/Data/market-hours" "$W/Lean/Data/symbol-properties" "$W/data/"
```

El entorno quedó con Python 3.11.14. numpy se fijó a 1.26.4 en el tercer intento, siguiendo
DockerfileLeanFoundation. LEAN carga Python.Runtime 2.0.66 desde NuGet. Que los paquetes Python importen
fuera de LEAN no verifica su inicialización embebida.

## Reproducción del fallo: no es un backtest validado

El runner establece PYTHONHOME, PYTHONNET_PYDLL y el entorno virtual; usa los handlers locales y token vacío.
Requiere el `strace` y `/usr/bin/time` ya disponibles en esta VM, sin instalar herramientas de sistema.

```bash
scripts/lean_spike/run_lean_local.sh SmaCrossSpike cash "$W"
# Falla con salida 134: PandasConverter y GIL. No volver a intentar en esta sesión.
```

No se ejecutó RsiSpike porque usa el mismo inicializador bloqueado. Las trazas `network.log` de los tres
arranques no contienen llamadas AF_INET/AF_INET6. Falta verificar la red durante una corrida completa.

El JSON de SMA tiene state.Status RuntimeError y closedTrades vacío. El parser ahora lo rechaza con salida 1:

```bash
python scripts/lean_spike/closed_trades_to_csv.py \
  "$W/results/SmaCrossSpike/SmaCrossSpike.json" --out "$W/lean_sma-cross.csv"
```

No exportar ni comparar ese resultado como si fuera un backtest de cero trades. Para revisar los controles:

```bash
python scripts/lean_spike/compare_trades.py "$W/control/adapter_sma-cross.csv" "$W/control/motor_sma-cross.csv"
python scripts/lean_spike/compare_trades.py "$W/control/adapter_rsi.csv" "$W/control/motor_rsi.csv"
# Salida 1 esperada: tienen fechas distintas por cierre versus apertura siguiente.
```

Las comparaciones LEAN/control, PnL de LEAN y clasificación de diferencias de RSI quedan sin verificar.
No se publican precios ni filas de trades en documentación; CSV y logs quedan en W.
