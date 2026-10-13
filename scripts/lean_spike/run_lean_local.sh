#!/usr/bin/env bash
# Launcher compilado desde fuente, sin Docker, CLI ni credenciales.
# Uso: run_lean_local.sh SmaCrossSpike cash /ruta/W
set -euo pipefail
ALGO="$1"; CUENTA="$2"; WORK="$(realpath "$3")"
HERE="$(cd "$(dirname "$0")" && pwd)"
export DOTNET_ROOT="$WORK/dotnet" DOTNET_CLI_HOME="$WORK/dotnet-home"
export NUGET_PACKAGES="$WORK/nuget" TMPDIR="$WORK/tmp"
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_PROCESSOR_COUNT=2
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNET_PYDLL="${PYTHONNET_PYDLL:-$("$WORK/py311/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR")+"/"+sysconfig.get_config_var("LDLIBRARY"))')}"
export PYTHONHOME="$("$WORK/py311/bin/python" -c 'import sys; print(sys.base_prefix)')"
export PYTHONPATH="$WORK/py311/lib/python3.11/site-packages"
BUILD="$WORK/Lean/Launcher/bin/Release"
mkdir -p "$WORK/results/$ALGO"
cp -rn "$WORK/Lean/Data/market-hours" "$WORK/Lean/Data/symbol-properties" "$WORK/data/"
"$WORK/py311/bin/python" - "$HERE" "$WORK" "$ALGO" "$CUENTA" <<'PY'
import json,sys
from pathlib import Path
here,work,algo,cuenta=sys.argv[1:]
c=json.loads(Path(here,'config.json').read_text())
c.update({'algorithm-type-name':algo,'algorithm-id':algo,'algorithm-location':str(Path(here,'lean_algos.py')),'data-folder':work+'/data/','results-destination-folder':work+'/results/'+algo,'python-venv':work+'/py311','python-additional-paths':[work+'/Lean/Algorithm.Python'],'object-store-root':work+'/storage'})
c['parameters']['cuenta']=cuenta
Path(work,'results',algo,'config.json').write_text(json.dumps(c,indent=2)+'\n')
PY
cd "$BUILD"
# strace permite verificar si hubo solicitudes de red durante el backtest.
/usr/bin/time -o "$WORK/results/$ALGO/run.time" -f '%e seconds' \
 strace -f -e trace=network -o "$WORK/results/$ALGO/network.log" \
 "$DOTNET_ROOT/dotnet" QuantConnect.Lean.Launcher.dll \
 --config "$WORK/results/$ALGO/config.json"
