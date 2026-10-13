#!/usr/bin/env bash
# QuantAgent-ryaw (spike): corre un algoritmo con el Launcher de LEAN, sin CLI ni login y sin red.
# NO EJECUTADO todavia (falta Docker en la notebook).
#
# Uso:  scripts/lean_spike/run_lean.sh <SmaCrossSpike|RsiSpike> <cash|margin> <carpeta-de-trabajo>
#
# <carpeta-de-trabajo> va fuera del repo y tiene data/ (salida de write_lean_data.py). Deja el JSON de
# resultados en <carpeta-de-trabajo>/results/<algoritmo>/. Las bases de horarios y de propiedades de
# simbolos se copian de la imagen: son de LEAN (Apache 2.0), no datos de mercado.
set -euo pipefail

ALGO="$1"; CUENTA="$2"; WORK="$(realpath "$3")"
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${LEAN_IMAGE:-quantconnect/lean:latest}"

mkdir -p "$WORK/results/$ALGO"
time docker run --rm --network none \
  -v "$HERE:/spike/algos:ro" \
  -v "$WORK/data:/spike/data" \
  -v "$WORK/results/$ALGO:/spike/results" \
  --entrypoint bash "$IMAGE" -c "
    cp -rn /Lean/Data/market-hours /Lean/Data/symbol-properties /spike/data/ &&
    cd /Lean/Launcher/bin/Debug &&
    dotnet QuantConnect.Lean.Launcher.dll --config /spike/algos/config.json \
      --algorithm-type-name $ALGO --parameters cuenta:$CUENTA --close-automatically true"
