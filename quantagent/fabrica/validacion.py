"""Corridas del motor por activo para las etapas de validación."""
import csv
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from quantagent.cli.sweep import registrar_intento


def correr_motor(h, snapshot, params_por_activo, desde, hasta, carpeta, etapa, abrir_prueba=False):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    filas = []
    with TemporaryDirectory() as temporal:
        params_path = Path(temporal) / "params.json"
        for activo, params in params_por_activo.items():
            params_path.write_text(json.dumps({h.clase: params}), encoding="utf-8")
            out = carpeta / f"{activo}.csv"
            comando = [
                sys.executable, "-m", "quantagent.cli", "backtest", "compare",
                "--strategies", h.clase, "--symbols", activo, "--snapshot", snapshot,
                "--from", str(desde), "--to", str(hasta), "--costos", "etf",
                "--llenado", "apertura", "--params", str(params_path), "--out", str(out),
                "--trades-dir", str(carpeta / "trades"), "--equity-dir", str(carpeta / "equity"),
            ]
            if abrir_prueba:
                comando.append("--abrir-prueba")
            resultado = subprocess.run(comando, capture_output=True, text=True)
            if resultado.returncode:
                raise RuntimeError(resultado.stderr)
            with out.open(newline="") as f:
                fila = next(csv.DictReader(f))
            registrar_intento(h.clase, activo, snapshot, "etf", 1.0, params, fila,
                              hipotesis=h.id, etapa=etapa)
            filas.append(fila)
    return filas
