"""Corridas del motor por activo para las etapas de validación."""
import csv
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from quantagent.backtesting.verdict import dsr_contra_intentos, evaluar
from quantagent.cli.sweep import registrar_intento
from quantagent.data.snapshot import snapshot_root


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


def validar_etapa2(h, snapshot, params_por_activo, carpeta):
    """Control independiente y V30; los motivos de corte quedan en las filas por activo."""
    filas = correr_motor(h, snapshot, params_por_activo, "2007-01-01", "2018-12-31", carpeta, "2")
    with TemporaryDirectory() as temporal:
        params_path = Path(temporal) / "params.json"
        for fila in filas:
            activo = fila["activo"]
            params_path.write_text(json.dumps({h.clase: params_por_activo[activo]}), encoding="utf-8")
            tabla = Path(carpeta) / f"{activo}.csv"
            resultado = subprocess.run([
                sys.executable, str(Path(__file__).resolve().parents[2] / "scripts" / "compare_control.py"),
                "--tabla", str(tabla),
                "--trades-dir", str(Path(carpeta) / "trades"), "--snapshot", snapshot,
                "--costos", "etf", "--llenado", "apertura", "--params", str(params_path),
            ], capture_output=True, text=True)
            if resultado.returncode:
                raise RuntimeError(resultado.stderr)
            with tabla.open(newline="") as f:
                fila["control"] = next(csv.DictReader(f))["control"]
            pasa, motivos = evaluar(*(float(fila[c]) if fila[c] != "" else None
                                      for c in ("trades", "pnl", "sharpe_tasa_0", "max_drawdown")))
            fila["v30"] = "PASA" if pasa else "NO PASA"
            if not fila["control"].startswith("coinciden"):
                motivos.append(fila["control"])
            fila["motivos"] = [f"{activo}: {motivo}" for motivo in motivos]
    pasa = all(f["control"].startswith("coinciden") for f in filas) and any(f["v30"] == "PASA" for f in filas)
    return "PASA" if pasa else "CORTA", filas


def validar_etapa3(h, snapshot, params_por_activo, carpeta):
    """Valida la prueba; el llamador debe abrir el candado antes de invocar."""
    filas = correr_motor(h, snapshot, params_por_activo, "2019-01-01", "2022-12-31",
                         carpeta, "3", abrir_prueba=True)
    for fila in filas:
        activo = fila["activo"]
        with (Path(carpeta) / "equity" / f"{h.clase}-{activo}.csv").open(newline="") as f:
            equity = [float(r["equity"]) for r in csv.DictReader(f)]
        pasa, motivos = evaluar(*(float(fila[c]) if fila[c] != "" else None
                                  for c in ("trades", "pnl", "sharpe_tasa_0", "max_drawdown")))
        fila["v30"] = "PASA" if pasa else "NO PASA"
        fila["dsr"] = dsr_contra_intentos(equity, snapshot_root() / "intentos.csv")
        if fila["dsr"] is None or not fila["dsr"] >= 0.95:
            motivos.append("sin DSR" if fila["dsr"] is None else "DSR < 0.95")
        fila["motivos"] = [f"{activo}: {motivo}" for motivo in motivos]
    return "PASA" if any(not f["motivos"] for f in filas) else "CORTA", filas
