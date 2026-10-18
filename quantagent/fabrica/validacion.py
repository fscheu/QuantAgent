"""Corridas del motor por activo para las etapas de validación."""
import csv
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from quantagent.backtesting.verdict import dsr_contra_intentos, evaluar
from quantagent.cli.sweep import registrar_intento
from quantagent.data.snapshot import snapshot_root


SHARPE_MINIMO_FILTRO = 0.10
SHARPE_MAXIMO_MOTOR_ALARMA = 0.0
PROPORCION_MINIMA_ALARMA = 0.5


def alarma_filtro_motor(resultados_etapa1, filas_etapa2):
    """Compara Sharpe por activo; alarma si diverge al menos la mitad del filtro aprobado."""
    aprobados = [activo for activo, sharpe, _ in resultados_etapa1
                 if sharpe is not None and sharpe >= SHARPE_MINIMO_FILTRO]
    motor = {fila["activo"]: fila["sharpe_tasa_0"] for fila in filas_etapa2}
    activos = [activo for activo in aprobados
               if motor[activo] in (None, "")
               or float(motor[activo]) <= SHARPE_MAXIMO_MOTOR_ALARMA]
    return bool(aprobados) and len(activos) >= PROPORCION_MINIMA_ALARMA * len(aprobados), activos


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
            fila_intento = {**fila, "sharpe": fila["sharpe_tasa_0"]}
            registrar_intento(h.clase, activo, snapshot, "etf", 1.0, params, fila_intento,
                              hipotesis=h.id, etapa=etapa, sharpe_base="tasa_0")
            filas.append(fila)
    return filas


def evaluar_etapa2_cartera(h, filas, carpeta, snapshot=None):
    """Combina retornos de celdas con pesos de cartera y evalúa V30 sumando trades."""
    from quantagent.backtesting.vbt_filter import equity_metrics
    from quantagent.fabrica.cartera import curva_cartera

    cierres = None
    if snapshot is not None:
        from quantagent.data.snapshot import read_snapshot
        cierres = pd.DataFrame({a: read_snapshot(snapshot, a)["close"].loc["2007-01-01":"2018-12-31"]
                                for p in h.piernas for a in p.activos})

    piernas_map = {i: list(p.activos) for i, p in enumerate(h.piernas)}
    retornos_dict = {}
    for i, p in enumerate(h.piernas):
        for activo in p.activos:
            p_eq = Path(carpeta) / "equity" / f"{p.clase}-{activo}.csv"
            if not p_eq.exists():
                p_eq = Path(carpeta) / "equity" / f"{activo}.csv"
            if p_eq.exists():
                with p_eq.open(newline="") as f:
                    curva = list(csv.DictReader(f))
                if len(curva) >= 2:
                    col = "date" if "date" in curva[0] else "timestamp"
                    eq = pd.Series([float(r["equity"]) for r in curva], index=pd.to_datetime([r[col] for r in curva]))
                    retornos_dict[(i, activo)] = eq.pct_change().dropna()

    total_trades = sum(int(f["trades"]) for f in filas if f.get("trades") not in (None, ""))
    eq_cartera = curva_cartera(cierres, piernas_map, retornos_dict)
    if eq_cartera is not None and not eq_cartera.empty:
        m = equity_metrics(eq_cartera, 0.0)
        return evaluar(total_trades, float(eq_cartera.iloc[-1] - 100_000.0), m["sharpe"], m["max_drawdown"])
    return False, ["curva de cartera vacía"]


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
            pasa_c, motivos = evaluar(*(float(fila[c]) if fila[c] != "" else None
                                        for c in ("trades", "pnl", "sharpe_tasa_0", "max_drawdown")))
            fila["v30"] = "PASA" if pasa_c else "NO PASA"
            if not fila["control"].startswith("coinciden"):
                motivos.append(fila["control"])
            fila["motivos"] = [f"{activo}: {motivo}" for motivo in motivos]
    if len(h.piernas) > 1:
        pasa_cartera, motivos_cartera = evaluar_etapa2_cartera(h, filas, carpeta, snapshot)
        pasa = all(f["control"].startswith("coinciden") for f in filas) and pasa_cartera
        if not pasa_cartera:
            for f in filas:
                f["motivos"].extend([f"cartera: {m}" for m in motivos_cartera if f"cartera: {m}" not in f["motivos"]])
    else:
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
