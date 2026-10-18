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


def alarma_cartera(sharpe_filtro, sharpe_motor):
    """Alarma si el filtro de cartera aprueba (>= 0.10) pero el motor no (> 0)."""
    if sharpe_filtro is None or (isinstance(sharpe_filtro, (int, float)) and sharpe_filtro != sharpe_filtro):
        return False
    if float(sharpe_filtro) < SHARPE_MINIMO_FILTRO:
        return False
    if sharpe_motor in (None, ""):
        return True
    return float(sharpe_motor) <= SHARPE_MAXIMO_MOTOR_ALARMA


def correr_motor(h, snapshot, params_por_activo, desde, hasta, carpeta, etapa, abrir_prueba=False):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    filas = []
    celdas = [(p.clase, a, params_por_activo.get((p.clase, a), {}))
              for p in h.piernas for a in p.activos] if len(h.piernas) > 1 else [
              (h.clase, a, p) for a, p in params_por_activo.items()]
    with TemporaryDirectory() as temporal:
        params_path = Path(temporal) / "params.json"
        for strat, activo, params in celdas:
            params_path.write_text(json.dumps({strat: params}), encoding="utf-8")
            out = carpeta / (f"{strat}-{activo}.csv" if len(h.piernas) > 1 else f"{activo}.csv")
            comando = [
                sys.executable, "-m", "quantagent.cli", "backtest", "compare",
                "--strategies", strat, "--symbols", activo, "--snapshot", snapshot,
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
            if len(h.piernas) <= 1:
                fila_intento = {**fila, "sharpe": fila["sharpe_tasa_0"]}
                registrar_intento(strat, activo, snapshot, "etf", 1.0, params, fila_intento,
                                  hipotesis=h.id, etapa=etapa, sharpe_base="tasa_0")
            filas.append(fila)
    return filas


def _curva_y_v30_cartera(h, filas, carpeta, snapshot, etapa, desde, hasta):
    from quantagent.backtesting.vbt_filter import equity_metrics
    from quantagent.fabrica.cartera import curva_cartera

    cierres = None
    if snapshot is not None:
        from quantagent.data.snapshot import read_snapshot
        cierres = pd.DataFrame({a: read_snapshot(snapshot, a)["close"].loc[desde:hasta]
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
        pasa, motivos = evaluar(total_trades, float(eq_cartera.iloc[-1] - 100_000.0), m["sharpe"], m["max_drawdown"])
        sharpe_val = round(m["sharpe"], 2) if m.get("sharpe") is not None else ""
        for f in filas:
            f["sharpe_cartera"] = sharpe_val
        if snapshot is not None:
            estrategia = "+".join(p.clase for p in h.piernas)
            combo = {p.clase: p.parametros for p in h.piernas}
            pnl_val = round(float(eq_cartera.iloc[-1] - 100_000.0), 2)
            registrar_intento(
                estrategia, "cartera", snapshot, "etf", 1.0, combo,
                {"desde": desde, "hasta": hasta, "trades": total_trades, "pnl": pnl_val, "sharpe": sharpe_val},
                hipotesis=h.id, etapa=etapa, sharpe_base="tasa_0",
            )
        return eq_cartera, pasa, motivos
    return None, False, ["curva de cartera vacía"]


def evaluar_etapa2_cartera(h, filas, carpeta, snapshot=None):
    """Combina retornos de celdas con pesos de cartera y evalúa V30 sumando trades."""
    _, pasa, motivos = _curva_y_v30_cartera(h, filas, carpeta, snapshot, "2", "2007-01-01", "2018-12-31")
    return pasa, motivos


def validar_etapa2(h, snapshot, params_por_activo, carpeta):
    """Control independiente y V30; los motivos de corte quedan en las filas por activo."""
    filas = correr_motor(h, snapshot, params_por_activo, "2007-01-01", "2018-12-31", carpeta, "2")
    with TemporaryDirectory() as temporal:
        params_path = Path(temporal) / "params.json"
        for fila in filas:
            activo = fila["activo"]
            strat = fila.get("estrategia", h.clase)
            params = params_por_activo.get((strat, activo), {}) if len(h.piernas) > 1 else params_por_activo.get(activo, {})
            params_path.write_text(json.dumps({strat: params}), encoding="utf-8")
            tabla = Path(carpeta) / (f"{strat}-{activo}.csv" if (Path(carpeta) / f"{strat}-{activo}.csv").exists() else f"{activo}.csv")
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
    if len(h.piernas) > 1:
        eq_cartera, pasa_v30, motivos = _curva_y_v30_cartera(
            h, filas, carpeta, snapshot, "3", "2019-01-01", "2022-12-31"
        )
        dsr = dsr_contra_intentos(eq_cartera, snapshot_root() / "intentos.csv") if eq_cartera is not None else None
        pasa_dsr = dsr is not None and dsr >= 0.95
        if not pasa_dsr:
            motivos.append("DSR de cartera < 0.95" if dsr is not None else "sin DSR")
        for fila in filas:
            fila["v30"] = "PASA" if pasa_v30 else "NO PASA"
            fila["dsr"] = dsr
            fila["motivos"] = list(motivos)
        pasa = pasa_v30 and pasa_dsr
        return "PASA" if pasa else "CORTA", filas
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
