"""Corridas del motor por activo para las etapas de validación."""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from quantagent.backtesting.verdict import dsr_contra_intentos, evaluar, n_efectivo
from quantagent.cli.sweep import registrar_intento
from quantagent.data.snapshot import snapshot_root
from quantagent.fabrica.criterio import anios_positivos, evaluar_etapa3_cartera, recargo_paper, sharpe_benchmark
from quantagent.fabrica.transversal import METRICAS, es_transversal, leer_seleccion, señales_transversales
from quantagent.strategy.registry import STRATEGY_REGISTRY, cli_aliases

SHARPE_MINIMO_FILTRO = 0.10
SHARPE_MAXIMO_MOTOR_ALARMA = 0.0
PROPORCION_MINIMA_ALARMA = 0.5
COLUMNAS_CELDAS = ["hipotesis", "pierna_clase", "estrategia", "activo", "desde", "hasta",
                   "trades", "pnl", "max_drawdown", "max_drawdown_cartera", "benchmark_mismo_peso"]


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
    paneles_trans = {}
    import quantagent.data.snapshot as snap_mod
    estrategia_objetivo = next((p for p in h.piernas
                                if (STRATEGY_REGISTRY.get(cli_aliases().get(p.clase, p.clase))
                                    and hasattr(STRATEGY_REGISTRY[cli_aliases().get(p.clase, p.clase)]["cls"],
                                                "posiciones_objetivo"))), None)
    if etapa == "2" and estrategia_objetivo:
        from quantagent.backtesting.vbt_filter import equity_metrics

        activos = list(estrategia_objetivo.activos)
        panel = pd.DataFrame({a: snap_mod.read_snapshot(snapshot, a)["close"].loc[desde:hasta]
                              for a in activos})
        estrategia = STRATEGY_REGISTRY[cli_aliases().get(estrategia_objetivo.clase,
                                                         estrategia_objetivo.clase)]["cls"]
        objetivos = estrategia.posiciones_objetivo(panel)
        for activo in activos:
            pesos = objetivos[activo].reindex(panel.index).fillna(0.0)
            retornos = panel[activo].pct_change().fillna(0.0) * pesos.shift(1, fill_value=0.0)
            equity = 100_000.0 * (1 + retornos).cumprod()
            trades = int(pesos.diff().fillna(0.0).ne(0).sum())
            metricas = equity_metrics(equity, 0.0)
            nombre = f"{estrategia_objetivo.clase}-{activo}.csv" if h.es_cartera else f"{activo}.csv"
            (carpeta / "equity").mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"date": equity.index, "equity": equity.values}).to_csv(
                carpeta / "equity" / nombre, index=False)
            filas.append({"activo": activo, "estrategia": estrategia_objetivo.clase,
                          "trades": str(trades), "pnl": str(round(float(equity.iloc[-1] - 100_000), 2)),
                          "sharpe_tasa_0": str(round(metricas["sharpe"], 2)),
                          "max_drawdown": str(round(metricas["max_drawdown"], 6))})
        return filas
    for p in h.piernas:
        if es_transversal(p.clase) and p.clase not in paneles_trans:
            activos = list(p.activos) or [a for c, a in params_por_activo.keys() if c == p.clase] or list(params_por_activo.keys())
            cierres = pd.DataFrame({a: snap_mod.read_snapshot(snapshot, a)["close"].loc[:hasta] for a in activos})
            paneles_trans[p.clase] = señales_transversales(cierres, METRICAS[p.clase])
            for a in activos:
                pd.DataFrame({"timestamp": cierres.index, "posicion": paneles_trans[p.clase][a].values}).to_csv(
                    carpeta / f"senal-{p.clase}-{a}.csv", index=False)
    celdas = [(p.clase, a, params_por_activo.get((p.clase, a), {}))
              for p in h.piernas for a in p.activos] if h.es_cartera else [
              (h.clase, a, p) for a, p in params_por_activo.items()]
    with TemporaryDirectory() as temporal:
        params_path = Path(temporal) / "params.json"
        for strat, activo, params in celdas:
            es_trans = es_transversal(strat)
            csv_path = carpeta / f"senal-{strat}-{activo}.csv"
            strat_cli = "senal-externa" if es_trans else strat
            cli_params = {"ruta": str(csv_path)} if es_trans else params
            params_path.write_text(json.dumps({strat_cli: cli_params}), encoding="utf-8")
            nombre = f"{strat}-{activo}.csv" if h.es_cartera else f"{activo}.csv"
            out = carpeta / nombre
            eq_out = carpeta / "equity" / nombre
            if not (out.exists() and eq_out.exists()):
                comando = [
                    sys.executable, "-m", "quantagent.cli", "backtest", "compare",
                    "--strategies", strat_cli, "--symbols", activo, "--snapshot", snapshot,
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
            if es_trans:
                for sub in ("equity", "trades"):
                    src = carpeta / sub / f"senal-externa-{activo}.csv"
                    if src.exists():
                        shutil.copy(src, carpeta / sub / f"{strat}-{activo}.csv")
                fila["pierna_clase"] = strat
                fila["ruta_senal"] = str(csv_path)
            if not h.es_cartera and not es_trans:
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
        import quantagent.data.snapshot as snap_mod
        cierres = pd.DataFrame({a: snap_mod.read_snapshot(snapshot, a)["close"].loc[desde:hasta]
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

    seleccion = leer_seleccion(carpeta, h.piernas, cierres.index if cierres is not None else None)
    if seleccion and cierres is None:
        raise ValueError("Faltan cierres para ponderar la selección transversal")
    total_trades = sum(int(f["trades"]) for f in filas if f.get("trades") not in (None, ""))
    eq_cartera = curva_cartera(cierres, piernas_map, retornos_dict, seleccion=seleccion)
    if eq_cartera is not None and not eq_cartera.empty:
        m = equity_metrics(eq_cartera, 0.0)
        pasa, motivos = evaluar(total_trades, float(eq_cartera.iloc[-1] - 100_000.0), m["sharpe"], m["max_drawdown"])
        sharpe_val = round(m["sharpe"], 2) if m.get("sharpe") is not None else ""
        for f in filas:
            f["sharpe_cartera"] = sharpe_val
            f["max_drawdown_cartera"] = m["max_drawdown"]
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
            clase = fila.get("pierna_clase", strat)
            es_trans = es_transversal(clase)
            trades = int(float(fila["trades"])) if fila.get("trades") not in (None, "") else 0
            if es_trans and trades == 0:
                fila["control"] = "coinciden 0 de 0"
                fila["v30"] = "PASA"
                fila["motivos"] = []
                continue
            if es_trans:
                params = {"ruta": fila["ruta_senal"]}
                src_tr = Path(carpeta) / "trades" / f"{clase}-{activo}.csv"
                if src_tr.exists():
                    shutil.copy(src_tr, Path(carpeta) / "trades" / f"senal-externa-{activo}.csv")
            else:
                params = params_por_activo.get((strat, activo), {}) if h.es_cartera else params_por_activo.get(activo, {})
            estrategia = STRATEGY_REGISTRY.get(cli_aliases().get(strat, strat))
            if estrategia and hasattr(estrategia["cls"], "posiciones_objetivo"):
                fila["control"] = f"no aplica: posiciones objetivo; {trades} órdenes del motor"
                fila["v30"] = "PASA" if evaluar(*(float(fila[c]) if fila[c] != "" else None
                                                    for c in ("trades", "pnl", "sharpe_tasa_0", "max_drawdown")))[0] else "NO PASA"
                fila["motivos"] = [] if fila["v30"] == "PASA" else [f"{activo}: métricas V30 no pasan"]
                continue
            params_path.write_text(json.dumps({strat: params}), encoding="utf-8")
            tabla = Path(carpeta) / (f"{clase}-{activo}.csv" if (Path(carpeta) / f"{clase}-{activo}.csv").exists() else f"{activo}.csv")
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
            if es_trans:
                fila["v30"] = "PASA"
                fila["motivos"] = [] if fila["control"].startswith("coinciden") else [f"{activo}: {fila['control']}"]
            else:
                pasa_c, motivos = evaluar(*(float(fila[c]) if fila[c] != "" else None
                                            for c in ("trades", "pnl", "sharpe_tasa_0", "max_drawdown")))
                fila["v30"] = "PASA" if pasa_c else "NO PASA"
                if not fila["control"].startswith("coinciden"):
                    motivos.append(fila["control"])
                fila["motivos"] = [f"{activo}: {motivo}" for motivo in motivos]
    if h.es_cartera:
        pasa_cartera, motivos_cartera = evaluar_etapa2_cartera(h, filas, carpeta, snapshot)
        pasa = all(f["control"].startswith(("coinciden", "no aplica:")) for f in filas) and pasa_cartera
        if not pasa_cartera:
            for f in filas:
                f["motivos"].extend([f"cartera: {m}" for m in motivos_cartera if f"cartera: {m}" not in f["motivos"]])
    else:
        pasa = all(f["control"].startswith(("coinciden", "no aplica:")) for f in filas) and any(f["v30"] == "PASA" for f in filas)
    with (Path(carpeta) / "celdas.csv").open("w", newline="") as f:
        escritor = csv.DictWriter(f, COLUMNAS_CELDAS, extrasaction="ignore")
        escritor.writeheader()
        for fila in filas:
            estrategia = fila.get("estrategia", h.clase)
            escritor.writerow({**fila, "hipotesis": h.id, "estrategia": estrategia,
                               "pierna_clase": fila.get("pierna_clase", estrategia),
                               "desde": "2007-01-01", "hasta": "2018-12-31"})
    return "PASA" if pasa else "CORTA", filas


def validar_etapa3(h, snapshot, params_por_activo, carpeta):
    """Valida la prueba; el llamador debe abrir el candado antes de invocar."""
    filas = correr_motor(h, snapshot, params_por_activo, "2019-01-01", "2022-12-31",
                         carpeta, "3", abrir_prueba=True)
    if h.es_cartera:
        eq_cartera, _, _ = _curva_y_v30_cartera(
            h, filas, carpeta, snapshot, "3", "2019-01-01", "2022-12-31"
        )
        import quantagent.data.snapshot as snap_mod
        activos = list(dict.fromkeys(a for p in h.piernas for a in p.activos))
        c3 = pd.DataFrame({a: snap_mod.read_snapshot(snapshot, a)["close"].loc["2019-01-01":"2022-12-31"] for a in activos}) if snapshot else None
        c2 = pd.DataFrame({a: snap_mod.read_snapshot(snapshot, a)["close"].loc["2007-01-01":"2018-12-31"] for a in activos}) if snapshot else None
        sb3 = sharpe_benchmark(c3) if c3 is not None and not c3.empty else 0.0
        sb2 = sharpe_benchmark(c2) if c2 is not None and not c2.empty else None
        from quantagent.fabrica.cartera import pesos as pesos_cartera
        piernas = {i: p.activos for i, p in enumerate(h.piernas)}
        seleccion = leer_seleccion(carpeta, h.piernas, c3.index if c3 is not None else None)
        w = pesos_cartera(c3, piernas, seleccion=seleccion).T.groupby(level=1).sum().T if c3 is not None and not c3.empty else None
        sb3_mismo_peso = sharpe_benchmark(c3, pesos=w) if w is not None else None

        intentos_csv = snapshot_root() / "intentos.csv"
        todas_filas, s2 = [], None
        if intentos_csv.exists():
            with open(intentos_csv, newline="", encoding="utf-8") as f:
                todas_filas = list(csv.DictReader(f))
            for r in todas_filas:
                if r.get("hipotesis") == h.id and r.get("etapa") == "2" and r.get("activo") == "cartera":
                    try:
                        s2 = float(r["sharpe"])
                    except (ValueError, TypeError):
                        pass

        tot_trades = sum(int(f["trades"]) for f in filas if f.get("trades") not in (None, ""))
        pasa, motivos = evaluar_etapa3_cartera(eq_cartera, sb3, s2, sb2, trades=tot_trades)

        neff = n_efectivo(todas_filas) if todas_filas else None
        try:
            dsr_eff = dsr_contra_intentos(eq_cartera, intentos_csv, neff) if eq_cartera is not None else None
            dsr_crudo = dsr_contra_intentos(eq_cartera, intentos_csv) if eq_cartera is not None else None
        except FileNotFoundError:
            dsr_eff, dsr_crudo = None, None

        ap = anios_positivos(eq_cartera) if eq_cartera is not None and not eq_cartera.empty else None
        recargo = recargo_paper(dsr_eff)
        for fila in filas:
            fila.update(v30="PASA" if pasa else "NO PASA", dsr=dsr_eff, dsr_crudo=dsr_crudo, recargo=recargo,
                        anios_positivos=f"{ap.positivos} de {ap.totales}" if ap else None,
                        benchmark=sb3, benchmark_mismo_peso=sb3_mismo_peso,
                        n_efectivo=neff, motivos=list(motivos))
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
