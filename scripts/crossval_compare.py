#!/usr/bin/env python
"""QuantAgent-832: compara trade por trade el motor del proyecto contra el port a backtesting.py.

Uso:  python scripts/crossval_compare.py [--intrabar]

Corre ambos sobre tests/fixtures/spy-90d.csv sin slippage ni comision: el CLI del motor como subproceso
(TRADING_SLIPPAGE_PCT=0, SQLite nueva y vacia, --out y --equity-out) y el port (scripts/crossval_rsi.py,
importado). Con --intrabar corre ambos con stops evaluados intravela (--intrabar-stops en el motor).
Sale con 0 solo si todo cae dentro de tolerancia; un "NO COMPARADO" tambien da distinto de 0.

Tolerancias (el motor guarda Numeric(18,8): redondeo de hasta 5e-9 por valor):
- entry_time, exit_time, symbol, side: exactos. Nº de trades y win rate: exactos. Profit factor: 1e-6 relativo.
- qty 1e-8: cota real 5e-9 (redondeo del motor) + 2**-28 ~ 3.7e-9 (el port cuantiza a 2**-27) = 8.7e-9.
- entry_price, exit_price 1e-8 (el fixture tiene 2 decimales; solo cuenta el redondeo a 8).
- pnl: 1e-8 * |exit - entry| + 1e-8 por trade (error de qty por el movimiento + redondeo); total PnL: su suma.
- max drawdown 1e-6; equity punto a punto 1e-3 USD (el CSV de equity del motor tiene 4 decimales).

Max drawdown: cada lado con su PROPIA curva (motor: --equity-out; port: stats._equity_curve de
backtesting.py) y la misma formula (scripts/recalc_metrics.py::recalc_equity: maximo corrido,
(pico - equity) / pico). backtesting.py suma un instante extra (la vela centinela del port, que se
descarta); se verifica que los 2160 instantes restantes sean los mismos antes de comparar.
"""

from __future__ import annotations

import argparse
import collections
import csv
import functools
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import crossval_rsi as port  # noqa: E402  (sale con un mensaje si falta backtesting.py)
import recalc_metrics  # noqa: E402

QTY_TOL, PRICE_TOL, PNL_TOL = 1e-8, 1e-8, 1e-8
PF_RTOL, DD_TOL, EQUITY_TOL = 1e-6, 1e-6, 1e-3
NUMERIC = ("qty", "entry_price", "exit_price", "pnl")

# Mide, sin tocar el motor, los rechazos del RiskManager (los rechazos no se persisten ni se loguean en el
# CLI): envuelve validate_trade / get_daily_pnl / on_trade_executed y corre el mismo CLI.
PROBE = r"""
import json, os, sys
from datetime import date
from quantagent.trading.risk_manager import RiskManager as R
st = {"validate_calls": 0, "rejections": {}, "min_daily_loss_margin": None, "breaker_active": 0,
      "today": str(date.today())}
_v, _d, _t = R.validate_trade, R.get_daily_pnl, R.on_trade_executed
def v(self, *a, **k):
    st["validate_calls"] += 1
    ok, why = _v(self, *a, **k)
    if not ok:
        key = why.split(":")[0]
        st["rejections"][key] = st["rejections"].get(key, 0) + 1
    return ok, why
def d(self):
    x = _d(self)
    m = x + self.portfolio.get_total_value() * self.max_daily_loss_pct  # < 0 => rechazaria
    cur = st["min_daily_loss_margin"]
    st["min_daily_loss_margin"] = m if cur is None else min(cur, m)
    return x
def t(self, trade):
    _t(self, trade)
    st["breaker_active"] += bool(self.circuit_breaker_triggered)
R.validate_trade, R.get_daily_pnl, R.on_trade_executed = v, d, t
from quantagent.cli.__main__ import cli
try:
    cli(sys.argv[1:], standalone_mode=False)
finally:
    json.dump(st, open(os.environ["PROBE_OUT"], "w"))
"""

_TMP = tempfile.TemporaryDirectory(prefix="crossval_")  # se borra al salir el proceso


@functools.cache
def run_engine(probe: bool = False, intrabar: bool = False) -> dict:
    """Corre el CLI del motor sobre una base SQLite nueva. Cacheado por proceso (los tests lo reusan)."""
    d = Path(_TMP.name) / f"{'probe' if probe else 'plain'}{'_intra' if intrabar else ''}"
    d.mkdir()
    db = d / "engine.db"
    db.touch()
    env = {**os.environ, "TRADING_SLIPPAGE_PCT": "0", "DATABASE_URL": f"sqlite:///{db}",
           "PROBE_OUT": str(d / "probe.json")}
    subprocess.run([sys.executable, "-c", "from quantagent.database import init_db; init_db()"],
                   cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    args = ["backtest", "run", "--strategy", "rsi", "--fixture", "spy-90d",
            "--out", str(d / "trades.csv"), "--equity-out", str(d / "equity.csv")]
    if intrabar:
        args.append("--intrabar-stops")
    else:
        args.append("--no-intrabar-stops")
    cmd = [sys.executable, "-c", PROBE, *args] if probe else [sys.executable, "-m", "quantagent.cli", *args]
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"el motor fallo ({p.returncode}):\n{p.stdout[-1500:]}\n{p.stderr[-1500:]}")
    out = p.stdout
    num = lambda pat: float(re.search(pat, out).group(1))  # noqa: E731
    pf = re.search(r"Profit factor: (\S+)", out).group(1)
    printed = {"trades": int(num(r"Trades: (\d+)")), "win_rate": num(r"Win rate: ([\d.]+)%") / 100,
               "profit_factor": None if pf == "n/a" else float(pf), "total_pnl": num(r"Total PnL: (-?[\d.]+)"),
               "max_drawdown": num(r"max drawdown: ([\d.]+)")}
    result = {"trades": d / "trades.csv", "equity": d / "equity.csv", "printed": printed}
    if probe:
        result["probe"] = json.loads((d / "probe.json").read_text())
    return result


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run_port_rows(history_bars: int, tag: str, intrabar: bool = False):
    """Devuelve (filas de trades como las del CSV, stats de backtesting.py) para el port."""
    stats = port.run_port_stats(history_bars, intrabar=intrabar)
    path = Path(_TMP.name) / f"port_{tag}.csv"
    port.write_csv(stats["_trades"].sort_values("EntryTime"), path)
    return read_rows(path), stats


def metrics_of(rows: list[dict], tag: str) -> dict:
    path = Path(_TMP.name) / f"rows_{tag}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return recalc_metrics.recalc(path)


def pnl_tol(row: dict) -> float:
    return PNL_TOL * abs(float(row["exit_price"]) - float(row["entry_price"])) + PNL_TOL


def compare_trades(eng: list[dict], prt: list[dict]) -> tuple[list[str], bool]:
    lines, ok = [f"Trades: motor={len(eng)} port={len(prt)}"], len(eng) == len(prt)
    if not ok:
        lines.append("  DIFERENTE: distinta cantidad de trades")
    maxdiff = dict.fromkeys(NUMERIC, 0.0)
    first = None
    for i, (e, p) in enumerate(zip(eng, prt)):
        bad = [c for c in ("entry_time", "exit_time", "symbol", "side") if e[c] != p[c]]
        tols = {"qty": QTY_TOL, "entry_price": PRICE_TOL, "exit_price": PRICE_TOL, "pnl": pnl_tol(e)}
        for c in NUMERIC:
            diff = abs(float(e[c]) - float(p[c]))
            maxdiff[c] = max(maxdiff[c], diff)
            if diff > tols[c]:
                bad.append(f"{c} (|dif|={diff:.3g} > tol {tols[c]:.3g})")
        if bad and first is None:
            first = (i, bad, e, p)
    if first:
        ok = False
        i, bad, e, p = first
        lines += [f"PRIMER TRADE DIFERENTE: nº {i + 1}, columnas: {', '.join(bad)}",
                  f"  motor: {json.dumps(e)}", f"  port : {json.dumps(p)}"]
    else:
        lines.append("Todos los trades comparados coinciden en entry_time, exit_time, symbol, side y dentro de tolerancia")
    lines.append("Maxima diferencia absoluta por columna (tolerancia): " + ", ".join(
        f"{c}={maxdiff[c]:.3g} ({'1e-8' if c != 'pnl' else '1e-8*|mov|+1e-8'})" for c in NUMERIC))
    return lines, ok


def port_equity(stats, n_real: int):
    eq = stats["_equity_curve"]["Equity"].iloc[:n_real]
    return [t.isoformat() for t in eq.index], [float(x) for x in eq.values]


def drawdown(timestamps: list[str], equities: list[float]) -> float:
    path = Path(_TMP.name) / "dd.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "equity"])
        w.writerows(zip(timestamps, equities))
    return recalc_metrics.recalc_equity(str(path))["max_drawdown"]


def compare_all(args) -> tuple[list[str], bool]:
    lines, ok = [], True
    intrabar = bool(args.intrabar)
    eng, proj = run_engine(intrabar=intrabar), port.project_params()
    eng_rows = read_rows(eng["trades"])
    tag = "intrabar" if intrabar else "main"
    prt_rows, stats = run_port_rows(proj["_required_history_bars"], tag, intrabar=intrabar)
    if args.port_pnl_delta:  # costura para el test "la comparacion puede fallar"
        idx, delta = args.port_pnl_delta.split(":")
        prt_rows[int(idx)]["pnl"] = str(float(prt_rows[int(idx)]["pnl"]) + float(delta))
    if getattr(args, "engine_pnl_delta", None):
        idx, delta = args.engine_pnl_delta.split(":")
        eng_rows[int(idx)]["pnl"] = str(float(eng_rows[int(idx)]["pnl"]) + float(delta))

    tl, t_ok = compare_trades(eng_rows, prt_rows)
    lines += ["== Trades ==", *tl]
    ok &= t_ok

    me = metrics_of(eng_rows, f"eng_{tag}") if getattr(args, "engine_pnl_delta", None) else recalc_metrics.recalc(eng["trades"])
    mp = metrics_of(prt_rows, tag)

    n_real = len(stats["_equity_curve"]) - 1  # sin la vela centinela
    eq_rows = read_rows(eng["equity"])
    e_ts, e_eq = [r["timestamp"] for r in eq_rows], [float(r["equity"]) for r in eq_rows]
    p_ts, p_eq = port_equity(stats, n_real)
    same_instants = e_ts == p_ts
    lines += ["== Curva de equity ==",
              f"Instantes: motor={len(e_ts)} port={len(p_ts)} (sin vela centinela); "
              + ("mismos instantes" if same_instants else "DISTINTOS")]
    if same_instants:
        d = max(abs(a - b) for a, b in zip(e_eq, p_eq))
        lines.append(f"Maxima diferencia de equity punto a punto: {d:.3g} USD (tolerancia {EQUITY_TOL})")
        ok &= d <= EQUITY_TOL
        dd_e, dd_p = drawdown(e_ts, e_eq), drawdown(p_ts, p_eq)
    else:
        ok = False
        lines.append("NO COMPARADO: max drawdown ni equity punto a punto; las curvas no estan muestreadas "
                     "en los mismos instantes")
        dd_e = dd_p = float("nan")

    pe, pp = me["profit_factor"], mp["profit_factor"]
    pf_ok = pe == pp or abs(pe - pp) <= PF_RTOL * abs(pe)
    pnl_tol_total = sum(pnl_tol(r) for r in eng_rows)
    checks = [("trades", me["trades"], mp["trades"], me["trades"] == mp["trades"]),
              ("total_pnl", me["total_pnl"], mp["total_pnl"],
               abs(me["total_pnl"] - mp["total_pnl"]) <= pnl_tol_total),
              ("win_rate", me["win_rate"], mp["win_rate"], me["win_rate"] == mp["win_rate"]),
              ("profit_factor", pe, pp, pf_ok),
              ("max_drawdown", dd_e, dd_p, abs(dd_e - dd_p) <= DD_TOL)]
    lines += ["== Metricas finales (recalculadas desde cada CSV con scripts/recalc_metrics.py) ==",
              f"{'metrica':<15}{'motor':>18}{'port':>18}  dentro de tolerancia"]
    for name, a, b, good in checks:
        lines.append(f"{name:<15}{a:>18.8g}{b:>18.8g}  {'SI' if good else 'NO'}")
        ok &= good

    pr = eng["printed"]
    own = {"trades": me["trades"], "win_rate": round(me["win_rate"], 4), "profit_factor": round(pe, 2),
           "total_pnl": round(me["total_pnl"], 2), "max_drawdown": round(dd_e, 6)}
    mism = [k for k, v in own.items() if abs(v - pr[k]) > {"max_drawdown": 1e-6, "win_rate": 1e-4}.get(k, 0.011)]
    lines.append("Lo que imprimio el CLI del motor vs lo recalculado desde su CSV: "
                 + ("coincide" if not mism else f"DIFIERE en {mism} (CLI={pr})"))
    ok &= not mism

    probe = run_engine(probe=True, intrabar=intrabar)
    same_probe = Path(probe["trades"]).read_bytes() == Path(eng["trades"]).read_bytes()
    pb = probe["probe"]
    lines += ["== Informativo: riesgo del motor (corrida instrumentada del mismo CLI) ==",
              f"validate_trade llamado {pb['validate_calls']} veces; rechazos: {pb['rejections'] or 'ninguno'}",
              f"Margen minimo al limite de perdida diaria: {pb['min_daily_loss_margin']:.2f} USD "
              f"(<0 rechazaria); circuit breaker activo tras {pb['breaker_active']} ejecuciones; "
              f"date.today() del reloj={pb['today']}",
              "La corrida instrumentada produjo el mismo CSV de trades que la normal: "
              + ("SI" if same_probe else "NO")]
    ok &= same_probe
    reasons = collections.Counter(r["exit_reason"] for r in eng_rows)
    lines.append(f"exit_reason del motor: {dict(reasons)} (TRAILING_STOP: {reasons['TRAILING_STOP']})")

    return lines, bool(ok)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--intrabar", action="store_true", help="evalua stop loss y take profit dentro de la vela")
    ap.add_argument("--port-pnl-delta", metavar="IDX:DELTA", help=argparse.SUPPRESS)  # solo para tests
    ap.add_argument("--engine-pnl-delta", metavar="IDX:DELTA", help=argparse.SUPPRESS)  # solo para tests
    args = ap.parse_args(argv)
    lines, ok = compare_all(args)
    print("\n".join(lines))
    print("\nRESULTADO: " + ("TODO DENTRO DE TOLERANCIA" if ok else "HAY DIFERENCIAS (exit 1)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
