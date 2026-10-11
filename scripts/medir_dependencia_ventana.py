"""Mide si la respuesta de una estrategia depende del largo de la ventana de historia (QuantAgent-4x6).

Lee las velas con `Backtest._get_history_df` (rama offline, la del CLI), arma para cada vela la ventana
de las últimas N y llama a la estrategia igual que `_analyze_and_trade`: `should_exit` con la ventana que
haya y `generate_signal` solo si la ventana está completa. Repite con otros largos y compara.

    python scripts/medir_dependencia_ventana.py --snapshot etf-1d-2026-10 --symbol SPY --from 2007-01-03 --to 2018-12-31
    python scripts/medir_dependencia_ventana.py --fixture spy-1y-4h --strategy triple-screen [--detalle] [--strategy-dir DIR]
"""

import argparse
import importlib
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

import quantagent.strategy as strategy_pkg  # noqa: E402
from quantagent.backtesting.backtest import Backtest  # noqa: E402
from quantagent.backtesting.fixtures import load_fixture, load_snapshot  # noqa: E402
from quantagent.data.provider import DataProvider  # noqa: E402
from quantagent.models import Base, MarketData  # noqa: E402
from quantagent.strategy.base import TradingStrategy  # noqa: E402

ESTRATEGIAS = {
    "rsi": ("rsi_strategy", "RSIMeanReversionStrategy"),
    "fifty-two-week-high": ("fifty_two_week_high_strategy", "FiftyTwoWeekHighStrategy"),
    "triple-screen": ("triple_screen_strategy", "TripleScreenStrategy"),
    "sma-cross": ("sma_cross_strategy", "SmaCrossStrategy"),
    "momentum-12m": ("momentum_12m_strategy", "Momentum12mStrategy"),
}
LARGOS = {"actual": lambda n: n, "+1": lambda n: n + 1, "+5": lambda n: n + 5, "doble": lambda n: 2 * n}


def cargar_historia(session):
    """Todas las velas de la base con el formato del motor: una sola llamada a `_get_history_df`."""
    ultima = session.query(MarketData).order_by(MarketData.timestamp.desc()).first()
    motor = SimpleNamespace(config={"offline_data": True}, db=session, timeframe=ultima.timeframe,
                            symbol=ultima.symbol, data_provider=DataProvider(session, offline=True))
    return motor, Backtest._get_history_df(motor, ultima.symbol, ultima.timestamp, session.query(MarketData).count())


def ventana(historia, i, largo):
    """Las últimas `largo` velas hasta la fila `i`: lo que devuelve `_get_history_df` para esa fecha."""
    return historia.iloc[max(0, i + 1 - largo): i + 1].reset_index(drop=True)


def respuestas(strategy, historia, motor, largo):
    """Por vela, (señal, salida). Señal None: ventana incompleta, el motor no llama a `generate_signal`."""
    salida_propia = type(strategy).should_exit is not TradingStrategy.should_exit
    out = []
    for i in range(len(historia)):
        df = ventana(historia, i, largo)
        precio = float(df.iloc[-1]["close"])
        senal = None
        if len(df) >= largo:
            signal = strategy.generate_signal(df.to_dict(orient="records"), motor.symbol, motor.timeframe, precio)
            senal = signal.decision if signal else "HOLD"
        out.append((senal, strategy.should_exit(None, precio, df)[0] if salida_propia else None))
    return out


def medir(session, strategy):
    """Una fila por largo de `LARGOS`, comparada contra el largo actual (`required_history_bars`)."""
    motor, historia = cargar_historia(session)
    actual = strategy.required_history_bars
    base = respuestas(strategy, historia, motor, actual)
    filas = []
    for etiqueta, calcular in LARGOS.items():
        otra = respuestas(strategy, historia, motor, calcular(actual))
        pares = list(zip(historia["timestamp"], base, otra))
        completas = [(f, b, o) for f, b, o in pares if b[0] is not None and o[0] is not None]
        filas.append({
            "largo": f"{etiqueta} ({calcular(actual)})",
            "fechas_comparables": len(completas),
            "senales_actual": sum(b[0] != "HOLD" for _, b, _ in completas),
            "cambia_senal": [f for f, b, o in completas if b[0] != o[0]],
            "solo_arranque": [f for f, b, o in pares if b[0] not in (None, "HOLD") and o[0] is None],
            "cambia_salida": [f for f, b, o in pares if b[1] != o[1]],
        })
    return filas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixture")
    ap.add_argument("--snapshot")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--from", dest="desde", type=datetime.fromisoformat)
    ap.add_argument("--to", dest="hasta", type=datetime.fromisoformat)
    ap.add_argument("--strategy", nargs="+", choices=list(ESTRATEGIAS), default=list(ESTRATEGIAS))
    ap.add_argument("--detalle", action="store_true", help="Lista las fechas donde cambia la señal")
    ap.add_argument("--strategy-dir", help="Carpeta con módulos de estrategia que no están en esta rama")
    args = ap.parse_args(argv)
    if bool(args.fixture) == bool(args.snapshot):
        ap.error("indicá exactamente uno: --fixture o --snapshot")
    if args.snapshot and (args.hasta is None or args.hasta >= datetime(2023, 1, 1)):
        ap.error("--snapshot requiere --to anterior a 2023-01-01 (reserva)")
    if args.strategy_dir:
        strategy_pkg.__path__.append(args.strategy_dir)

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    if args.fixture:
        load_fixture(session, args.fixture)
    else:
        load_snapshot(session, args.snapshot, args.symbol, args.desde, args.hasta)

    print("| estrategia | largo | fechas comparables | señales (largo actual) | cambia la señal | solo arranque | cambia la salida |")
    print("|---|---|---|---|---|---|---|")
    for nombre in args.strategy:
        try:
            strategy = getattr(importlib.import_module(f"quantagent.strategy.{ESTRATEGIAS[nombre][0]}"), ESTRATEGIAS[nombre][1])()
        except ModuleNotFoundError:
            print(f"| {nombre} | no está en esta rama | | | | | |")
            continue
        for fila in medir(session, strategy):
            salida = len(fila["cambia_salida"]) if type(strategy).should_exit is not TradingStrategy.should_exit else "n/a"
            print(f"| {nombre} | {fila['largo']} | {fila['fechas_comparables']} | {fila['senales_actual']} "
                  f"| {len(fila['cambia_senal'])} | {len(fila['solo_arranque'])} | {salida} |")
            if args.detalle and fila["cambia_senal"]:
                print("  " + " ".join(f"{f:%Y-%m-%d}" for f in fila["cambia_senal"]))


if __name__ == "__main__":
    main()
