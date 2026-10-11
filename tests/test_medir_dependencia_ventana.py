"""QuantAgent-4x6: scripts/medir_dependencia_ventana.py sobre el fixture spy-smoke (120 velas de 1h)."""

import csv
import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quantagent.backtesting.backtest import Backtest
from quantagent.backtesting.fixtures import FIXTURES_DIR, load_fixture
from quantagent.models import Base
from quantagent.strategy.base import TradingSignal, TradingStrategy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import medir_dependencia_ventana as mdv  # noqa: E402


class PrimeraContraUltima(TradingStrategy):
    """LONG si el último cierre supera al primero de la ventana: depende del largo a propósito."""

    required_history_bars = 10

    def generate_signal(self, kline_data, symbol, timeframe, current_price):
        if kline_data[-1]["close"] <= kline_data[0]["close"]:
            return None
        return TradingSignal(decision="LONG", confidence=1.0)

    def should_reevaluate(self, position, current_price):
        return False


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    load_fixture(s, "spy-smoke")
    yield s
    s.close()


def test_la_ventana_es_la_que_arma_el_motor(session):
    """`ventana` devuelve lo mismo que `_get_history_df`, con ventana incompleta (i=3) y completa."""
    motor, historia = mdv.cargar_historia(session)
    assert len(historia) == 120
    for i, largo in [(3, 10), (50, 10), (50, 11), (119, 20)]:
        del_motor = Backtest._get_history_df(motor, motor.symbol, historia["timestamp"].iloc[i], largo)
        pd.testing.assert_frame_equal(mdv.ventana(historia, i, largo), del_motor)


def test_medir_cuenta_las_fechas_donde_cambia_la_senal(session):
    """Los conteos de la fila +1 (ventana de 11) coinciden con los calculados acá desde el CSV."""
    with (FIXTURES_DIR / "spy-smoke.csv").open() as f:
        c = [float(row["close"]) for row in csv.DictReader(f)]
    con_10 = {i: c[i] > c[i - 9] for i in range(9, len(c))}
    con_11 = {i: c[i] > c[i - 10] for i in range(10, len(c))}
    cambia = [i for i in con_11 if con_10[i] != con_11[i]]
    assert cambia, "el fixture tiene que tener fechas donde la señal cambia"

    filas = {f["largo"]: f for f in mdv.medir(session, PrimeraContraUltima())}

    assert filas["actual (10)"]["cambia_senal"] == []
    assert filas["actual (10)"]["senales_actual"] == sum(con_10.values())
    fechas = list(mdv.cargar_historia(session)[1]["timestamp"])
    assert filas["+1 (11)"]["fechas_comparables"] == len(con_11) == 110
    assert filas["+1 (11)"]["cambia_senal"] == [fechas[i] for i in cambia]
    # La vela 9 tiene ventana de 10 pero no de 11: si daba señal, se pierde solo por el arranque.
    assert len(filas["+1 (11)"]["solo_arranque"]) == int(con_10[9])


def test_main_imprime_la_tabla_del_fixture(capsys):
    """RSI promedia las últimas 14 diferencias: su señal no cambia con ningún largo de ventana."""
    mdv.main(["--fixture", "spy-smoke", "--strategy", "rsi"])
    filas = [line.split("|") for line in capsys.readouterr().out.splitlines() if line.startswith("| rsi")]
    assert [f[2].strip() for f in filas] == ["actual (30)", "+1 (31)", "+5 (35)", "doble (60)"]
    assert [f[3].strip() for f in filas] == ["91", "90", "86", "61"]  # 120 velas - (largo - 1)
    assert [f[5].strip() for f in filas] == ["0", "0", "0", "0"]
