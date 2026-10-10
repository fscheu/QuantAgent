"""QuantAgent-i6n: el adaptador generico (scripts/bt_adapter.py) corre una TradingStrategy dentro de backtesting.py.

Estrategia minima armada aca y seis velas a mano; sin red ni snapshot. Se omite si `backtesting` no esta
instalado (dependencia de desarrollo). La estrategia compra (confianza 0.5, stop 5 % abajo, take profit
10 % arriba) cuando la ultima vela trae volumen 2.

Los escenarios corren en UN subproceso (este mismo archivo como script): crossval_rsi fija
TRADING_SLIPPAGE_PCT=0 al importarse y no debe filtrarse a otros tests del mismo worker.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.oraculo  # el subproceso tarda ~5 s en importar el proyecto y la libreria

# dia, open, high, low, close, volume
CANDLES = [
    (1, 100.0, 101.0, 99.0, 100.0, 1),
    (2, 100.0, 101.0, 99.0, 100.0, 2),  # senal: stop 95, take profit 110
    (3, 100.5, 102.0, 98.0, 101.0, 1),  # no toca ni stop ni take profit
    (4, 101.0, 101.0, 94.0, 96.0, 1),  # el minimo 94 perfora el stop 95 dentro de la vela (cierra en 96)
    (5, 96.0, 97.0, 95.5, 96.0, 2),  # senal: stop 91.2
    (6, 96.0, 97.0, 95.5, 96.0, 1),  # ultima vela
]
SCENARIOS = {
    "default": {},
    "exit_after_1": {"exit_after": 1},
    "exit_after_0": {"exit_after": 0},
    "next_open": {"fill": "next-open", "commission_pct": 0.001},
}


def day(d: int) -> str:
    return datetime(2024, 1, d).isoformat()


def _run_scenarios(workdir: Path) -> dict:
    """Solo en el subproceso: arma la estrategia minima y corre cada escenario con el adaptador."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import bt_adapter
    import crossval_rsi

    from quantagent.strategy.base import TradingSignal, TradingStrategy

    class MiniStrategy(TradingStrategy):
        def __init__(self, exit_after=None):
            self.exit_after = exit_after  # None = should_exit por defecto de TradingStrategy
            self.signal_calls, self.exit_calls = [], []

        @property
        def required_history_bars(self) -> int:
            return 2

        def generate_signal(self, kline_data, symbol, timeframe, current_price):
            self.signal_calls.append({"timestamps": [k["timestamp"].isoformat() for k in kline_data],
                                      "keys": sorted(kline_data[0]), "symbol": symbol, "timeframe": timeframe,
                                      "price": current_price})
            if kline_data[-1]["volume"] != 2:
                return None
            return TradingSignal(decision="LONG", confidence=0.5, stop_loss=current_price * 0.95,
                                 take_profit=current_price * 1.10)

        def should_exit(self, position, current_price, ohlc_data):
            if self.exit_after is None:
                return super().should_exit(position, current_price, ohlc_data)
            self.exit_calls.append([position.side.value, float(position.entry_price), float(position.stop_loss),
                                    float(position.quantity), position.candles_since_entry, current_price,
                                    list(ohlc_data["close"])])
            return (True, "REGLA_PROPIA") if position.candles_since_entry >= self.exit_after else (False, None)

        def should_reevaluate(self, position, current_price):
            return False

    path = workdir / "velas.csv"
    lines = ["timestamp,open,high,low,close,volume"] + [f"{day(d)},{o},{h},{lo},{c},{v}" for d, o, h, lo, c, v in CANDLES]
    path.write_text("\n".join(lines) + "\n")
    out = {}
    for name, kwargs in SCENARIOS.items():
        kwargs = dict(kwargs)
        strategy = MiniStrategy(exit_after=kwargs.pop("exit_after", None))
        stats = bt_adapter.run_adapter(strategy, crossval_rsi.load_fixture(path), symbol="XYZ", timeframe="1d", **kwargs)
        rows = [{**vars(r), "entry_time": r.entry_time.isoformat(), "exit_time": r.exit_time.isoformat()}
                for r in bt_adapter.trade_rows(stats, "XYZ")]
        out[name] = {"rows": rows, "signal_calls": strategy.signal_calls, "exit_calls": strategy.exit_calls}
    return out


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    pytest.importorskip("backtesting")
    proc = subprocess.run([sys.executable, __file__, str(tmp_path_factory.mktemp("bt_adapter"))],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.splitlines()[-1])


def test_entrada_al_cierre_y_stop_tocado_dentro_de_la_vela(results):
    """La estrategia recibe la ventana que declara; la libreria llena al cierre y ejecuta el stop contra el minimo."""
    first, second = results["default"]["rows"]

    call = results["default"]["signal_calls"][0]
    assert call["timestamps"] == [day(1), day(2)]  # las ultimas 2 velas, no mas
    assert call["keys"] == ["close", "high", "low", "open", "timestamp", "volume"]
    assert (call["symbol"], call["timeframe"], call["price"]) == ("XYZ", "1d", 100.0)

    # tamano = equity 100000 * 5 % * confianza 0.5 / precio 100 = 25
    assert (first["entry_time"], first["entry_price"], first["side"], first["qty"]) == (day(2), 100.0, "buy", 25.0)
    # la vela 4 cierra en 96 pero el stop se ejecuta en 95, dentro de la vela
    assert (first["exit_time"], first["exit_price"], first["stop_loss"]) == (day(4), 95.0, 95.0)
    assert first["exit_reason"] == "STOP_LOSS"
    assert first["pnl"] == pytest.approx(-125.0)

    # la segunda entrada se dimensiona con el equity de la libreria despues de la perdida, y sigue abierta al final
    assert second["entry_time"] == day(5)
    assert second["qty"] == pytest.approx((100000 - 125) * 0.05 * 0.5 / 96.0, abs=1e-8)
    assert (second["exit_time"], second["exit_price"], second["exit_reason"]) == (day(6), 96.0, "backtest_end")


def test_salida_por_should_exit_personalizado(results):
    """should_exit propio recibe la posicion traducida y su cierre es al close; el stop viejo no abre nada despues."""
    # vela 3: primera evaluacion (0 velas desde la entrada, no sale); vela 4: el stop nativo cerro antes de preguntar
    late = results["exit_after_1"]
    assert late["exit_calls"][0] == ["buy", 100.0, 95.0, 25.0, 0, 101.0, [100.0, 101.0]]
    assert late["rows"][0]["exit_reason"] == "STOP_LOSS"

    # sale en la vela 3 al close 101; el minimo 94 de la vela 4 ya no encuentra posicion ni abre una contraria
    rows = results["exit_after_0"]["rows"]
    assert [(r["side"], r["entry_time"], r["exit_time"], r["exit_price"], r["exit_reason"]) for r in rows] == [
        ("buy", day(2), day(3), 101.0, "REGLA_PROPIA"),
        ("buy", day(5), day(6), 96.0, "REGLA_PROPIA"),
    ]
    assert rows[0]["pnl"] == pytest.approx(25.0)


def test_llenado_en_la_apertura_siguiente_y_comision_nativa(results):
    """--fill next-open entra al open de la vela siguiente a la senal; la comision de la libreria llega al PnL."""
    first, second = results["next_open"]["rows"]

    assert (first["entry_time"], first["entry_price"], first["qty"]) == (day(3), 100.5, 25.0)  # tamano de la senal
    assert (first["exit_time"], first["exit_price"], first["exit_reason"]) == (day(4), 95.0, "STOP_LOSS")
    # 25 * (95 - 100.5) menos 0.1 % del nocional en la entrada y en la salida
    assert first["pnl"] == pytest.approx(-137.5 - 0.001 * 25 * (100.5 + 95.0))
    assert (second["entry_time"], second["exit_time"], second["exit_reason"]) == (day(6), day(6), "backtest_end")


if __name__ == "__main__":
    print(json.dumps(_run_scenarios(Path(sys.argv[1]))))
