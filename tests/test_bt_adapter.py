"""QuantAgent-i6n: el adaptador generico (scripts/bt_adapter.py) corre una TradingStrategy dentro de backtesting.py.

Estrategia minima armada aca y seis velas a mano; sin red ni snapshot. Se omite si `backtesting` no esta
instalado (dependencia de desarrollo). La estrategia compra (confianza 0.5, stop 5 % abajo, take profit
10 % arriba) cuando la ultima vela trae volumen 2.

Los escenarios corren en UN subproceso (este mismo archivo como script): crossval_rsi fija
TRADING_SLIPPAGE_PCT=0 al importarse y no debe filtrarse a otros tests del mismo worker.
"""

import json
import os
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
# QuantAgent-ws3d: la vela 3 abre en 115 (> take profit 110) y luego el minimo 90 toca el stop 95
CANDLES_OPENING_GAP = [
    (1, 100.0, 101.0, 99.0, 100.0, 1),
    (2, 100.0, 101.0, 99.0, 100.0, 2),  # senal: entrada en 100, stop 95, take profit 110
    (3, 115.0, 115.0, 90.0, 95.0, 1),  # gap alcista > tp y minimo < sl
    (4, 95.0, 96.0, 94.0, 95.0, 1),
]
SCENARIOS = {
    "default": {},
    "exit_after_1": {"exit_after": 1},
    "exit_after_0": {"exit_after": 0},
    "next_open": {"fill": "next-open", "commission_pct": 0.001},
    "slippage": {"exit_after": 0, "always": True, "slippage_pct": 0.01},  # QuantAgent-xyf6
    "recorte": {"confidence": 1.0, "pct": 1.0, "slippage_pct": 0.01, "commission_pct": 0.001},  # QuantAgent-gti6
    "recorte_next_open": {"fill": "next-open", "confidence": 1.0, "pct": 1.0,
                          "slippage_pct": 0.01, "commission_pct": 0.001},
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
        def __init__(self, exit_after=None, always=False, confidence=0.5):
            self.confidence = confidence
            self.exit_after = exit_after  # None = should_exit por defecto de TradingStrategy
            self.always = always  # senal en todas las velas, no solo con volumen 2
            self.signal_calls, self.exit_calls = [], []

        @property
        def required_history_bars(self) -> int:
            return 2

        def generate_signal(self, kline_data, symbol, timeframe, current_price):
            self.signal_calls.append({"timestamps": [k["timestamp"].isoformat() for k in kline_data],
                                      "keys": sorted(kline_data[0]), "symbol": symbol, "timeframe": timeframe,
                                      "price": current_price})
            if not self.always and kline_data[-1]["volume"] != 2:
                return None
            return TradingSignal(decision="LONG", confidence=self.confidence, stop_loss=current_price * 0.95,
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
        bt_adapter.settings.TRADING_BASE_POSITION_PCT = kwargs.pop("pct", 0.05)
        strategy = MiniStrategy(exit_after=kwargs.pop("exit_after", None), always=kwargs.pop("always", False),
                                confidence=kwargs.pop("confidence", 0.5))
        stats = bt_adapter.run_adapter(strategy, crossval_rsi.load_fixture(path), symbol="XYZ", timeframe="1d", **kwargs)
        rows = [{**vars(r), "entry_time": r.entry_time.isoformat(), "exit_time": r.exit_time.isoformat()}
                for r in bt_adapter.trade_rows(stats, "XYZ")]
        out[name] = {"rows": rows, "signal_calls": strategy.signal_calls, "exit_calls": strategy.exit_calls}

    # QuantAgent-ws3d: corrida con apertura que supera el take profit y minimo que toca el stop
    gap_path = workdir / "gap.csv"
    gap_lines = ["timestamp,open,high,low,close,volume"] + [f"{day(d)},{o},{h},{lo},{c},{v}" for d, o, h, lo, c, v in CANDLES_OPENING_GAP]
    gap_path.write_text("\n".join(gap_lines) + "\n")
    gap_strat = MiniStrategy()
    bt_adapter.settings.TRADING_BASE_POSITION_PCT = 0.05
    gap_stats = bt_adapter.run_adapter(gap_strat, crossval_rsi.load_fixture(gap_path), symbol="XYZ", timeframe="1d")
    out["opening_gap"] = [{**vars(r), "entry_time": r.entry_time.isoformat(), "exit_time": r.exit_time.isoformat()}
                          for r in bt_adapter.trade_rows(gap_stats, "XYZ")]

    return out


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    pytest.importorskip("backtesting")
    proc = subprocess.run([sys.executable, __file__, str(tmp_path_factory.mktemp("bt_adapter"))],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.splitlines()[-1])


def test_salto_en_apertura_sale_por_take_profit_antes_del_stop(results):
    """QuantAgent-ws3d: si la vela abre por encima del take profit y toca el stop, sale en la apertura con TAKE_PROFIT."""
    rows = results["opening_gap"]
    assert len(rows) == 1
    trade = rows[0]
    assert (trade["entry_time"], trade["entry_price"]) == (day(2), 100.0)
    assert (trade["exit_time"], trade["exit_price"], trade["exit_reason"]) == (day(3), 115.0, "TAKE_PROFIT")
    assert trade["pnl"] == pytest.approx(25.0 * (115.0 - 100.0))


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


def test_slippage_con_la_regla_del_paper_broker(results):
    """Compra a referencia * (1 + s), vende a referencia * (1 - s), en la entrada, la salida por regla y el stop."""
    s = 0.01
    first, second = results["slippage"]["rows"][:2]

    # tamano y entry_price de should_exit usan la referencia 100; solo el precio de llenado lleva slippage
    assert (first["entry_time"], first["qty"], first["entry_price"]) == (day(2), 25.0, pytest.approx(100.0 * (1 + s)))
    assert results["slippage"]["exit_calls"][0][1] == 100.0
    assert (first["exit_time"], first["exit_reason"]) == (day(3), "REGLA_PROPIA")
    assert first["exit_price"] == pytest.approx(101.0 * (1 - s))
    pnl = 25.0 * (101.0 * (1 - s) - 100.0 * (1 + s))
    assert first["pnl"] == pytest.approx(pnl)

    # reentra en la vela de la salida: el tamano ya descuenta el slippage de esa salida. Stop 95.95 tocado en la vela 4
    assert second["entry_time"] == day(3)
    assert second["qty"] == pytest.approx((100000 + pnl) * 0.05 * 0.5 / 101.0, abs=1e-8)
    assert second["entry_price"] == pytest.approx(101.0 * (1 + s))
    assert (second["exit_time"], second["exit_reason"]) == (day(4), "STOP_LOSS")
    assert second["exit_price"] == pytest.approx(101.0 * 0.95 * (1 - s))


def test_recorte_de_la_compra_al_efectivo_con_slippage_y_comision(results):
    """Al 100 % con confianza 1.0 la compra se achica a lo que el efectivo paga con slippage y comision."""
    first = results["recorte"]["rows"][0]  # sin el recorte la libreria cancela la orden y no hay trades

    # la regla de OrderManager._cap_entry_qty: efectivo / (precio * (1 + slippage) * (1 + comision))
    assert first["qty"] == pytest.approx(100000 / (100.0 * 1.01 * 1.001), abs=1e-7)
    assert (first["entry_time"], first["entry_price"]) == (day(2), pytest.approx(101.0))


def test_recorte_en_apertura_con_gap_y_costos(results):
    """El gap de 100 a 100.5 ya no cancela la compra al 100 %; la segunda usa el efectivo restante."""
    first, second = results["recorte_next_open"]["rows"]
    qty = 100000 / (100.5 * 1.01 * 1.001)
    assert (first["entry_time"], first["entry_price"]) == (day(3), pytest.approx(100.5 * 1.01))
    assert first["qty"] == pytest.approx(qty, abs=1e-8)
    assert first["qty"] * first["entry_price"] * 1.001 <= 100000
    pnl = first["qty"] * (95 * 0.99 - 100.5 * 1.01) - 0.001 * first["qty"] * (100.5 * 1.01 + 95 * 0.99)
    assert first["pnl"] == pytest.approx(pnl)
    assert second["entry_time"] == day(6)
    assert second["qty"] == pytest.approx((100000 + pnl) / (96 * 1.01 * 1.001), abs=1e-8)


@pytest.mark.parametrize("strategy, fixture, pct, expected", [
    ("rsi", "spy-90d", None, (340, 4571.27)),  # referencia de M1, al 5 % por defecto
    # QuantAgent-gti6: confianza 1.0 al 100 %. Sin el recorte la libreria cancela las ordenes y da 0 trades
    ("sma-cross", "spy-2y-1d", "1.0", (2, -6305.04)),
])
def test_motor_y_adaptador_coinciden_con_slippage(tmp_path, strategy, fixture, pct, expected):
    """Con slippage 0.0005: mismos trades que el motor, con las tolerancias de crossval_compare."""
    pytest.importorskip("backtesting")
    (tmp_path / "engine.db").touch()
    env = {**os.environ, "TRADING_SLIPPAGE_PCT": "0.0005", "DATABASE_URL": f"sqlite:///{tmp_path / 'engine.db'}"}
    size = ["--tamano-posicion", pct] if pct else []
    env.update({"TRADING_BASE_POSITION_PCT": pct} if pct else {})  # de ahi toma el tamano el adaptador
    compare = ("import sys; sys.path.insert(0, 'scripts'); import crossval_compare as c; "
               "e, a = c.read_rows(sys.argv[1]), c.read_rows(sys.argv[2]); lines, ok = c.compare_trades(e, a); "
               "print(len(e), sum(float(r['pnl']) for r in e), float(a[0]['qty']) * float(a[0]['entry_price'])); "
               "print(*lines, sep='\\n'); "
               "sys.exit(not ok or [r['exit_reason'] for r in e] != [r['exit_reason'] for r in a])")
    for cmd in (["-c", "from quantagent.database import init_db; init_db()"],
                ["-m", "quantagent.cli", "backtest", "run", "--strategy", strategy, "--fixture", fixture, *size,
                 "--out", "motor.csv"],
                ["scripts/bt_adapter.py", "--strategy", strategy, "--fixture", fixture, "--slippage-pct", "0.0005",
                 "--out", "adaptador.csv"],
                ["-c", compare, "motor.csv", "adaptador.csv"]):
        cmd = [str(tmp_path / a) if a.endswith(".csv") else a for a in cmd]
        proc = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    trades, total_pnl, first_cost = proc.stdout.splitlines()[0].split()
    assert (int(trades), round(float(total_pnl), 2)) == expected  # no es una corrida vacia
    if pct:  # la primera compra del adaptador, pagada con slippage, gasta todo el efectivo inicial y no mas
        assert float(first_cost) == pytest.approx(100000.0, abs=1e-4)


if __name__ == "__main__":
    print(json.dumps(_run_scenarios(Path(sys.argv[1]))))
