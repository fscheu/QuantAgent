"""QuantAgent-j2rt: una estrategia definida acá y registrada con UNA llamada corre en `backtest run` y en el
adaptador sin tocar `quantagent/backtesting/` ni `quantagent/cli/`. Subproceso: importar el adaptador fija
TRADING_SLIPPAGE_PCT=0 (crossval_rsi) y no debe filtrarse a otros tests."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.oraculo  # el subproceso importa el proyecto y backtesting.py


def _run(workdir: Path) -> dict:
    """Solo en el subproceso: registra la estrategia y la corre por el CLI y por el adaptador."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import io
    from contextlib import redirect_stdout

    from click.testing import CliRunner
    from sqlalchemy import create_engine

    from quantagent.cli.backtest import backtest_group
    from quantagent.models import Base
    from quantagent.strategy.base import TradingSignal, TradingStrategy
    from quantagent.strategy.registry import STRATEGY_REGISTRY, register_strategy

    class ProbeStrategy(TradingStrategy):
        """Compra cuando el cierre sube respecto de la vela anterior."""

        @property
        def required_history_bars(self) -> int:
            return 2

        def generate_signal(self, kline_data, symbol, timeframe, current_price):
            if kline_data[-1]["close"] <= kline_data[-2]["close"]:
                return None
            return TradingSignal(decision="LONG", confidence=0.5, stop_loss=current_price * 0.98,
                                 take_profit=current_price * 1.03)

        def should_reevaluate(self, position, current_price):
            return False

    register_strategy(ProbeStrategy, min_bars=2, cli_alias="probe")  # la única línea de registro

    db = f"sqlite:///{workdir / 'cli.db'}"
    Base.metadata.create_all(create_engine(db))
    os.environ["DATABASE_URL"] = db
    cli = CliRunner().invoke(backtest_group, ["run", "--strategy", "probe", "--fixture", "spy-smoke"])

    import bt_adapter

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = bt_adapter.main(["--strategy", "probe", "--fixture", "spy-smoke"])
    import quantagent

    return {"package": quantagent.__file__, "registered": "ProbeStrategy" in STRATEGY_REGISTRY, "cli_exit": cli.exit_code, "cli_out": cli.output,
            "adapter_exit": code, "adapter_out": buf.getvalue()}


def _trades(output: str) -> int:
    return int(next(l for l in output.splitlines() if l.startswith("Trades:")).split()[1])


def test_estrategia_registrada_con_una_linea_corre_en_cli_y_adaptador(tmp_path):
    pytest.importorskip("backtesting")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}
    proc = subprocess.run([sys.executable, __file__, str(tmp_path)], cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    res = json.loads(proc.stdout.splitlines()[-1])

    assert Path(res["package"]).resolve().is_relative_to(ROOT)  # el árbol bajo prueba, no el instalado
    assert res["registered"]
    assert res["cli_exit"] == 0, res["cli_out"]
    assert _trades(res["cli_out"]) > 0
    assert res["adapter_exit"] == 0
    assert _trades(res["adapter_out"]) > 0


def test_alias_actuales_del_cli():
    from quantagent.strategy.registry import cli_aliases

    assert cli_aliases() == {
        "rsi": "RSIMeanReversionStrategy",
        "fifty-two-week-high": "FiftyTwoWeekHighStrategy",
        "triple-screen": "TripleScreenStrategy",
        "sma-cross": "SmaCrossStrategy",
        "momentum-12m": "Momentum12mStrategy",
        "rsi-regimen": "RsiRegimenStrategy",
        "curva-pendiente": "CurvaPendienteStrategy",
        "halloween": "HalloweenStrategy",
        "turn-of-the-month": "TurnOfTheMonthStrategy",
        "senal-externa": "SenalExterna",
        "arquetipo-a": "ArquetipoAStrategy",
    }


if __name__ == "__main__":
    print(json.dumps(_run(Path(sys.argv[1]))))
