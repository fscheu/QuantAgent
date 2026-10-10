"""`backtest run|verify --costos` (QuantAgent-dgi), con el motor real sobre spy-smoke.

El PnL esperado sale de scripts/recalc_metrics.py (no importa nada de quantagent) aplicado al
CSV de trades con la comisión del perfil, nunca del motor.
"""

import importlib.util
import re
from pathlib import Path

import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine

import quantagent.database as database
from quantagent import settings
from quantagent.cli.backtest import backtest_group
from quantagent.models import Base

_spec = importlib.util.spec_from_file_location(
    "recalc_metrics", Path(__file__).resolve().parents[1] / "scripts" / "recalc_metrics.py"
)
recalc_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recalc_metrics)

BASE = ["--strategy", "rsi", "--fixture", "spy-smoke"]


@pytest.fixture
def run_cli(monkeypatch, tmp_path):
    """Corre `backtest <args>` sobre una base SQLite nueva por llamada."""
    original_url = settings.DATABASE_URL
    calls = []

    def _run(*args):
        db_url = f"sqlite:///{tmp_path / f'db{len(calls)}.db'}"
        calls.append(db_url)
        monkeypatch.setenv("DATABASE_URL", db_url)
        Base.metadata.create_all(create_engine(db_url, connect_args={"check_same_thread": False}))
        return CliRunner().invoke(backtest_group, list(args))

    try:
        yield _run
    finally:
        settings.DATABASE_URL = original_url
        database._engine = None
        database._SessionLocal = None


def _total_pnl(output):
    return float(re.search(r"^Total PnL: (-?[\d.]+)$", output, re.M).group(1))


@pytest.mark.parametrize("perfil", sorted(settings.COSTOS_PERFILES))
def test_perfil_fija_costos_y_pnl_coincide_con_recalculo(run_cli, tmp_path, perfil):
    costos = settings.COSTOS_PERFILES[perfil]
    out = tmp_path / "trades.csv"
    result = run_cli("run", *BASE, "--costos", perfil, "--out", str(out))
    assert result.exit_code == 0, result.output

    assert f"Costos: {perfil}" in result.output
    assert f"Slippage: {costos['slippage_pct'] * 100:.2f}% por lado" in result.output
    assert f"Comisión: {costos['commission_pct'] * 100:.2f}% por lado" in result.output

    recalc = recalc_metrics.recalc(str(out), costos["commission_pct"])
    assert recalc["trades"] > 0
    assert recalc["matched"] == recalc["trades"]  # cada PnL del CSV se reproduce con esa comisión
    assert _total_pnl(result.output) == pytest.approx(recalc["total_pnl"], abs=0.01)


def test_diferencia_entre_perfiles_coincide_con_recalculo(run_cli, tmp_path):
    pnl = {}
    for perfil in settings.COSTOS_PERFILES:
        out = tmp_path / f"{perfil}.csv"
        result = run_cli("run", *BASE, "--costos", perfil, "--out", str(out))
        assert result.exit_code == 0, result.output
        pnl[perfil] = (_total_pnl(result.output), recalc_metrics.recalc(str(out), settings.COSTOS_PERFILES[perfil]["commission_pct"])["total_pnl"])

    for a, b in [("etf", "cripto"), ("etf", "etf-conservador")]:
        assert pnl[a][0] - pnl[b][0] > 0  # más comisión, menos PnL
        assert pnl[a][0] - pnl[b][0] == pytest.approx(pnl[a][1] - pnl[b][1], abs=0.01)


def test_costos_con_commission_pct_se_rechaza(run_cli):
    for comando in ("run", "verify"):
        result = run_cli(comando, *BASE, "--costos", "etf", "--commission-pct", "0.001")
        assert result.exit_code != 0
        assert "--costos" in result.output and "--commission-pct" in result.output
        assert "Trades:" not in result.output


def test_sin_costos_la_salida_es_la_de_antes(run_cli):
    sin = run_cli("run", *BASE)
    con_etf = run_cli("run", *BASE, "--costos", "etf")
    assert sin.exit_code == 0 and con_etf.exit_code == 0
    assert "Costos:" not in sin.output
    # etf coincide con los defaults (slippage 0.0005, comisión 0): solo se agrega la línea Costos.
    assert con_etf.output.replace("Costos: etf\n", "") == sin.output


def test_verify_con_costos_es_reproducible(run_cli):
    result = run_cli("verify", *BASE, "--costos", "cripto")
    assert result.exit_code == 0, result.output
    assert "OK reproducible" in result.output
