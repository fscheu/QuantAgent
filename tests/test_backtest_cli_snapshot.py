"""`backtest run --snapshot` (V07). Usa un snapshot armado en tmp_path a partir de un fixture; sin red ni snapshot real."""

import csv
from datetime import datetime, timezone

import pandas as pd
import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import quantagent.database as database
from quantagent import settings
from quantagent.backtesting.fixtures import FIXTURES_DIR
from quantagent.cli.backtest import backtest_group
from quantagent.data.snapshot import write_snapshot
from quantagent.models import Base, MarketData

DAILY = "spy-2y-1d"


def _daily_frame(factor: float = 1.0) -> pd.DataFrame:
    """Las velas de `spy-2y-1d.csv` en el formato del snapshot; `adj_close = close * factor` (dividendo)."""
    with (FIXTURES_DIR / f"{DAILY}.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    df = pd.DataFrame(
        {
            "open": [float(r["open"]) for r in rows],
            "high": [float(r["high"]) for r in rows],
            "low": [float(r["low"]) for r in rows],
            "close": [float(r["close"]) for r in rows],
            "volume": [int(r["volume"]) for r in rows],
        },
        index=pd.DatetimeIndex([r["timestamp"] for r in rows], name="timestamp"),
    )
    df.insert(4, "adj_close", df["close"] * factor)
    return df


@pytest.fixture
def snap_env(monkeypatch, tmp_path):
    """Base SQLite aislada, `$QUANTAGENT_SNAPSHOT_DIR` en tmp_path y un runner."""
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path / "snaps"))
    monkeypatch.setenv("QUANTAGENT_RESERVA_DESDE", "2030-01-01")  # los datos del fixture son de 2024-2025
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    original_database_url = settings.DATABASE_URL
    try:
        yield CliRunner(), engine
    finally:
        settings.DATABASE_URL = original_database_url
        database._engine = None
        database._SessionLocal = None


def _run(runner, *args):
    return runner.invoke(backtest_group, ["run", "--strategy", "fifty-two-week-high", *args])


def _metrics(output: str) -> tuple[str, str]:
    lines = output.strip().splitlines()
    return lines[0], next(line for line in lines if line.startswith("Total PnL"))


def test_snapshot_da_lo_mismo_que_el_fixture_del_que_sale(snap_env):
    runner, _ = snap_env
    write_snapshot("t", {"SPY": _daily_frame()})
    by_snapshot = _run(runner, "--snapshot", "t", "--symbol", "SPY")
    assert by_snapshot.exit_code == 0, by_snapshot.output
    assert len(by_snapshot.output.strip().splitlines()) == 6
    assert _metrics(by_snapshot.output) == ("Trades: 6", "Total PnL: -249.35")


def test_carga_velas_ajustadas_y_solo_las_del_rango(snap_env):
    runner, engine = snap_env
    factor = 50 / 51
    write_snapshot("t", {"SPY": _daily_frame(factor)})
    result = _run(runner, "--snapshot", "t", "--symbol", "spy", "--from", "2024-03-01", "--to", "2024-09-30")
    assert result.exit_code == 0, result.output

    with Session(engine) as session:
        rows = session.query(MarketData).order_by(MarketData.timestamp).all()
    expected = _daily_frame(factor).loc["2024-03-01":"2024-09-30"]
    assert len(rows) == len(expected)
    assert rows[0].timestamp == datetime(2024, 3, 1) and rows[-1].timestamp == datetime(2024, 9, 30)
    assert float(rows[0].close) == pytest.approx(expected["close"].iloc[0] * factor, rel=1e-6)
    assert float(rows[0].close) != pytest.approx(expected["close"].iloc[0], rel=1e-3)  # ajustado, no el guardado


def test_fixture_y_snapshot_son_excluyentes_y_snapshot_pide_symbol(snap_env):
    runner, _ = snap_env
    write_snapshot("t", {"SPY": _daily_frame()})
    both = _run(runner, "--fixture", DAILY, "--snapshot", "t", "--symbol", "SPY")
    assert both.exit_code == 2 and "exactamente uno" in both.output
    assert _run(runner).exit_code == 2
    no_symbol = _run(runner, "--snapshot", "t")
    assert no_symbol.exit_code == 2 and "--symbol" in no_symbol.output
    assert _run(runner, "--fixture", DAILY, "--symbol", "SPY").exit_code == 2


def test_simbolo_o_rango_inexistente_sale_con_1_sin_cargar_nada(snap_env):
    runner, engine = snap_env
    write_snapshot("t", {"SPY": _daily_frame()})
    unknown = _run(runner, "--snapshot", "t", "--symbol", "QQQ")
    assert unknown.exit_code == 1 and "QQQ" in unknown.output
    empty = _run(runner, "--snapshot", "t", "--symbol", "SPY", "--from", "2030-01-01")
    assert empty.exit_code == 1 and "rango" in empty.output
    with Session(engine) as session:
        assert session.query(MarketData).count() == 0


def test_base_con_otros_datos_en_el_rango_aborta_como_con_fixtures(snap_env):
    runner, engine = snap_env
    write_snapshot("t", {"SPY": _daily_frame()})
    with Session(engine) as session:
        session.add(MarketData(symbol="SPY", timeframe="1d", timestamp=datetime(2024, 6, 3), open=1, high=1, low=1, close=1, volume=1))
        session.commit()
    result = _run(runner, "--snapshot", "t", "--symbol", "SPY")
    assert result.exit_code == 1
    assert "otros datos" in result.output and "snapshot" in result.output


RESERVA = "2024-07-01"
LOG = "reserva-accesos.log"


def _con_reserva(snap_env, monkeypatch, tmp_path, desde=RESERVA):
    monkeypatch.setenv("QUANTAGENT_RESERVA_DESDE", desde)
    write_snapshot("t", {"SPY": _daily_frame()})
    return snap_env[0], snap_env[1], tmp_path / "snaps" / LOG


def test_sin_to_el_rango_termina_el_dia_antes_de_la_reserva(snap_env, monkeypatch, tmp_path):
    runner, engine, log = _con_reserva(snap_env, monkeypatch, tmp_path)
    result = _run(runner, "--snapshot", "t", "--symbol", "SPY")
    assert result.exit_code == 0, result.output
    with Session(engine) as session:
        last = max(row.timestamp for row in session.query(MarketData))
    assert last == _daily_frame().loc[:"2024-06-30"].index.max().to_pydatetime()
    assert not log.exists()


def test_to_en_la_reserva_sin_el_flag_sale_con_1_y_no_toca_el_log(snap_env, monkeypatch, tmp_path):
    runner, engine, log = _con_reserva(snap_env, monkeypatch, tmp_path)
    log.write_text("antes\n")
    result = _run(runner, "--snapshot", "t", "--symbol", "SPY", "--to", "2024-09-30")
    assert result.exit_code == 1
    assert "2024-09-30" in result.output and RESERVA in result.output
    assert log.read_text() == "antes\n"
    with Session(engine) as session:
        assert session.query(MarketData).count() == 0


def test_candado_usa_2023_01_01_si_no_hay_variable(snap_env, monkeypatch, tmp_path):
    runner, _, log = _con_reserva(snap_env, monkeypatch, tmp_path)
    monkeypatch.delenv("QUANTAGENT_RESERVA_DESDE")
    result = _run(runner, "--snapshot", "t", "--symbol", "SPY", "--to", "2023-06-30")
    assert result.exit_code == 1 and "2023-01-01" in result.output
    assert not log.exists()


def test_abrir_reserva_corre_y_suma_una_linea_al_log(snap_env, monkeypatch, tmp_path):
    runner, _, log = _con_reserva(snap_env, monkeypatch, tmp_path)
    log.write_text("antes\n")
    result = _run(runner, "--snapshot", "t", "--symbol", "SPY", "--to", "2024-09-30", "--abrir-reserva")
    assert result.exit_code == 0, result.output
    lines = log.read_text().splitlines()
    assert len(lines) == 2 and lines[0] == "antes"
    fecha, comando = lines[1].split("\t")
    assert abs((datetime.now(timezone.utc) - datetime.fromisoformat(fecha)).total_seconds()) < 300
    assert "--snapshot t --symbol SPY --to 2024-09-30 --abrir-reserva" in comando


def test_abrir_reserva_con_to_anterior_no_es_una_apertura(snap_env, monkeypatch, tmp_path):
    runner, _, log = _con_reserva(snap_env, monkeypatch, tmp_path)
    result = _run(runner, "--snapshot", "t", "--symbol", "SPY", "--to", "2024-06-28", "--abrir-reserva")
    assert result.exit_code == 0, result.output
    assert not log.exists()


def test_abrir_reserva_con_fixture_es_error_de_uso(snap_env):
    assert _run(snap_env[0], "--fixture", DAILY, "--abrir-reserva").exit_code == 2


def _verify(runner, *args):
    return runner.invoke(backtest_group, ["verify", "--strategy", "fifty-two-week-high", *args])


def test_verify_snapshot_imprime_ok_reproducible(snap_env):
    runner, _ = snap_env
    write_snapshot("t", {"SPY": _daily_frame()})
    result = _verify(runner, "--snapshot", "t", "--symbol", "SPY")
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "OK reproducible"


def test_verify_snapshot_respeta_reserva_y_registra_apertura(snap_env, monkeypatch, tmp_path):
    runner, _, log = _con_reserva(snap_env, monkeypatch, tmp_path)
    blocked = _verify(runner, "--snapshot", "t", "--symbol", "SPY", "--to", "2024-09-30")
    assert blocked.exit_code == 1 and "reserva" in blocked.output
    assert not log.exists()

    opened = _verify(runner, "--snapshot", "t", "--symbol", "SPY", "--to", "2024-09-30", "--abrir-reserva")
    assert opened.exit_code == 0, opened.output
    assert opened.output.strip() == "OK reproducible"
    assert log.exists()
    lines = log.read_text().splitlines()
    assert len(lines) == 1
    assert "backtest verify --strategy fifty-two-week-high --snapshot t --symbol SPY --to 2024-09-30 --abrir-reserva" in lines[0]


def test_verify_fixture_y_snapshot_son_excluyentes(snap_env):
    runner, _ = snap_env
    write_snapshot("t", {"SPY": _daily_frame()})
    both = _verify(runner, "--fixture", DAILY, "--snapshot", "t", "--symbol", "SPY")
    assert both.exit_code == 2 and "exactamente uno" in both.output
    assert _verify(runner).exit_code == 2
