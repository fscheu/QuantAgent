"""`backtest run --snapshot` (V07). Usa un snapshot armado en tmp_path a partir de un fixture; sin red ni snapshot real."""

import csv
import math
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
    monkeypatch.setenv("QUANTAGENT_PRUEBA_DESDE", "2030-01-01")
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
    assert len(by_snapshot.stdout.strip().splitlines()) == 14
    assert _metrics(by_snapshot.stdout) == ("Trades: 6", "Total PnL: -249.35")


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
    monkeypatch.setenv("QUANTAGENT_PRUEBA_DESDE", desde)
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
    assert _run(snap_env[0], "--fixture", DAILY, "--abrir-prueba").exit_code == 2


def _verify(runner, *args):
    return runner.invoke(backtest_group, ["verify", "--strategy", "fifty-two-week-high", *args])


def test_verify_snapshot_imprime_ok_reproducible(snap_env):
    runner, _ = snap_env
    write_snapshot("t", {"SPY": _daily_frame()})
    result = _verify(runner, "--snapshot", "t", "--symbol", "SPY")
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "OK reproducible"


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


def _compare(runner, *args):
    return runner.invoke(backtest_group, ["compare", *args])


def test_compare_snapshot_arma_csv_e_imprime_tabla(snap_env, tmp_path):
    from quantagent.backtesting.export import trades_to_csv
    from quantagent.backtesting.fixtures import snapshot_metadata
    from quantagent.cli.backtest import _engine_config, correr_par

    runner, _ = snap_env
    spy_frame = _daily_frame()
    iwm_frame = _daily_frame().iloc[20:].copy()
    iwm_frame[["open", "high", "low", "close", "adj_close"]] += 50.0
    write_snapshot("t", {"SPY": spy_frame, "IWM": iwm_frame})
    out_csv = tmp_path / "cmp.csv"
    trades_dir = tmp_path / "trades"
    res = _compare(
        runner,
        "--strategies", "rsi,fifty-two-week-high",
        "--symbols", "SPY,IWM",
        "--snapshot", "t",
        "--to", "2024-09-30",
        "--out", str(out_csv),
        "--trades-dir", str(trades_dir),
    )
    assert res.exit_code == 0, res.output
    with out_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert (trades_dir / "rsi-SPY.csv").exists()
    assert (trades_dir / "fifty-two-week-high-IWM.csv").exists()

    spy_rows = {r["estrategia"]: r for r in rows if r["activo"] == "SPY"}
    iwm_rows = {r["estrategia"]: r for r in rows if r["activo"] == "IWM"}
    for strat in ["rsi", "fifty-two-week-high"]:
        assert {k: v for k, v in spy_rows[strat].items() if k != "activo"} != {
            k: v for k, v in iwm_rows[strat].items() if k != "activo"
        }
        assert spy_rows[strat]["desde"] != iwm_rows[strat]["desde"]
        assert spy_rows[strat]["bh_pnl"] != iwm_rows[strat]["bh_pnl"]

    to_dt = datetime(2024, 9, 30)
    redondeos = {
        "win_rate": 4, "sharpe": 2, "pnl": 2, "max_drawdown": 6,
        "bh_pnl": 2, "bh_sharpe": 2, "bh_max_drawdown": 6,
        "retorno_anualizado": 4, "exceso_sobre_referencia": 4,
        "tiempo_en_mercado": 4, "capital_invertido_medio": 4,
        "sharpe_tasa_0": 2,
    }
    for row in rows:
        meta = snapshot_metadata("t", row["activo"], None, to_dt)
        r = correr_par(
            row["estrategia"], meta, _engine_config(None, None, 1.0), None,
            None, "t", row["activo"], None, to_dt, con_filas=True,
        )
        assert row["desde"] == f"{meta.start_date:%Y-%m-%d}"
        assert row["hasta"] == f"{meta.end_date:%Y-%m-%d}"
        assert int(row["trades"]) == r["trades"]
        for col, dec in redondeos.items():
            val = r[col]
            if val is None:
                assert row[col] == ""
            else:
                assert float(row[col]) == round(val, dec)
        pf = r["profit_factor"]
        if pf is None or math.isinf(pf):
            assert row["profit_factor"] == ""
        else:
            assert float(row["profit_factor"]) == round(pf, 2)

        trades_file = trades_dir / f"{row['estrategia']}-{row['activo']}.csv"
        assert trades_file.read_bytes().decode() == trades_to_csv(r["filas"])


def test_compare_reserva_bloquea_y_no_escribe_csv(snap_env, monkeypatch, tmp_path):
    runner, _, log = _con_reserva(snap_env, monkeypatch, tmp_path)
    out_csv = tmp_path / "cmp.csv"
    res = _compare(
        runner,
        "--strategies", "rsi",
        "--symbols", "SPY",
        "--snapshot", "t",
        "--to", "2024-09-30",
        "--out", str(out_csv),
    )
    assert res.exit_code == 1 and "reserva" in res.output
    assert not out_csv.exists()


LOG_PRUEBA = "prueba-accesos.log"


def _multiyear_frame() -> pd.DataFrame:
    df = pd.concat([_daily_frame(), _daily_frame()]).iloc[:600].copy()
    df.index = pd.date_range("2018-01-01", periods=len(df), freq="B")
    return df


def test_sin_to_no_lee_desde_prueba_y_stderr_trae_aviso(snap_env, monkeypatch, tmp_path):
    runner, engine = snap_env
    monkeypatch.delenv("QUANTAGENT_PRUEBA_DESDE", raising=False)
    monkeypatch.delenv("QUANTAGENT_RESERVA_DESDE", raising=False)
    write_snapshot("p", {"SPY": _multiyear_frame()})

    res_run = _run(runner, "--snapshot", "p", "--symbol", "SPY")
    assert res_run.exit_code == 0, res_run.output
    assert "Aviso: sin --to" in res_run.stderr
    assert "2018-12-31" in res_run.stderr
    with Session(engine) as session:
        last = max(row.timestamp for row in session.query(MarketData))
    assert last == datetime(2018, 12, 31)

    out_csv = tmp_path / "cmp_p.csv"
    res_cmp = _compare(
        runner,
        "--strategies", "fifty-two-week-high",
        "--symbols", "SPY",
        "--snapshot", "p",
        "--out", str(out_csv),
    )
    assert res_cmp.exit_code == 0, res_cmp.output
    assert "Aviso: sin --to" in res_cmp.stderr
    assert "2018-12-31" in res_cmp.stderr
    with out_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["hasta"] == "2018-12-31"


def test_to_en_prueba_sin_flag_sale_con_1_y_con_flag_registra(snap_env, monkeypatch, tmp_path):
    runner, engine = snap_env
    monkeypatch.delenv("QUANTAGENT_PRUEBA_DESDE", raising=False)
    monkeypatch.delenv("QUANTAGENT_RESERVA_DESDE", raising=False)
    write_snapshot("p", {"SPY": _multiyear_frame()})
    log = tmp_path / "snaps" / LOG_PRUEBA

    blocked = _run(runner, "--snapshot", "p", "--symbol", "SPY", "--to", "2020-01-01")
    assert blocked.exit_code == 1
    assert "2020-01-01" in blocked.output and "prueba" in blocked.output
    assert not log.exists()

    opened = _run(runner, "--snapshot", "p", "--symbol", "SPY", "--to", "2020-01-01", "--abrir-prueba")
    assert opened.exit_code == 0, opened.output
    assert log.exists()
    lines = log.read_text().splitlines()
    assert len(lines) == 1
    fecha, comando = lines[0].split("\t")
    assert abs((datetime.now(timezone.utc) - datetime.fromisoformat(fecha)).total_seconds()) < 300
    assert "--snapshot p --symbol SPY --to 2020-01-01 --abrir-prueba" in comando


def test_salida_estandar_sin_aviso_cuando_se_pasa_to(snap_env, monkeypatch):
    runner, _ = snap_env
    monkeypatch.delenv("QUANTAGENT_PRUEBA_DESDE", raising=False)
    monkeypatch.delenv("QUANTAGENT_RESERVA_DESDE", raising=False)
    write_snapshot("p", {"SPY": _multiyear_frame()})

    res = _run(runner, "--snapshot", "p", "--symbol", "SPY", "--to", "2018-12-31")
    assert res.exit_code == 0, res.output
    assert "Aviso: sin --to" not in res.stderr
    assert "Aviso: sin --to" not in res.stdout


def test_sin_to_con_reserva_fijada_sin_prueba_termina_en_2018(snap_env, monkeypatch):
    runner, engine = snap_env
    monkeypatch.setenv("QUANTAGENT_RESERVA_DESDE", "2024-01-01")
    monkeypatch.delenv("QUANTAGENT_PRUEBA_DESDE", raising=False)
    write_snapshot("p", {"SPY": _multiyear_frame()})

    res = _run(runner, "--snapshot", "p", "--symbol", "SPY")
    assert res.exit_code == 0, res.output
    assert "Aviso: sin --to" in res.stderr
    assert "2018-12-31" in res.stderr
    with Session(engine) as session:
        last = max(row.timestamp for row in session.query(MarketData))
    assert last == datetime(2018, 12, 31)


def test_sin_to_con_prueba_tardia_y_reserva_temprana_no_lee_reserva(snap_env, monkeypatch, tmp_path):
    runner, engine = snap_env
    monkeypatch.setenv("QUANTAGENT_PRUEBA_DESDE", "2030-01-01")
    monkeypatch.setenv("QUANTAGENT_RESERVA_DESDE", "2019-06-01")
    write_snapshot("p", {"SPY": _multiyear_frame()})
    log = tmp_path / "snaps" / LOG

    res = _run(runner, "--snapshot", "p", "--symbol", "SPY")
    assert res.exit_code == 0, res.output
    assert "Aviso: sin --to" in res.stderr
    assert "2019-05-31" in res.stderr
    with Session(engine) as session:
        velas_reserva = session.query(MarketData).filter(MarketData.timestamp >= datetime(2019, 6, 1)).count()
        last = max(row.timestamp for row in session.query(MarketData))
    assert velas_reserva == 0
    assert last == datetime(2019, 5, 31)
    assert not log.exists()


