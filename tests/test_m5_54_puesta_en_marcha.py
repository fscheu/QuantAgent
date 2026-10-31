from types import SimpleNamespace

from click.testing import CliRunner

from quantagent.cli.__main__ import cli


def _m5_fakes(monkeypatch):
    monkeypatch.setattr("quantagent.trading.bucket_m5.cargar_credenciales", lambda: None)
    monkeypatch.setattr("quantagent.cli.reconcile.build_source", lambda name: SimpleNamespace(name=name))
    monkeypatch.setattr("quantagent.cli.reconcile.DataProvider", lambda session, source: (session, source))
    monkeypatch.setattr(
        "quantagent.cli.reconcile.StrategyAssembler.build_components",
        lambda resolved, db_session, build_graph: SimpleNamespace(broker=SimpleNamespace()),
    )
    monkeypatch.setattr("quantagent.cli.reconcile.StrategyAssembler.from_profiles", lambda **kwargs: object())
    monkeypatch.setattr("quantagent.cli.reconcile._config_m5", lambda resolved: resolved)


def test_m5_correr_diario_usa_db_y_as_of(tmp_path, monkeypatch):
    _m5_fakes(monkeypatch)
    seen = {}

    def run(as_of, *, data_provider, db, broker):
        seen.update(as_of=as_of, data_provider=data_provider, db=db, broker=broker)
        return [SimpleNamespace(order_id=1)]

    monkeypatch.setattr("quantagent.cli.reconcile.correr_diario", run)
    db_path = tmp_path / "m5.db"
    result = CliRunner().invoke(cli, ["trading", "m5-correr", "--modo", "diario",
                                      "--as-of", "2026-11-02T13:00:00+0000", "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert seen["as_of"].isoformat() == "2026-11-02T13:00:00+00:00"
    assert seen["db"].bind.url.database == str(db_path)
    assert "1 intents, 1 órdenes" in result.output


def test_m5_correr_semanal_invoca_pipeline(tmp_path, monkeypatch):
    _m5_fakes(monkeypatch)
    seen = {}

    def run(as_of, *, data_provider, db):
        seen.update(as_of=as_of, db=db)
        return {"estado": "risk-on"}

    monkeypatch.setattr("quantagent.cli.reconcile.correr_semanal", run)
    result = CliRunner().invoke(cli, ["trading", "m5-correr", "--modo", "semanal",
                                      "--db", str(tmp_path / "m5.db")])
    assert result.exit_code == 0, result.output
    assert seen["db"].bind.url.database == str(tmp_path / "m5.db")
    assert "régimen risk-on" in result.output


def test_m5_correr_excepcion_sale_con_uno(tmp_path, monkeypatch):
    _m5_fakes(monkeypatch)
    monkeypatch.setattr("quantagent.cli.reconcile.correr_diario", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("falló")))
    result = CliRunner().invoke(cli, ["trading", "m5-correr", "--modo", "diario", "--db", str(tmp_path / "m5.db")])
    assert result.exit_code == 1
    assert "RuntimeError: falló" in result.output


def test_pre_apertura_m5_usa_db_y_falla_si_sin_stop(tmp_path, monkeypatch):
    _m5_fakes(monkeypatch)
    seen = {}

    def preopen(session, broker):
        seen["db"] = session.bind.url.database
        return {"reenviados": [], "entradas_vencidas": [], "avisos": [], "sin_stop": ["SPY"]}

    monkeypatch.setattr("quantagent.cli.reconcile.correr_pre_apertura", preopen)
    db_path = tmp_path / "m5.db"
    result = CliRunner().invoke(cli, ["trading", "pre-apertura", "--bucket", "m5", "--db", str(db_path)])
    assert result.exit_code == 1
    assert seen["db"] == str(db_path)
    assert "ALARMA: SPY sin stop" in result.output
