import json
from datetime import datetime, timedelta, timezone

import pytest
from click.testing import CliRunner

from quantagent.cli.__main__ import cli


@pytest.fixture
def db_intentos(tmp_path, monkeypatch):
    from quantagent import database, settings

    url = f"sqlite:///{tmp_path / 'intentos.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setattr(settings, "DATABASE_URL", url, raising=False)
    if database._engine is not None:
        database._engine.dispose()
    database._engine = None
    database._SessionLocal = None
    database.init_db()
    from quantagent.fabrica.registro_db import registrar_intento

    row = {"desde": "2020-01-01", "hasta": "2021-01-01", "trades": 3, "pnl": 2.5, "sharpe": 0.8}
    for day in range(1, 6):
        registrar_intento("rsi", "SPY", "snap", None, 1.0, {"day": day}, row,
                          "h", "1", fecha_utc=f"2026-01-0{day}T00:00:00+00:00")
    registrar_intento("rsi", "SPY", "snap", None, 1.0, {}, row,
                      "otra", "1", fecha_utc="2026-01-03T00:00:00+00:00")
    yield
    database._engine.dispose()
    database._engine = None
    database._SessionLocal = None


@pytest.mark.parametrize("opciones,dias", [
    ([], [1, 2, 3, 4, 5]),
    (["--desde", "2026-01-02T00:00:00Z"], [2, 3, 4, 5]),
    (["--hasta", "2026-01-04T00:00:00+00:00"], [1, 2, 3, 4]),
    (["--desde", "2026-01-02T00:00:00Z", "--hasta", "2026-01-04T00:00:00Z"], [2, 3, 4]),
    (["--desde", "2027-01-01T00:00:00Z"], []),
])
def test_dry_run(db_intentos, opciones, dias):
    result = CliRunner().invoke(cli, ["intentos", "invalidar", "--hipotesis", "h", "--motivo", "bug", *opciones])
    assert result.exit_code == 0, result.output
    filas = [json.loads(linea) for linea in result.output.splitlines()[1:]]
    assert [datetime.fromisoformat(f["fecha_utc"]).day for f in filas] == dias
    from quantagent.fabrica.registro_db import intentos
    assert all(fila["estado"] == "" for fila in intentos(valida_solo=False))


@pytest.mark.parametrize("opciones,error", [
    (["--hipotesis", "ausente"], "Hipótesis inexistente"),
    *[([limite, fecha], "ISO con zona UTC") for limite in ("--desde", "--hasta")
      for fecha in ("mal", "2026-01-01", "2026-01-01T00:00:00+03:00")],
])
def test_errores(db_intentos, opciones, error):
    result = CliRunner().invoke(cli, ["intentos", "invalidar", "--hipotesis", "h", "--motivo", "bug", *opciones])
    assert result.exit_code == 1
    assert error in result.output


def test_aplicar_marca_en_db_y_excluye_filas(db_intentos):
    result = CliRunner().invoke(cli, ["intentos", "invalidar", "--hipotesis", "h", "--motivo", "bug",
                                      "--desde", "2026-01-02T00:00:00Z", "--hasta", "2026-01-04T00:00:00Z",
                                      "--aplicar"])
    assert result.exit_code == 0, result.output
    assert "Aplicado: 3 filas" in result.output
    from quantagent.backtesting.verdict import filas_validas
    from quantagent.fabrica.registro_db import intentos
    todas = intentos(valida_solo=False)
    assert sum(f["estado"] == "invalida_bug" for f in todas) == 3
    assert len(filas_validas(todas)) == 3


@pytest.mark.parametrize("desde", ["2026-01-02T03:04:05Z", "2026-01-02T03:04:05+00:00"])
def test_epoca_limpia(db_intentos, tmp_path, monkeypatch, desde):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    antes = datetime.now(timezone.utc)
    result = CliRunner().invoke(cli, ["intentos", "epoca-limpia", "--desde", desde,
                                     "--motivo", "Corrección de cálculo: época nueva"])
    despues = datetime.now(timezone.utc)
    assert result.exit_code == 0, result.output
    marca = json.loads((tmp_path / "epoca_limpia.json").read_text(encoding="utf-8"))
    assert set(marca) == {"desde_utc", "motivo", "creada_utc"}
    creada = datetime.fromisoformat(marca["creada_utc"])
    assert antes <= creada <= despues


@pytest.mark.parametrize("desde", ["mal", "2026-01-01", "2026-01-01T00:00:00", "2026-01-01T00:00:00+03:00"])
def test_epoca_limpia_fecha_invalida(db_intentos, desde):
    result = CliRunner().invoke(cli, ["intentos", "epoca-limpia", "--desde", desde, "--motivo", "bug"])
    assert result.exit_code == 1
    assert "ISO con zona UTC" in result.output


def test_registrar_intento_guarda_el_intento_en_la_base_sin_archivos(db_intentos, tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    from quantagent.cli.sweep import registrar_intento

    row = {"desde": "2020-01-01", "hasta": "2021-01-01", "trades": 3, "pnl": 2.5, "sharpe": 0.8}
    registrar_intento("rsi", "SPY", "snap-1", None, 1.0, {"period": 14}, row, hipotesis="H1", etapa="1")
    from quantagent.fabrica.registro_db import intentos
    assert intentos()[-1]["snapshot"] == "snap-1"
    assert intentos()[-1]["hipotesis"] == "H1"
    assert not (tmp_path / "registro-hipotesis.jsonl").exists()
    assert not (tmp_path / "intentos.csv").exists()
