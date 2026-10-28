import csv
import json

import pytest
from click.testing import CliRunner

from quantagent.cli.__main__ import cli


@pytest.fixture
def registro(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    path = tmp_path / "intentos.csv"
    with path.open("w", newline="") as archivo:
        writer = csv.writer(archivo)
        writer.writerow(["hipotesis", "fecha_utc", "estado"])
        writer.writerows([("h", f"2026-01-0{dia}T00:00:00+00:00", "") for dia in range(1, 6)])
        writer.writerow(["otra", "2026-01-03T00:00:00Z", "invalida_bug"])
    return path


@pytest.mark.parametrize("opciones,dias", [
    ([], [1, 2, 3, 4, 5]),
    (["--desde", "2026-01-02T00:00:00Z"], [2, 3, 4, 5]),
    (["--hasta", "2026-01-04T00:00:00+00:00"], [1, 2, 3, 4]),
    (["--desde", "2026-01-02T00:00:00Z", "--hasta", "2026-01-04T00:00:00Z"], [2, 3, 4]),
    (["--desde", "2027-01-01T00:00:00Z"], []),
])
def test_dry_run(registro, opciones, dias):
    original = registro.read_bytes()
    result = CliRunner().invoke(cli, ["intentos", "invalidar", "--hipotesis", "h", "--motivo", "bug", *opciones])
    assert result.exit_code == 0, result.output
    filas = [json.loads(linea) for linea in result.output.splitlines()[1:]]
    assert filas == [{"hipotesis": "h", "fecha_utc": f"2026-01-0{dia}T00:00:00+00:00", "estado": ""} for dia in dias]
    assert registro.read_bytes() == original
    assert list(registro.parent.iterdir()) == [registro]


@pytest.mark.parametrize("opciones,error", [
    (["--hipotesis", "ausente"], "Hipótesis inexistente"),
    *[([limite, fecha], "ISO con zona UTC") for limite in ("--desde", "--hasta")
      for fecha in ("mal", "2026-01-01", "2026-01-01T00:00:00+03:00")],
])
def test_errores(registro, opciones, error):
    result = CliRunner().invoke(cli, ["intentos", "invalidar", "--hipotesis", "h", "--motivo", "bug", *opciones])
    assert result.exit_code == 1
    assert error in result.output
