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


@pytest.mark.parametrize("con_estado", [True, False])
def test_aplicar_conserva_registro_y_backup(registro, con_estado):
    with registro.open(newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    filas[0]["estado"] = "otro_estado"
    campos = ["hipotesis", "fecha_utc", "detalle"] + (["estado"] if con_estado else [])
    for i, fila in enumerate(filas):
        fila["detalle"] = f'dato {i}, con "comillas" y ñ\nsiguiente línea'
        if not con_estado:
            del fila["estado"]
    with registro.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)
    original = registro.read_bytes()
    result = CliRunner().invoke(cli, ["intentos", "invalidar", "--hipotesis", "h", "--motivo", "bug",
                                     "--desde", "2026-01-02T00:00:00Z", "--hasta", "2026-01-04T00:00:00Z",
                                     "--aplicar"])
    assert result.exit_code == 0, result.output
    assert "Aplicado: 3 filas" in result.output
    backups = list(registro.parent.glob("intentos.csv.bak-*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    with registro.open(newline="", encoding="utf-8") as archivo:
        reader = csv.DictReader(archivo)
        nuevas = list(reader)
        assert reader.fieldnames == campos + ([] if con_estado else ["estado"])
    esperadas = [dict(fila, estado="invalida_bug" if i in (1, 2, 3) else fila.get("estado", ""))
                 for i, fila in enumerate(filas)]
    assert nuevas == esperadas
