import csv
import json
from datetime import datetime, timedelta, timezone

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


@pytest.mark.parametrize("desde", ["2026-01-02T03:04:05Z", "2026-01-02T03:04:05+00:00"])
def test_epoca_limpia(registro, desde):
    original = registro.read_bytes()
    antes = datetime.now(timezone.utc)
    result = CliRunner().invoke(cli, ["intentos", "epoca-limpia", "--desde", desde,
                                     "--motivo", "Corrección de cálculo: época nueva"])
    despues = datetime.now(timezone.utc)
    assert result.exit_code == 0, result.output
    path = registro.parent / "epoca_limpia.json"
    marca = json.loads(path.read_text(encoding="utf-8"))
    assert set(marca) == {"desde_utc", "motivo", "creada_utc"}
    assert datetime.fromisoformat(marca["desde_utc"]) == datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert marca["motivo"] == "Corrección de cálculo: época nueva"
    creada = datetime.fromisoformat(marca["creada_utc"])
    assert creada.utcoffset() == timedelta(0)
    assert antes <= creada <= despues
    assert registro.read_bytes() == original
    assert set(registro.parent.iterdir()) == {registro, path}


@pytest.mark.parametrize("desde", ["mal", "2026-01-01", "2026-01-01T00:00:00", "2026-01-01T00:00:00+03:00"])
def test_epoca_limpia_fecha_invalida(registro, desde):
    original = registro.read_bytes()
    result = CliRunner().invoke(cli, ["intentos", "epoca-limpia", "--desde", desde, "--motivo", "bug"])
    assert result.exit_code == 1
    assert "ISO con zona UTC" in result.output
    assert registro.read_bytes() == original
    assert list(registro.parent.iterdir()) == [registro]
