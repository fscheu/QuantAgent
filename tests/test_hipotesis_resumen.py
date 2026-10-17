import csv

from click.testing import CliRunner

from quantagent.cli.hipotesis import hipotesis_group
from quantagent.fabrica.estado import COLUMNAS


def test_resumen_ultimos_fecha_causas_e_intentos(tmp_path, monkeypatch):
    monkeypatch.setattr("quantagent.fabrica.estado.snapshot_root", lambda: tmp_path)
    filas = [("2026-01-01T00:00:00+00:00", "vieja", "0", "CORTA", "sin historia"),
             ("2026-02-01T00:00:00+00:00", "h1", "0", "PASA", "aceptada"),
             ("2026-02-02T00:00:00+00:00", "h1", "3", "CORTA", "SPY: DSR < 0.95"),
             ("2026-02-02T00:00:00+00:00", "h2", "3", "CORTA", "XLK: DSR < 0.95"),
             ("2026-02-02T00:00:00+00:00", "h3", "2", "ALARMA", "filtro optimista")]
    for nombre, columnas, datos in [("hipotesis.csv", COLUMNAS, filas),
                                    ("intentos.csv", ("fecha_utc", "hipotesis"),
                                     [(r[0], r[1]) for r in filas])]:
        with (tmp_path / nombre).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(columnas)
            w.writerows(datos)
    runner = CliRunner()
    completo = runner.invoke(hipotesis_group, ["resumen"])
    assert completo.exit_code == 0
    assert "| 0 | CORTA | 1 |" in completo.output
    assert "| 3 | CORTA | 2 |" in completo.output
    assert "| 2 | ALARMA | 1 |" in completo.output
    assert "PASA" not in completo.output
    assert "DSR < 0.95: 2" in completo.output
    assert "Intentos: 5" in completo.output
    out = tmp_path / "resumen.md"
    filtrado = runner.invoke(hipotesis_group, ["resumen", "--desde", "2026-02-01", "--out", str(out)])
    assert filtrado.exit_code == 0
    assert "sin historia" not in filtrado.output
    assert "Intentos: 4" in filtrado.output
    assert out.read_text() == filtrado.output
