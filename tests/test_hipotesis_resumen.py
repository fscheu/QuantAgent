import csv
from datetime import datetime

from click.testing import CliRunner

from quantagent.cli.hipotesis import hipotesis_group
from quantagent.fabrica.estado import COLUMNAS
from quantagent.fabrica.estado import resumen_tanda


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


def test_resumen_causa_en_varios_activos_cuenta_una_vez_por_hipotesis(tmp_path, monkeypatch):
    monkeypatch.setattr("quantagent.fabrica.estado.snapshot_root", lambda: tmp_path)
    filas = [
        ("2026-02-01T00:00:00+00:00", "h1", "3", "CORTA", "SPY: DSR < 0.95; XLK: DSR < 0.95; QQQ: DSR < 0.95"),
        ("2026-02-01T00:00:00+00:00", "h2", "3", "CORTA", "EFA: DSR < 0.95; IWM: DSR < 0.95"),
    ]
    with (tmp_path / "hipotesis.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNAS)
        w.writerows(filas)
    with (tmp_path / "intentos.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("fecha_utc", "hipotesis"))
        w.writerows([(r[0], r[1]) for r in filas])
    runner = CliRunner()
    completo = runner.invoke(hipotesis_group, ["resumen"])
    assert completo.exit_code == 0
    assert "- DSR < 0.95: 2" in completo.output


def test_resumen_usa_eventos_etapa_y_fallback_sin_eventos(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    with (tmp_path / "hipotesis.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNAS)
        writer.writerow(["2026-02-01T00:00:00+00:00", "H1", "0", "PASA", "csv"])
    with (tmp_path / "intentos.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fecha_utc", "hipotesis", "estado", "snapshot"])
        writer.writerow(["2026-02-01T00:30:00+00:00", "H1", "", "real"])
        writer.writerow(["2026-02-01T01:00:00+00:00", "H1", "invalida_bug", "s1"])
        writer.writerow(["2026-02-01T02:00:00+00:00", "H1", "", "sintetico"])

    solo_csv = resumen_tanda()
    assert "| 0 | PASA | 1 |" in solo_csv
    assert "Intentos: 1" in solo_csv
    from quantagent.fabrica.registro import importar_csv, registrar_evento
    importar_csv()
    assert resumen_tanda() == solo_csv
    assert resumen_tanda(desde=datetime.fromisoformat("2026-02-01T00:00:00+00:00")) == solo_csv

    registrar_evento("H2", "etapa", "3", "CORTA", "registro",
                     fecha_utc="2026-02-02T00:00:00+00:00")
    assert "| 3 | CORTA | 1 |" in resumen_tanda()

