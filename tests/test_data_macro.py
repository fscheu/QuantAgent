"""`data macro create|verify` (vj37). El cliente FRED se reemplaza por uno falso; sin red ni clave."""

import pandas as pd
import pytest
from click.testing import CliRunner

from quantagent.cli.data import data_group
from quantagent.data.macro import crear_snapshot_macro, verificar_snapshot_macro
from quantagent.data.snapshot import SnapshotError


class ClienteFalso:
    """Devuelve el formato de `FredClient.observaciones`; UNRATE tiene una revisión (dos versiones de marzo)."""

    llamadas = []

    def observaciones(self, serie, desde, hasta, todas_las_versiones=True):
        self.llamadas.append((serie, todas_las_versiones))
        fin = pd.Timestamp("9999-12-31")
        filas = [("2020-03-01", 4.0, "2020-04-05", "2020-05-03"), ("2020-03-01", 4.2, "2020-05-03", fin)]
        if serie == "CPIAUCSL":
            filas = [("2020-03-01", 258.1, "2020-04-10", fin)]
        return pd.DataFrame(
            [(pd.Timestamp(f), v, pd.Timestamp(a), pd.Timestamp(b) if isinstance(b, str) else b) for f, v, a, b in filas],
            columns=["fecha", "valor", "realtime_start", "realtime_end"],
        )


def test_crea_parquet_y_manifiesto_y_verify_pasa(tmp_path):
    manifiesto = crear_snapshot_macro("m", ["UNRATE", "CPIAUCSL"], ClienteFalso(), base_dir=tmp_path)
    assert manifiesto["source"] == "alfred"
    assert manifiesto["series"]["UNRATE"]["rows"] == 2
    assert manifiesto["series"]["UNRATE"]["start"] == manifiesto["series"]["UNRATE"]["end"] == "2020-03-01"
    df = pd.read_parquet(tmp_path / "m" / "UNRATE.parquet")
    assert df["valor"].tolist() == [4.0, 4.2]
    assert df["realtime_end"].iloc[1] == pd.Timestamp("9999-12-31")
    assert verificar_snapshot_macro("m", base_dir=tmp_path) == manifiesto


def test_verify_nombra_la_serie_alterada(tmp_path):
    crear_snapshot_macro("m", ["UNRATE", "CPIAUCSL"], ClienteFalso(), base_dir=tmp_path)
    ruta = tmp_path / "m" / "UNRATE.parquet"
    ruta.write_bytes(ruta.read_bytes() + b"x")
    with pytest.raises(SnapshotError) as exc:
        verificar_snapshot_macro("m", base_dir=tmp_path)
    assert "UNRATE" in str(exc.value) and "CPIAUCSL" not in str(exc.value)


def test_segundo_create_con_el_mismo_nombre_falla_sin_tocar_nada(tmp_path):
    crear_snapshot_macro("m", ["UNRATE"], ClienteFalso(), base_dir=tmp_path)
    antes = {p.name: p.read_bytes() for p in (tmp_path / "m").iterdir()}
    with pytest.raises(SnapshotError, match="ya existe"):
        crear_snapshot_macro("m", ["UNRATE", "CPIAUCSL"], ClienteFalso(), base_dir=tmp_path)
    assert {p.name: p.read_bytes() for p in (tmp_path / "m").iterdir()} == antes


def test_serie_sin_filas_no_deja_carpeta(tmp_path):
    class Vacio:
        def observaciones(self, *a, **k):
            return pd.DataFrame(columns=["fecha", "valor", "realtime_start", "realtime_end"])

    with pytest.raises(SnapshotError, match="sin filas: X"):
        crear_snapshot_macro("m", ["X"], Vacio(), base_dir=tmp_path)
    assert not (tmp_path / "m").exists()


def test_cli_create_y_verify(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("FRED_API_KEY", "clave-falsa")
    monkeypatch.setattr("quantagent.data.sources.fred.FredClient", lambda: ClienteFalso())
    runner = CliRunner()
    creado = runner.invoke(data_group, ["macro", "create", "--name", "m", "--series", "unrate,CPIAUCSL"])
    assert creado.exit_code == 0, creado.output
    assert "2 series, 3 filas" in creado.output
    ok = runner.invoke(data_group, ["macro", "verify", "m"])
    assert ok.exit_code == 0 and ok.output.strip() == "OK 2 series, 3 filas"
    (tmp_path / "m" / "UNRATE.parquet").write_bytes(b"roto")
    mal = runner.invoke(data_group, ["macro", "verify", "m"])
    assert mal.exit_code == 1 and "UNRATE" in mal.output
    otra_vez = runner.invoke(data_group, ["macro", "create", "--name", "m", "--series", "UNRATE"])
    assert otra_vez.exit_code == 1 and "ya existe" in otra_vez.output
