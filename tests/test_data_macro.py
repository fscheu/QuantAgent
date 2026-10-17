"""`data macro create|verify` (vj37). El cliente FRED se reemplaza por uno falso; sin red ni clave."""

import pandas as pd
import pytest
from click.testing import CliRunner

from quantagent.cli.data import data_group
from quantagent.data.macro import (
    crear_snapshot_macro,
    serie_punto_en_el_tiempo,
    valor_conocido,
    verificar_snapshot_macro,
)
from quantagent.data.snapshot import SnapshotError


class ClienteFalso:
    """Devuelve el formato de `FredClient.observaciones`; UNRATE tiene una revisión (dos versiones de marzo)."""

    llamadas = []

    def observaciones(self, serie, desde, hasta, todas_las_versiones=True):
        self.llamadas.append((serie, todas_las_versiones))
        fin = pd.Timestamp("9999-12-31")
        filas = [("2020-03-01", 4.0, "2020-04-05", "2020-05-02"), ("2020-03-01", 4.2, "2020-05-03", fin)]
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


def _con_revision(tmp_path):
    crear_snapshot_macro("m", ["UNRATE"], ClienteFalso(), base_dir=tmp_path)


def test_punto_en_el_tiempo_ve_la_version_vigente_en_cada_fecha(tmp_path):
    _con_revision(tmp_path)
    mar = pd.Timestamp("2020-03-01")
    assert valor_conocido("m", "UNRATE", "2020-04-01", base_dir=tmp_path) is None
    assert valor_conocido("m", "UNRATE", "2020-04-10", base_dir=tmp_path) == (mar, 4.0)
    assert valor_conocido("m", "UNRATE", "2020-05-02", base_dir=tmp_path) == (mar, 4.0)  # último día de la versión: realtime_end es inclusivo
    assert valor_conocido("m", "UNRATE", "2020-05-03", base_dir=tmp_path) == (mar, 4.2)
    assert valor_conocido("m", "UNRATE", "2020-05-10", base_dir=tmp_path) == (mar, 4.2)
    s = serie_punto_en_el_tiempo("m", "UNRATE", ["2020-04-01", "2020-04-10", "2020-05-10"], base_dir=tmp_path)
    assert s.isna().tolist() == [True, False, False] and s.dropna().tolist() == [4.0, 4.2]


def test_truncar_versiones_posteriores_a_T_no_cambia_la_serie_hasta_T(tmp_path):
    _con_revision(tmp_path)
    T = pd.Timestamp("2020-04-20")
    dias = pd.date_range("2020-04-01", T)
    completa = serie_punto_en_el_tiempo("m", "UNRATE", dias, base_dir=tmp_path)
    ruta = tmp_path / "m" / "UNRATE.parquet"
    df = pd.read_parquet(ruta)
    assert len(df[df["realtime_start"] > T]) == 1  # la revisión del 3 de mayo sí se quita
    df[df["realtime_start"] <= T].to_parquet(ruta, index=False)
    truncada = serie_punto_en_el_tiempo("m", "UNRATE", dias, base_dir=tmp_path)
    pd.testing.assert_series_equal(completa, truncada)
