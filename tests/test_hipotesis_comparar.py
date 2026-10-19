"""`hipotesis comparar` (QuantAgent-d4cq.73): ΔDSR de la capa con el N de intentos.csv."""
import csv

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from quantagent.cli import hipotesis
from quantagent.cli.sweep import registrar_intento
from quantagent.fabrica.comparar import comparar

DIAS = 2500


def _curva(retornos):
    return pd.Series(100_000.0 * np.cumprod(1 + retornos), index=pd.bdate_range("2007-01-01", periods=len(retornos)))


def _series():
    rng = np.random.default_rng(7)
    nucleo = rng.normal(0.0004, 0.01, DIAS)
    ruido = rng.normal(0.0, 0.01, DIAS)
    independiente = rng.normal(0.0004, 0.01, DIAS)
    return nucleo, _curva(0.5 * nucleo + 0.5 * ruido), _curva(0.5 * nucleo + 0.5 * independiente)


def _intentos(ruta, sharpes):
    filas = [{"sharpe": s, "sharpe_base": "tasa_0"} for s in sharpes]
    with ruta.open("w", newline="") as f:
        w = csv.DictWriter(f, ["sharpe", "sharpe_base"])
        w.writeheader()
        w.writerows(filas)


@pytest.fixture
def intentos(tmp_path):
    ruta = tmp_path / "intentos.csv"
    _intentos(ruta, [0.1, 0.4, 0.9, 0.3, 0.6, 0.55, 0.62])
    return ruta


def test_capa_que_agrega_ruido_se_descarta(intentos):
    nucleo, con_ruido, _ = _series()
    c = comparar(_curva(nucleo), con_ruido, intentos, margen=0.0)
    assert c["delta_dsr"] <= 0 and c["delta_sharpe"] < 0
    assert c["veredicto"] == "DESCARTA"


def test_capa_con_serie_independiente_positiva_sube_el_dsr(intentos):
    nucleo, _, con_serie = _series()
    c = comparar(_curva(nucleo), con_serie, intentos, margen=0.0)
    assert c["delta_dsr"] > 0 and c["delta_sharpe"] > 0
    assert c["veredicto"] == "ACEPTA"
    assert comparar(_curva(nucleo), con_serie, intentos, margen=c["delta_dsr"] + 0.01)["veredicto"] == "DESCARTA"
    assert (c["n_con_capa"], c["n_sin_capa"]) == (7, 6)


def test_cli_registra_una_fila_por_cartera_y_muestra_las_dos_cuentas_de_n(tmp_path, monkeypatch):
    (tmp_path / "datos").mkdir()
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path / "datos"))
    nucleo, _, con_serie = _series()
    curvas = {"nucleo.yaml": ("n", _curva(nucleo)), "capa.yaml": ("c", con_serie)}
    for s in (0.2, 0.5, 0.35, 0.6):
        registrar_intento("x", "SPY", "s", "etf", 1.0, {}, {"desde": "", "hasta": "", "trades": 1, "pnl": 1, "sharpe": s},
                          hipotesis="previo", etapa="1", sharpe_base="tasa_0")

    def correr(archivo, snapshot, desde, hasta, carpeta):
        id_, eq = curvas[archivo.split("/")[-1]]
        registrar_intento(id_, "cartera", snapshot, "etf", 1.0, {}, {"desde": desde, "hasta": hasta, "trades": 5,
                          "pnl": 1, "sharpe": 0.5}, hipotesis=id_, etapa="comparar", sharpe_base="tasa_0")
        return id_, eq

    monkeypatch.setattr(hipotesis, "correr_cartera", correr)
    (tmp_path / "nucleo.yaml").write_text("")
    (tmp_path / "capa.yaml").write_text("")
    out = tmp_path / "informe.html"
    r = CliRunner().invoke(hipotesis.hipotesis_group, [
        "comparar", str(tmp_path / "nucleo.yaml"), str(tmp_path / "capa.yaml"), "--snapshot", "s",
        "--carpeta", str(tmp_path / "m"), "--margen", "0", "--out", str(out)])
    assert r.exit_code == 0, r.output
    with (tmp_path / "datos" / "intentos.csv").open() as f:
        filas = [x for x in csv.DictReader(f) if x["etapa"] == "comparar"]
    assert [x["hipotesis"] for x in filas] == ["n", "c"]
    assert "N 6 con la capa, 5 sin ella" in r.output
    assert "6 contando la capa, 5 sin ella" in out.read_text()


def test_cli_rechaza_ventana_de_prueba(tmp_path):
    (tmp_path / "a.yaml").write_text("")
    r = CliRunner().invoke(hipotesis.hipotesis_group, [
        "comparar", str(tmp_path / "a.yaml"), str(tmp_path / "a.yaml"), "--snapshot", "s",
        "--carpeta", str(tmp_path), "--margen", "0", "--hasta", "2022-12-31"])
    assert r.exit_code != 0 and "prueba" in r.output


def test_correr_cartera_desempaqueta_parametros_escalares(tmp_path, monkeypatch):
    """Verifica que correr_cartera extrae parámetros escalares de p.combinaciones() en vez de pasar listas."""
    yaml_path = tmp_path / "cartera.yaml"
    yaml_path.write_text("""id: test-cartera
autor: fede
justificacion: prueba unitaria
timeframe: 1d
piernas:
  - clase: sma-cross
    parametros:
      fast: [1]
      slow: [210]
    activos: [SPY]
    universo: prueba
""")
    capturados = {}

    def fake_correr_motor(h, snapshot, params, desde, hasta, carpeta, etapa):
        capturados.update(params)
        return [{"activo": "SPY", "trades": "5"}]

    def fake_curva(h, filas, carpeta, snapshot, etapa, desde, hasta):
        idx = pd.bdate_range("2007-01-01", periods=10)
        return pd.Series(100_000.0, index=idx), True, []

    monkeypatch.setattr("quantagent.fabrica.validacion.correr_motor", fake_correr_motor)
    monkeypatch.setattr("quantagent.fabrica.validacion._curva_y_v30_cartera", fake_curva)

    id_, eq = hipotesis.correr_cartera(str(yaml_path), "snap", "2007-01-01", "2018-12-31", str(tmp_path))
    assert id_ == "test-cartera"
    assert capturados == {("sma-cross", "SPY"): {"fast": 1, "slow": 210}}

