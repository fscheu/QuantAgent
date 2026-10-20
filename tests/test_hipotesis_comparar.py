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
    # sube ΔDSR y ΔSharpe, pero con esta serie al azar pierde años positivos (7 → 5): la regla m9cx la frena
    assert c["veredicto"] == "DESCARTA" and "ΔDSR" not in c["motivo"] and "años positivos 5 < 7" in c["motivo"]
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

    def correr(archivo, snapshot, desde, hasta, carpeta, abrir_prueba=False):
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
    assert r.exit_code in (0, 1), r.output  # 1 = la capa se descarta; este test mira el registro y las cuentas de N
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



def _por_anios(medias, sesiones=252, ruido=0.005, semilla=3):
    """Curva con un tramo de `sesiones` por año, cada uno con su retorno medio diario."""
    rng = np.random.default_rng(semilla)
    ret = np.concatenate([rng.normal(m, ruido, sesiones) for m in medias])
    return pd.Series(100_000.0 * np.cumprod(1 + ret), index=pd.bdate_range("2007-01-01", periods=len(ret)))


def test_capa_con_dsr_alto_pero_sharpe_menor_se_descarta(intentos, monkeypatch):
    nucleo, capa = _por_anios([0.001] * 4), _por_anios([0.0005] * 4, semilla=4)
    dsr = iter([0.50, 0.80])
    monkeypatch.setattr("quantagent.fabrica.comparar.dsr_contra_intentos", lambda *a, **k: next(dsr))
    c = comparar(nucleo, capa, intentos, margen=0.05)
    assert c["delta_dsr"] == pytest.approx(0.30) and c["delta_sharpe"] < 0
    assert c["veredicto"] == "DESCARTA" and "ΔSharpe" in c["motivo"] and "ΔDSR" not in c["motivo"]


def test_capa_que_pierde_un_anio_positivo_se_descarta(intentos, monkeypatch):
    nucleo = _por_anios([0.001, 0.001, 0.001, 0.001])
    capa = _por_anios([0.003, 0.003, 0.003, -0.0005], semilla=4)
    dsr = iter([0.50, 0.80])
    monkeypatch.setattr("quantagent.fabrica.comparar.dsr_contra_intentos", lambda *a, **k: next(dsr))
    c = comparar(nucleo, capa, intentos, margen=0.05)
    assert c["delta_sharpe"] > 0 and c["delta_dsr"] > 0.05
    assert (c["anios_positivos_nucleo"].positivos, c["anios_positivos_capa"].positivos) == (4, 3)
    assert c["veredicto"] == "DESCARTA" and "años positivos 3 < 4" in c["motivo"]


def test_capa_que_cumple_las_tres_condiciones_se_acepta(intentos, monkeypatch):
    nucleo, capa = _por_anios([0.001] * 4), _por_anios([0.002] * 4, semilla=4)
    dsr = iter([0.50, 0.80])
    monkeypatch.setattr("quantagent.fabrica.comparar.dsr_contra_intentos", lambda *a, **k: next(dsr))
    c = comparar(nucleo, capa, intentos, margen=0.05)
    assert c["veredicto"] == "ACEPTA" and c["anios_positivos_capa"].positivos == 4


def test_cli_abrir_prueba_abre_el_candado_de_la_capa_una_sola_vez(tmp_path, monkeypatch):
    (tmp_path / "datos").mkdir()
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path / "datos"))
    nucleo, _, con_serie = _series()
    pedidos = []

    def correr(archivo, snapshot, desde, hasta, carpeta, abrir_prueba=False):
        pedidos.append((archivo.split("/")[-1], desde, hasta, abrir_prueba))
        return archivo.split("/")[-1][:-5], _curva(nucleo) if "nucleo" in archivo else con_serie

    monkeypatch.setattr(hipotesis, "correr_cartera", correr)
    monkeypatch.setattr(hipotesis, "cargar_hipotesis", lambda a: type("H", (), {"id": "cartera-capa"})())
    (tmp_path / "datos" / "intentos.csv").write_text("sharpe,sharpe_base\n0.1,tasa_0\n0.4,tasa_0\n0.9,tasa_0\n")
    (tmp_path / "nucleo.yaml").write_text("")
    (tmp_path / "capa.yaml").write_text("")
    args = ["comparar", str(tmp_path / "nucleo.yaml"), str(tmp_path / "capa.yaml"), "--snapshot", "s",
            "--carpeta", str(tmp_path / "m"), "--margen", "0", "--desde", "2019-01-01", "--hasta", "2022-12-31",
            "--abrir-prueba"]
    r = CliRunner().invoke(hipotesis.hipotesis_group, args)
    assert r.exit_code in (0, 1) and "Años con Sharpe positivo" in r.output, r.output
    assert pedidos == [("nucleo.yaml", "2019-01-01", "2022-12-31", True), ("capa.yaml", "2019-01-01", "2022-12-31", True)]
    assert (tmp_path / "datos" / "hipotesis-candado.csv").read_text().startswith("cartera-capa,prueba,")
    segunda = CliRunner().invoke(hipotesis.hipotesis_group, args)
    assert segunda.exit_code != 0 and "se abre una sola vez" in segunda.output
    assert len(pedidos) == 2
