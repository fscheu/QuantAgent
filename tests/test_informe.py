import csv

import pytest
from click.testing import CliRunner

from quantagent.cli.backtest import backtest_group
from quantagent.cli.sweep import CAMPOS_INTENTOS
from quantagent.fabrica.estado import registrar_estado


@pytest.fixture
def raiz(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    registrar_estado("h-corta1", "1", "CORTA", "2 de 2 bajo 0,10")
    for e, v, m in (("1", "PASA", "ok"), ("2", "PASA", "ok"), ("3", "CORTA", "DSR bajo en QQQ")):
        registrar_estado("h-corta3", e, v, m)
    filas = [("h-corta1", "1", "SPY", "0.02", "12"), ("h-corta1", "1", "IWM", "0.05", "20"),
             ("h-corta3", "2", "SPY", "0.9", "40"), ("h-corta3", "3", "QQQ", "0.31", "55"),
             ("h-corta3", "3", "EFA", "0.44", "61")]
    with (tmp_path / "intentos.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_INTENTOS, restval="")
        w.writeheader()
        for h, e, a, s, t in filas:
            w.writerow({"hipotesis": h, "etapa": e, "activo": a, "sharpe": s, "trades": t, "pnl": "123.4"})
    return tmp_path


def informe(raiz, hid):
    out = raiz / "i.html"
    return CliRunner().invoke(backtest_group, ["hipotesis", "informe", hid, "--out", str(out)]), out


def test_corte_en_etapa_1_nombra_etapa_motivo_y_activos(raiz):
    r, out = informe(raiz, "h-corta1")
    html = out.read_text()
    assert r.exit_code == 0, r.output
    assert "CORTA en la etapa 1" in html and "2 de 2 bajo 0,10" in html
    assert "SPY" in html and "IWM" in html and "QQQ" not in html


def test_corte_en_etapa_3_usa_solo_activos_de_esa_etapa(raiz):
    r, out = informe(raiz, "h-corta3")
    html = out.read_text()
    assert "CORTA en la etapa 3" in html and "DSR bajo en QQQ" in html
    assert "QQQ" in html and "EFA" in html and "<td>SPY</td>" not in html
    assert "pnl" not in html.lower() and "123.4" not in html and "Prueba abierta: no" in html


def test_hipotesis_desconocida_falla(raiz):
    r, out = informe(raiz, "nada")
    assert r.exit_code != 0 and "desconocida" in r.output and not out.exists()
