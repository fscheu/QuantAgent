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


def informe(raiz, hid, archivo=None):
    out = raiz / "i.html"
    args = ["hipotesis", "informe", hid, "--out", str(out)]
    if archivo:
        args.extend(["--archivo", str(archivo)])
    return CliRunner().invoke(backtest_group, args), out


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
    assert "al menos 2 trades, pnl &gt; 0, Sharpe con tasa 0 &gt;= 0,10 y caída máxima &lt;= 35%" in html
    assert "<th>Sharpe registrado</th>" in html and "puede pasar V30 con Sharpe registrado negativo" in html
    assert "<th>pnl" not in html.lower() and "123.4" not in html and "Prueba abierta: no" in html


def test_hipotesis_desconocida_falla(raiz):
    r, out = informe(raiz, "nada")
    assert r.exit_code != 0 and "desconocida" in r.output and not out.exists()


def test_informe_cartera_dos_piernas_contiene_universos_y_fila_cartera_sin_precios(raiz):
    import yaml
    yaml_path = raiz / "cartera.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "id": "h-cartera",
        "autor": "fede",
        "justificacion": "Cartera de 2 piernas.",
        "timeframe": "1d",
        "piernas": [
            {"clase": "momentum-12m", "parametros": {"lookback": [252]},
             "activos": ["SPY", "QQQ"], "universo": "Universo 1: 12 ETFs de acciones"},
            {"clase": "sma-cross", "parametros": {"fast": [1], "slow": [210]},
             "activos": ["TLT", "IEF"], "universo": "Universo 2: ETFs de bonos"}
        ]
    }))
    registrar_estado("h-cartera", "1", "PASA", "ok")
    with (raiz / "intentos.csv").open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_INTENTOS, restval="")
        w.writerow({"hipotesis": "h-cartera", "etapa": "1", "activo": "cartera", "sharpe": "0.45", "trades": "120", "pnl": "5000.0"})
        w.writerow({"hipotesis": "h-cartera", "etapa": "1", "activo": "SPY", "sharpe": "0.20", "trades": "60", "pnl": "2500.0"})
    r, out = informe(raiz, "h-cartera", archivo=yaml_path)
    assert r.exit_code == 0, r.output
    html = out.read_text()
    assert "Universo 1: 12 ETFs de acciones" in html
    assert "Universo 2: ETFs de bonos" in html
    assert "<td>cartera</td>" in html
    assert html.index("<td>cartera</td>") < html.index("<td>SPY</td>")
    assert "<th>pnl" not in html.lower() and "5000.0" not in html and "2500.0" not in html
    assert "Cartera" in html and "Pierna" in html and "Igual riesgo" in html


def test_informe_etapa3_muestra_cuatro_cifras_recargo_y_glosario_nuevo(raiz):
    from quantagent.cli.hipotesis import motivo_etapa3_cartera

    for e in ("1", "2"):
        registrar_estado("h-etapa3-completa", e, "PASA", "ok")
    fila_e3 = {"dsr": 0.80, "dsr_crudo": 0.55, "anios_positivos": "3 de 4", "benchmark": 0.12, "n_efectivo": 5}
    registrar_estado("h-etapa3-completa", "3", "PASA", motivo_etapa3_cartera("cartera pasa V30 y DSR", fila_e3))
    with (raiz / "intentos.csv").open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=CAMPOS_INTENTOS, restval="").writerow({"hipotesis": "h-etapa3-completa", "etapa": "3", "activo": "cartera", "sharpe": "0.45", "trades": "100"})
    r, out = informe(raiz, "h-etapa3-completa")
    assert r.exit_code == 0, r.output
    html = out.read_text()
    for h, v in (("N crudo", "6"), ("N efectivo", "5"), ("DSR efectivo", "0.80"), ("DSR crudo", "0.55"),
                 ("Recargo paper", "75% (6 sem)"), ("Años positivos", "3 de 4"), ("Benchmark", "0.12")):
        assert f"<th>{h}</th>" in html and f"<td>{v}</td>" in html
    assert "pasa con 0,95" not in html and "fija el recargo de paper trading" in html


def test_informe_etapa3_motivo_sin_claves_muestra_guiones(raiz):
    for e, v, m in (("1", "PASA", "ok"), ("2", "PASA", "ok"), ("3", "CORTA", "DSR bajo en QQQ")):
        registrar_estado("h-sin-claves", e, v, m)
    r, out = informe(raiz, "h-sin-claves")
    assert r.exit_code == 0 and "<h2>Criterio de etapa 3</h2>" in out.read_text() and out.read_text().count("<td>-</td>") >= 7


