import ast
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from quantagent.fabrica import estado
from quantagent.fabrica import registro_db as registro
from quantagent.fabrica import registro_vista as vista


@pytest.fixture
def raiz(db_temporal):
    return db_temporal


def _keys(interface):
    source = Path("docs/03_design/web-design-system/api-types.ts").read_text()
    match = re.search(rf"interface {interface}(?: extends (\w+))? \{{(.*?)\n\}}", source, re.S)
    keys = set(re.findall(r"^\s*(\w+):", match.group(2), re.M))
    return keys | (_keys(match.group(1)) if match.group(1) else set())


def test_embudo_claves_y_agregados(raiz):
    registro.abrir_ciclo("2026-01", ["A", "B", "C"])
    registro.registrar_evento("A", "etapa", "0", "PASA", fecha_utc="2026-01-01T00:00:00+00:00")
    registro.registrar_evento("A", "etapa", "1", "PASA", fecha_utc="2026-01-02T00:00:00+00:00")
    estado.registrar_estado("A", "2", "CORTA", medida=estado.medida("sharpe", 0.2, 0.5, ">=", "ratio"), causa="sharpe_below")
    estado.registrar_estado("B", "1", "ERROR", medida=estado.medida("trades", 1, 5, ">=", "count"), causa="few_trades")
    registro.registrar_evento("C", "etapa", "0", "ALARMA", fecha_utc="2026-01-05T00:00:00+00:00")

    result = vista.embudo()
    assert set(result) == _keys("Funnel")
    assert all(set(s) == _keys("StageCounts") for s in result["stages"])
    assert all(set(r) == _keys("DropReason") for r in result["drop_reasons"])
    assert result["stages"] == [
        {"stage": "e0", "reached": 2, "here": 0, "passed": 1, "cut": 0, "error": 0, "alarm": 1},
        {"stage": "e1", "reached": 2, "here": 0, "passed": 1, "cut": 0, "error": 1, "alarm": 0},
        {"stage": "e2", "reached": 1, "here": 0, "passed": 0, "cut": 1, "error": 0, "alarm": 0},
        {"stage": "e3", "reached": 0, "here": 0, "passed": 0, "cut": 0, "error": 0, "alarm": 0},
    ]
    actual = registro.estados_actuales()
    assert all(s["here"] + s["cut"] + s["error"] + s["alarm"] == sum(e["etapa"] == str(i) for e in actual.values())
               for i, s in enumerate(result["stages"]))
    assert {reason["verdict"] for reason in result["drop_reasons"]} == {"CORTA", "ERROR", "ALARMA"}
    assert next(r for r in result["drop_reasons"] if r["cause_code"] == "sharpe_below")["example"]["value"] == 0.2


def test_ciclos_abierto_cerrado_claves_y_limite(raiz):
    registro.abrir_ciclo("2026-01", ["A"])
    registro.registrar_evento("A", "etapa", "1", "CORTA", "motivo", ciclo="2026-01")
    registro.abrir_ciclo("2026-02", ["B"])
    registro.cerrar_ciclo("2026-02")
    result = vista.ciclos(5)
    assert all(set(c) == _keys("CycleSummary") for c in result)
    assert result[0]["cycle"] == "2026-02" and result[0]["status"] == "closed"
    assert result[0]["headline"] is None and result[0]["receipt_id"] == "2026-02"
    assert result[1]["cycle"] == "2026-01" and result[1]["closed_at"] is None
    assert result[1]["cut"] == 1 and result[1]["receipt_id"] is None
    assert result[1]["reached_by_stage"] == {"e0": None, "e1": 1, "e2": None, "e3": None}
    assert result[1]["milestones"] == [
        {"key": "proposal", "day": 1, "date": "2026-01-01", "done": True},
        {"key": "funnel_start", "day": 8, "date": "2026-01-08", "done": True},
        {"key": "paper_review", "day": 25, "date": "2026-01-25", "done": False}]
    assert len(vista.ciclos(1)) == 1


def test_ciclos_held_cuenta_error_y_alarma(raiz):
    registro.abrir_ciclo("2026-03", ["A", "B"])
    registro.registrar_evento("A", "etapa", "0", "ERROR", ciclo="2026-03")
    registro.registrar_evento("B", "etapa", "1", "ALARMA", ciclo="2026-03")
    ciclo = vista.ciclos(1)[0]
    assert ciclo["held"] == 2 and ciclo["cut"] == 0 and ciclo["active"] == 0


def test_regla_revision_racha_cerrada_y_ciclos_abiertos(raiz):
    for mes in range(1, 7):
        ciclo = f"2026-{mes:02d}"
        registro.abrir_ciclo(ciclo, [f"H{mes}"])
        registro.cerrar_ciclo(ciclo)
    registro.abrir_ciclo("2026-07", ["OPEN"])
    regla = vista.regla_revision()
    assert set(regla) == _keys("ReviewRule")
    assert all(set(ciclo) == {"cycle", "promoted"} for ciclo in regla["last_cycles"])
    assert regla == {"limit": 6, "streak": 6, "reached": True,
        "last_cycles": [{"cycle": f"2026-{mes:02d}", "promoted": 0} for mes in range(1, 7)]}


def test_regla_revision_racha_se_corta_con_promocion(raiz):
    for mes in range(1, 8):
        ciclo = f"2026-{mes:02d}"
        registro.abrir_ciclo(ciclo, [f"H{mes}"])
        if mes == 5:
            for etapa in ("0", "1", "2", "3"):
                registro.registrar_evento(f"H{mes}", "etapa", etapa, "PASA", ciclo=ciclo)
        registro.cerrar_ciclo(ciclo)
    regla = vista.regla_revision()
    assert regla["streak"] == 2 and not regla["reached"]
    assert regla["last_cycles"][-3:] == [{"cycle": "2026-05", "promoted": 1},
        {"cycle": "2026-06", "promoted": 0}, {"cycle": "2026-07", "promoted": 0}]


def test_hipotesis_claves_filtros_y_orden(raiz):
    registro.abrir_ciclo("2026-01", ["A", "B", "C", "D", "E"])
    registro.registrar_evento("A", "etapa", "0", "PASA", fecha_utc="2026-01-01T00:00:00+00:00")
    registro.registrar_evento("B", "etapa", "2", "CORTA", "falla", fecha_utc="2026-01-02T00:00:00+00:00")
    registro.registrar_evento("C", "etapa", "3", "PASA", ciclo="2026-01")
    registro.registrar_evento("D", "etapa", "1", "ERROR", ciclo="2026-01")
    registro.registrar_evento("E", "etapa", "1", "ALARMA", ciclo="2026-01")
    result = vista.hipotesis()
    assert all(set(h) == _keys("Hypothesis") for h in result)
    assert [h["id"] for h in result] == ["C", "B", "E", "D", "A"]
    assert {h["id"]: h["status"] for h in result} == {"A": "active", "B": "discarded", "C": "promoted", "D": "held", "E": "held"}
    assert vista.hipotesis(estado="discarded")[0]["id"] == "B"
    assert vista.hipotesis(etapa="e0")[0]["id"] == "A"
    assert len(vista.hipotesis(ciclo="2026-01")) == 5


def test_hipotesis_intentos_invalidos_y_detalle(raiz):
    registro.abrir_ciclo("2026-01", ["A"])
    registro.registrar_evento("A", "etapa", "0", "CORTA", "umbral", ciclo="2026-01",
        datos={"measure": {"metric": "sharpe", "value": 0.2, "threshold": 0.5, "unit": "ratio", "comparator": ">="}})
    row = {"desde": "2026-01-01", "hasta": "2026-01-02", "trades": 3, "pnl": 1, "sharpe": 0.4}
    registro.registrar_intento("estrategia", "SPY", "snap", None, None, {}, row, "A", "0")
    registro.registrar_intento("estrategia", "SPY", "snap", None, None, {}, row, "A", "0")
    from quantagent.fabrica.registro_db import invalidar_intentos
    assert invalidar_intentos({"hipotesis": "A"}) == 2
    h = vista.hipotesis()[0]
    assert (h["attempts_valid"], h["attempts_invalid"]) == (0, 2)
    detalle = vista.hipotesis_detalle("A")
    assert set(detalle) == _keys("HypothesisDetail")
    assert detalle["timeline"][0]["measure"]["value"] == 0.2
    assert all(a["invalid"] for a in detalle["attempts"])
    assert detalle["later_phases"] == [] and detalle["backtests"] == [] and detalle["paper_account"] is None
    assert vista.hipotesis_detalle("missing") is None


def test_nombre_idea_yaml_y_vistas(raiz, tmp_path, monkeypatch):
    from quantagent.fabrica.hipotesis import nombre_e_idea

    contenido = ("id: alpha\nautor: fede\nnombre: Reversión\njustificacion: |\n"
                 "  Primera idea.\n\n  Segunda línea.\nclase: rsi\nparametros: {}\n"
                 "activos: [SPY]\nuniverso: SPY\ntimeframe: 1d\n")
    (tmp_path / "alpha.yaml").write_text(contenido)
    assert nombre_e_idea("alpha", tmp_path) == ("Reversión", "Primera idea.")
    assert nombre_e_idea("ausente", tmp_path) == ("ausente", None)
    (tmp_path / "roto.yaml").write_text("no: [yaml")
    assert nombre_e_idea("roto", tmp_path) == ("roto", None)

    monkeypatch.setenv("QUANTAGENT_HIPOTESIS_DIR", str(tmp_path))
    registro.abrir_ciclo("2026-08", ["alpha", "ausente", "roto"])
    for h in ("alpha", "ausente", "roto"):
        registro.registrar_evento(h, "etapa", "0", "PASA", ciclo="2026-08")
    filas = {h["id"]: h for h in vista.hipotesis()}
    assert (filas["alpha"]["name"], filas["alpha"]["idea"]) == ("Reversión", "Primera idea.")
    assert (filas["ausente"]["name"], filas["ausente"]["idea"]) == ("ausente", None)
    assert (filas["roto"]["name"], filas["roto"]["idea"]) == ("roto", None)
    detalle = vista.hipotesis_detalle("alpha")
    assert (detalle["name"], detalle["idea"]) == ("Reversión", "Primera idea.")


def test_periodo_usa_utc_y_ahora_inyectable(raiz):
    registro.registrar_evento("A", "etapa", "0", "PASA", fecha_utc="2026-01-01T00:00:00+00:00")
    registro.registrar_evento("B", "etapa", "0", "PASA", fecha_utc="2026-01-09T00:00:00+00:00")
    result = vista.embudo(periodo=2, ahora=datetime(2026, 1, 10, tzinfo=timezone.utc))
    assert result["total"] == 1


def test_sin_imports_de_framework_web():
    tree = ast.parse(Path("quantagent/fabrica/registro_vista.py").read_text())
    imports = {alias.name.split(".")[0] for node in ast.walk(tree)
               if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    imports |= {node.module.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module}
    assert not imports.intersection({"fastapi", "flask", "streamlit"})


def test_informe_ciclo_cerrado_promocion_y_cli(raiz):
    from click.testing import CliRunner
    from quantagent.cli.__main__ import cli

    registro.abrir_ciclo("2026-01", ["A"])
    for etapa in ("0", "1", "2", "3"):
        registro.registrar_evento("A", "etapa", etapa, "PASA")
    registro.cerrar_ciclo("2026-01")
    texto = vista.informe_ciclo("2026-01")
    assert "1 hipótesis entraron" in texto and "1 promovieron" in texto
    assert "Cierre:" not in texto
    resultado = CliRunner().invoke(cli, ["registro", "ciclo", "informe", "2026-01"])
    assert resultado.exit_code == 0
    assert texto in resultado.output


def test_informe_ciclo_sin_promocion_y_causa(raiz):
    registro.abrir_ciclo("2026-02", ["A"])
    registro.registrar_evento("A", "etapa", "1", "CORTA", "RIESGO: exposición alta")
    registro.cerrar_ciclo("2026-02")
    texto = vista.informe_ciclo("2026-02")
    assert "Cierre: ninguna hipótesis promovida" in texto
    assert "A: e1 CORTA" in texto
    assert "Motivo: RIESGO: exposición alta" in " ".join(texto.split())
    assert "Indica que la evaluación detectó: exposición alta" in " ".join(texto.split())


def test_informe_hashes_parciales_validacion_y_ancho(raiz):
    import hashlib
    import json

    for ciclo, nombre, contenido in (("2026-03", "snap-a", "uno"),
                                     ("2026-04", "snap-b", "dos")):
        carpeta = raiz / nombre
        carpeta.mkdir()
        (carpeta / "manifest.json").write_text(json.dumps({"contenido": contenido}))
        registro.abrir_ciclo(ciclo, ["A"])
        registro.registrar_intento("s", "SPY", nombre, None, None, {},
            {"desde": "2026-01-01", "hasta": "2026-01-02", "trades": 1, "pnl": 1, "sharpe": 1}, "A")
        registro.cerrar_ciclo(ciclo)
    primero = vista.informe_ciclo("2026-03")
    segundo = vista.informe_ciclo("2026-04")
    assert primero != segundo
    assert hashlib.sha256((raiz / "snap-a" / "manifest.json").read_bytes()).hexdigest()[:12] in primero

    registro.abrir_ciclo("2026-05", ["A"])
    parcial = vista.informe_ciclo("2026-05")
    assert "AVISO: ciclo abierto, informe parcial." in parcial
    assert all(len(linea) <= 60 for linea in (primero + "\n" + segundo + "\n" + parcial).splitlines())
    with pytest.raises(ValueError, match="no existe"):
        vista.informe_ciclo("2026-06")
    with pytest.raises(ValueError, match="ciclo inválido"):
        vista.informe_ciclo("2026-13")


def test_recibo_cuenta_intentos_desde_sql_y_persiste_sin_evento_intento(raiz):
    registro.abrir_ciclo("2026-06", ["A"])
    registro.registrar_intento("s", "SPY", "snap", None, None, {},
        {"desde": "2026-01-01", "hasta": "2026-01-02", "trades": 1, "pnl": 1, "sharpe": 1}, "A")
    recibo = registro.cerrar_ciclo("2026-06")
    assert recibo["intentos"] == {"A": 1}
    assert not any(e["evento"] == "intento" for e in registro.eventos())
    assert registro.recibo_ciclo("2026-06") == recibo
