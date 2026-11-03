import ast
import re
from collections import Counter
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
    linea = re.search(rf"export interface {interface} \{{(.*)$", source, re.M)
    if linea.group(1).rstrip().endswith("}"):  # interface en una sola línea
        return set(re.findall(r"(?:^|[;\s])(\w+)\??:", linea.group(1)))
    body = re.search(rf"export interface {interface} \{{(.*?)\n\}}", source, re.S).group(1)
    return set(re.findall(r"^\s*(\w+)\??:", body, re.M))


def test_embudo_claves_y_agregados(raiz):
    registro.abrir_ciclo("2026-01", ["A", "B"])
    registro.registrar_evento("A", "etapa", "0", "PASA", fecha_utc="2026-01-01T00:00:00+00:00")
    registro.registrar_evento("A", "etapa", "1", "PASA", fecha_utc="2026-01-02T00:00:00+00:00")
    registro.registrar_evento("A", "etapa", "2", "CORTA", "X: uno; Y: dos", fecha_utc="2026-01-03T00:00:00+00:00")
    registro.registrar_evento("B", "etapa", "1", "ERROR", "X: uno; Y: dos", fecha_utc="2026-01-04T00:00:00+00:00")

    result = vista.embudo()
    assert set(result) == _keys("Funnel")
    assert all(set(s) == _keys("FunnelStage") for s in result["stages"])
    assert all(set(r) == _keys("DropReason") for r in result["drop_reasons"])
    assert result["stages"] == [
        {"stage": "e0", "reached": 1, "here": 0, "dropped": 0},
        {"stage": "e1", "reached": 2, "here": 0, "dropped": 1},
        {"stage": "e2", "reached": 1, "here": 0, "dropped": 1},
        {"stage": "e3", "reached": 0, "here": 0, "dropped": 0},
    ]
    actual = registro.estados_actuales()
    assert all(s["here"] + s["dropped"] == sum(e["etapa"] == str(i) for e in actual.values())
               for i, s in enumerate(result["stages"]))
    summary = estado.resumen_tanda()
    counts = Counter({reason: int(count) for reason, count in
                      re.findall(r"- (.*): (\d+)", summary.split("## Causas más frecuentes", 1)[1])})
    actual = Counter()
    for reason in result["drop_reasons"]:
        actual[reason["reason"]] += reason["count"]
    assert actual == counts


def test_ciclos_abierto_cerrado_claves_y_limite(raiz):
    registro.abrir_ciclo("2026-01", ["A"])
    registro.registrar_evento("A", "etapa", "1", "CORTA", "motivo", ciclo="2026-01")
    registro.abrir_ciclo("2026-02", ["B"])
    registro.cerrar_ciclo("2026-02")
    result = vista.ciclos(5)
    assert all(set(c) == _keys("CycleSummary") for c in result)
    assert result[0]["cycle"] == "2026-02" and result[0]["ended_at"] is not None
    assert result[0]["headline"] is None and result[0]["dropped"] == 0
    assert result[1]["cycle"] == "2026-01" and result[1]["ended_at"] is None
    assert result[1]["dropped"] == 1
    assert len(vista.ciclos(1)) == 1


def test_hipotesis_claves_filtros_y_orden(raiz):
    registro.abrir_ciclo("2026-01", ["A", "B"])
    registro.registrar_evento("A", "etapa", "0", "PASA", fecha_utc="2026-01-01T00:00:00+00:00")
    registro.registrar_evento("B", "etapa", "2", "CORTA", "falla", fecha_utc="2026-01-02T00:00:00+00:00")
    result = vista.hipotesis()
    assert all(set(h) == _keys("Hypothesis") for h in result)
    assert [h["id"] for h in result] == ["B", "A"]
    assert vista.hipotesis(estado="discarded")[0]["id"] == "B"
    assert vista.hipotesis(etapa="e0")[0]["id"] == "A"
    assert vista.hipotesis(ciclo="2026-01") == result


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
