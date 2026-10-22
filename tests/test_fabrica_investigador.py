"""QuantAgent-d4cq.42: pedido al LLM sin cifras de rendimiento y escritura en catalogo/propuestas/."""
import sys

import pytest
import yaml
from click.testing import CliRunner

from quantagent.cli import hipotesis
from quantagent.cli.backtest import backtest_group
from quantagent.fabrica import propuestas
from quantagent.fabrica.hipotesis import cargar_hipotesis
from quantagent.fabrica.investigador import armar_pedido, escribir_propuestas
from quantagent.fabrica.propuestas import validar_propuesta


@pytest.fixture(autouse=True)
def crossref_simulado(monkeypatch):
    monkeypatch.setattr(propuestas, "_titulo_crossref", lambda doi: "Time Series Momentum" if "2011.11.003" in doi else "A Century of Evidence on Trend Following Investing")


def test_escribir_propuestas_desde_respuesta_simulada(tmp_path):
    respuesta = yaml.safe_dump({"propuestas": [
        {"id": "prop-a", "idea": "momentum", "clase": "momentum-12m", "ventana": {"lookback": [252]}, "tema": "tendencia", "fuente_de_retorno": "prima-de-riesgo", "hipotesis_economica": "Continuación a 12m.",
         "fuente": ["10.1016/j.jfineco.2011.11.003"], "cita_texto": ["Time Series Momentum"], "universo": {"activos": ["SPY", "QQQ"], "criterio": "ETFs de acciones"}, "parametros_libres": 0},
        {"id": "prop-b", "idea": "tendencia", "clase": "sma-cross", "tema": "tendencia", "fuente_de_retorno": "prima-de-riesgo", "hipotesis_economica": "Tendencia secular.",
         "fuente": ["10.3905/jwm.2007.674809"], "cita_texto": ["A Century of Evidence on Trend Following Investing"], "universo": {"activos": ["SPY", "TLT"], "criterio": "Multiactivo líquido"}, "parametros_libres": 0},
    ]})
    rutas, motivo = escribir_propuestas(respuesta, tmp_path / "propuestas")
    assert motivo is None and [r.name for r in rutas] == ["prop-a.yaml", "prop-b.yaml"]
    assert all(validar_propuesta(yaml.safe_load(r.read_text(encoding="utf-8"))) is None for r in rutas)
    with pytest.raises(ValueError, match="hipotesis"):
        escribir_propuestas(respuesta, tmp_path / "hipotesis")


def test_escribir_propuestas_rechaza_propuesta_incompleta(tmp_path):
    propuesta = {"id": "prop-incompleta", "idea": "momentum", "clase": "momentum-12m",
                 "tema": "tendencia", "hipotesis_economica": "Continuación a 12m.",
                 "fuente": ["10.1016/j.jfineco.2011.11.003"], "cita_texto": ["Time Series Momentum"],
                 "universo": {"activos": ["SPY"], "criterio": "ETFs líquidos"}, "parametros_libres": 0}
    rutas, motivo = escribir_propuestas(yaml.safe_dump({"propuestas": [propuesta]}), tmp_path / "propuestas")
    assert rutas == [] and "fuente_de_retorno" in motivo
    assert not list((tmp_path / "propuestas").glob("*.yaml"))


def test_escribir_propuestas_no_persiste_doi_sin_verificar(tmp_path, monkeypatch):
    monkeypatch.setattr(propuestas, "_titulo_crossref", lambda _: (_ for _ in ()).throw(OSError("offline")))
    propuesta = {"id": "prop-offline", "idea": "momentum", "clase": "momentum-12m", "tema": "tendencia", "fuente_de_retorno": "prima-de-riesgo",
                 "hipotesis_economica": "Continuación.", "fuente": ["10.1016/j.jfineco.2011.11.003"], "cita_texto": ["Time Series Momentum"],
                 "universo": {"activos": ["SPY"], "criterio": "ETFs líquidos"}, "parametros_libres": 0}
    rutas, motivo = escribir_propuestas(yaml.safe_dump({"propuestas": [propuesta]}), tmp_path / "propuestas")
    assert rutas == [] and "sin verificar" in motivo and not list(tmp_path.rglob("*.yaml"))


def test_pedido_sin_cifras_de_sharpe_dsr_ni_pnl(tmp_path):
    pm, intentos = tmp_path / "postmortem", tmp_path / "intentos.csv"
    pm.mkdir()
    (pm / "c1.yaml").write_text(yaml.safe_dump({"id": "c1", "causas": ["pocos_trades"], "sharpe": 0.4321, "dsr": 0.9876, "pnl": -1234.56, "trades": 789}), encoding="utf-8")
    intentos.write_text("hipotesis,sharpe,dsr,pnl,trades\nc1,0.4321,0.9876,-1234.56,789\nc2,0.5555,0.8888,999.0,654\n", encoding="utf-8")
    pedido = armar_pedido("catalogo/primas.yaml", pm, intentos)
    assert all(e in pedido for e in ("Cómo se evalúa", "comprar y mantener", "AGG: '2007-01-03'", "DGS3MO", "RSIMeanReversionStrategy", "momentum-transversal", "deriva-pre-fomc", "intentos_registrados: 2", "pocos_trades", "tendencia-series-de-tiempo"))
    assert "diversificador" not in pedido
    assert all(p not in pedido.lower() for p in ("sharpe", "dsr", "pnl", "trades:", "0.4321", "0.9876", "1234.56", "0.5555", "0.8888", "999.0", "789", "654"))


@pytest.mark.parametrize("invalida", [":::no_yaml: [", "texto suelto", "", "propuestas: [{idea: x}]", "propuestas: [{id: p1}, {id: p1}]", "propuestas: [{id: p1, idea: x}]"])
def test_respuesta_yaml_invalido_no_escribe_nada(tmp_path, invalida):
    rutas, motivo = escribir_propuestas(invalida, tmp_path / "propuestas")
    assert rutas == [] and motivo and not list(tmp_path.rglob("*.yaml"))


def test_proponer_cli_lista_aceptadas_y_rechazadas_sin_correr_hipotesis(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    for fn in ("admitir_hipotesis", "filtrar_hipotesis", "validar_etapa2", "validar_etapa3"):
        monkeypatch.setattr(hipotesis, fn, lambda *a, **k: pytest.fail(f"No debe ejecutar {fn}"))
    h_ex = cargar_hipotesis("hipotesis/momentum-12m.yaml")
    resp_file, stdin_file, prop_dir = tmp_path / "resp.yaml", tmp_path / "stdin.txt", tmp_path / "propuestas"
    resp_file.write_text(yaml.safe_dump({"propuestas": [
        {"id": "prop-ok", "idea": "carry-divisas", "clase": "momentum-12m", "ventana": {"lookback": [63]}, "tema": "tendencia", "fuente_de_retorno": "prima-de-riesgo", "hipotesis_economica": "Carry corto en bonos.",
         "fuente": ["10.1016/j.jfineco.2011.11.003"], "cita_texto": ["Time Series Momentum"], "universo": {"activos": ["IEF", "TLT"], "criterio": "Bonos del Tesoro"}, "parametros_libres": 0},
        {"id": "prop-sin-doi", "idea": "otra", "clase": "rsi", "tema": "tendencia", "fuente_de_retorno": "prima-de-riesgo", "hipotesis_economica": "Reversión.", "fuente": ["sin-doi"], "cita_texto": ["Time Series Momentum"], "universo": {"activos": ["SPY"], "criterio": "Acciones"}, "parametros_libres": 0},
        {"id": "prop-dup", "idea": "mom-dup", "clase": "momentum-12m", "ventana": h_ex.parametros, "tema": "tendencia", "fuente_de_retorno": "prima-de-riesgo", "hipotesis_economica": "Igual.", "fuente": ["10.1016/j.jfineco.2011.11.003"], "cita_texto": ["Time Series Momentum"], "universo": {"activos": list(h_ex.activos), "criterio": "12 ETFs"}, "parametros_libres": 0},
    ]}), encoding="utf-8")
    (script := tmp_path / "fake_llm.py").write_text(f"import sys, pathlib; pathlib.Path({str(stdin_file)!r}).write_text(sys.stdin.read()); print(pathlib.Path({str(resp_file)!r}).read_text())\n", encoding="utf-8")
    monkeypatch.setenv("QUANTAGENT_LLM_CMD", f"{sys.executable} {script}")
    res = CliRunner().invoke(backtest_group, ["hipotesis", "proponer", "--carpeta", str(prop_dir)])
    assert res.exit_code == 0 and f"ACEPTADA prop-ok: {prop_dir / 'prop-ok.yaml'}" in res.output
    assert "RECHAZADA prop-sin-doi: fuente sin DOI válido" in res.output and "RECHAZADA prop-dup: parecida a momentum-12m" in res.output
    assert [p.name for p in prop_dir.glob("*.yaml")] == ["prop-ok.yaml"] and all(p not in stdin_file.read_text().lower() for p in ("sharpe", "dsr", "pnl"))
    assert not (tmp_path / "hipotesis.csv").exists() and not (tmp_path / "intentos.csv").exists()


@pytest.mark.parametrize("cmd,esperado", [("", "QUANTAGENT_LLM_CMD no está definido"), (f"{sys.executable} -c 'import sys; sys.stderr.write(\"fallo de red\\n\"); sys.exit(2)'", "fallo de red")])
def test_proponer_cli_comando_falla_no_escribe_nada(tmp_path, monkeypatch, cmd, esperado):
    monkeypatch.setenv("QUANTAGENT_LLM_CMD", cmd)
    res = CliRunner().invoke(backtest_group, ["hipotesis", "proponer", "--carpeta", str(tmp_path / "propuestas")])
    assert res.exit_code != 0 and esperado in res.output and not list(tmp_path.rglob("*.yaml"))


def test_propuestas_aprobar_rechazar_y_tanda_ignora_no_aprobadas(tmp_path, monkeypatch):
    import csv
    from datetime import datetime
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path / "datos"))
    prop_dir, hip_dir, runner = tmp_path / "catalogo/propuestas", tmp_path / "hipotesis", CliRunner()
    escribir_propuestas(yaml.safe_dump({"propuestas": [
        {"id": f"prop-{s}", "idea": f"idea-{s}", "clase": "momentum-12m", "ventana": {"lookback": [63 + i]}, "tema": "tendencia", "fuente_de_retorno": "prima-de-riesgo", "hipotesis_economica": "Carry corto en bonos.",
         "fuente": ["10.1016/j.jfineco.2011.11.003"], "cita_texto": ["Time Series Momentum"], "universo": {"activos": ["IEF", "TLT"], "criterio": "Bonos"}, "parametros_libres": 0}
        for i, s in enumerate(("ok", "no", "pendiente"))
    ]}), prop_dir)
    r_ok = runner.invoke(backtest_group, ["hipotesis", "propuestas", "aprobar", "prop-ok", "--quien", "fede", "--propuestas-dir", str(prop_dir), "--hipotesis-dir", str(hip_dir)])
    assert r_ok.exit_code == 0 and "APROBADA prop-ok" in r_ok.output and not (prop_dir / "prop-ok.yaml").exists() and (hip_dir / "prop-ok.yaml").exists() and not (hip_dir / "prop-pendiente.yaml").exists()
    r_no = runner.invoke(backtest_group, ["hipotesis", "propuestas", "rechazar", "prop-no", "--quien", "fede", "--motivo", "universo redundante", "--propuestas-dir", str(prop_dir)])
    assert r_no.exit_code == 0 and not (prop_dir / "prop-no.yaml").exists() and not (hip_dir / "prop-no.yaml").exists()
    assert yaml.safe_load((prop_dir / "rechazadas/prop-no.yaml").read_text(encoding="utf-8"))["motivo_rechazo"] == "universo redundante"
    filas = list(csv.DictReader((prop_dir / "registro.csv").read_text(encoding="utf-8").splitlines()))
    assert [(f["id"], f["decision"], f["quien"], f["motivo"]) for f in filas] == [("prop-ok", "APROBADA", "fede", ""), ("prop-no", "RECHAZADA", "fede", "universo redundante")]
    assert all(datetime.fromisoformat(f["fecha_utc"]) for f in filas)
    corridas = []
    monkeypatch.setattr(hipotesis.correr_command, "callback", lambda archivo, snapshot_name, carpeta: corridas.append(cargar_hipotesis(archivo).id))
    r_tanda = runner.invoke(backtest_group, ["hipotesis", "tanda", str(hip_dir), "--snapshot", "snap", "--carpeta", str(tmp_path / "salida")])
    assert r_tanda.exit_code == 0 and corridas == ["prop-ok"]


@pytest.mark.parametrize("args,err", [(["rechazar", "prop-1", "--quien", "fede"], "--motivo"), (["rechazar", "prop-1", "--quien", "fede", "--motivo", "   "], "--motivo"), (["rechazar", "prop-1", "--motivo", "x"], "--quien"), (["rechazar", "no-existe", "--quien", "fede", "--motivo", "x"], "no-existe"), (["aprobar", "prop-1"], "--quien"), (["aprobar", "prop-1", "--quien", "   "], "--quien"), (["aprobar", "no-existe", "--quien", "fede"], "no-existe")])
def test_propuestas_rechazar_sin_motivo_o_inexistente_falla(tmp_path, monkeypatch, args, err):
    monkeypatch.setattr(hipotesis, "HIPOTESIS_DIR", tmp_path / "hipotesis")
    (prop_dir := tmp_path / "propuestas").mkdir()
    (prop_dir / "prop-1.yaml").write_text("id: prop-1\n", encoding="utf-8")
    res = CliRunner().invoke(backtest_group, ["hipotesis", "propuestas", *args, "--propuestas-dir", str(prop_dir)])
    assert res.exit_code != 0 and err in res.output and (prop_dir / "prop-1.yaml").exists() and not (tmp_path / "hipotesis").exists() and not (prop_dir / "registro.csv").exists()
