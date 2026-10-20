"""QuantAgent-d4cq.42: pedido al LLM sin cifras de rendimiento y escritura en catalogo/propuestas/."""
import pytest
import yaml

from quantagent.fabrica.investigador import armar_pedido, escribir_propuestas
from quantagent.fabrica.propuestas import validar_propuesta


def test_escribir_propuestas_desde_respuesta_simulada(tmp_path):
    respuesta = yaml.safe_dump({"propuestas": [
        {"id": "prop-a", "idea": "momentum", "clase": "momentum-12m", "ventana": {"lookback": [252]}, "hipotesis_economica": "Continuación a 12m.",
         "fuente": ["10.1016/j.jfineco.2011.11.003"], "universo": {"activos": ["SPY", "QQQ"], "criterio": "ETFs de acciones"}, "parametros_libres": 0},
        {"id": "prop-b", "idea": "tendencia", "clase": "sma-cross", "hipotesis_economica": "Tendencia secular.",
         "fuente": ["10.3905/jwm.2007.674809"], "universo": {"activos": ["SPY", "TLT"], "criterio": "Multiactivo líquido"}, "parametros_libres": 0},
    ]})
    rutas, motivo = escribir_propuestas(respuesta, tmp_path / "propuestas")
    assert motivo is None and [r.name for r in rutas] == ["prop-a.yaml", "prop-b.yaml"]
    assert all(validar_propuesta(yaml.safe_load(r.read_text(encoding="utf-8"))) is None for r in rutas)
    with pytest.raises(ValueError, match="hipotesis"):
        escribir_propuestas(respuesta, tmp_path / "hipotesis")


def test_pedido_sin_cifras_de_sharpe_dsr_ni_pnl(tmp_path):
    pm, intentos = tmp_path / "postmortem", tmp_path / "intentos.csv"
    pm.mkdir()
    (pm / "c1.yaml").write_text(yaml.safe_dump({"id": "c1", "causas": ["pocos_trades"], "sharpe": 0.4321, "dsr": 0.9876, "pnl": -1234.56, "trades": 789}), encoding="utf-8")
    intentos.write_text("hipotesis,sharpe,dsr,pnl,trades\nc1,0.4321,0.9876,-1234.56,789\nc2,0.5555,0.8888,999.0,654\n", encoding="utf-8")
    pedido = armar_pedido("catalogo/primas.yaml", pm, intentos)
    assert all(e in pedido for e in ("D1", "D2", "clase", "ventana", "intentos_registrados: 2", "pocos_trades", "tendencia-series-de-tiempo"))
    assert all(p not in pedido.lower() for p in ("sharpe", "dsr", "pnl", "trades:", "0.4321", "0.9876", "1234.56", "0.5555", "0.8888", "999.0", "789", "654"))


@pytest.mark.parametrize("invalida", [":::no_yaml: [", "texto suelto", "", "propuestas: [{idea: x}]", "propuestas: [{id: p1}, {id: p1}]"])
def test_respuesta_yaml_invalido_no_escribe_nada(tmp_path, invalida):
    rutas, motivo = escribir_propuestas(invalida, tmp_path / "propuestas")
    assert rutas == [] and motivo and not list(tmp_path.rglob("*.yaml"))
