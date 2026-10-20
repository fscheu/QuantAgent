"""QuantAgent-ui1p: la ficha de postmortem clasifica causas estructurales y no filtra cifras de rendimiento."""

import pytest
import yaml

from quantagent.fabrica.hipotesis import Hipotesis
from quantagent.fabrica.postmortem import ficha_postmortem

H = Hipotesis(id="cartera-x", autor="t", justificacion="t", piernas=())


def _fila(pierna="a", trades=20, pnl=1000.0, dd=0.1):
    return {"estrategia": pierna, "desde": "2007-01-01", "hasta": "2018-12-31",
            "trades": trades, "pnl": pnl, "max_drawdown": dd}


@pytest.mark.parametrize("filas,causas", [
    ([_fila(trades=5)], ["pocos_trades"]),
    ([_fila(trades=40, pnl=3000.0)], ["costos_comen_ganancia"]),
    ([_fila(trades=200, pnl=90000.0)], ["rotacion_alta"]),
    ([_fila("a", pnl=9000.0), _fila("b", pnl=-500.0)], ["una_pierna_domina"]),
    ([_fila("a", pnl=9000.0), _fila("b", pnl=8000.0, dd=0.5)], ["drawdown"]),
    ([_fila("a", pnl=9000.0), _fila("b", pnl=8000.0)], []),
])
def test_causas(tmp_path, filas, causas):
    assert ficha_postmortem(H, filas, tmp_path)["causas"] == causas


def test_yaml_sin_cifras_de_rendimiento(tmp_path):
    filas = [dict(_fila(trades=3, pnl=1234.56), sharpe="0.4321", sharpe_tasa_0="0.4321", dsr="0.9876")]
    ficha_postmortem(H, filas, tmp_path)
    texto = (tmp_path / "cartera-x.yaml").read_text(encoding="utf-8")
    assert yaml.safe_load(texto) == {"id": "cartera-x", "causas": ["pocos_trades"]}
    for prohibido in ("sharpe", "dsr", "pnl", "1234", "0.4321", "0.9876"):
        assert prohibido not in texto.lower()
