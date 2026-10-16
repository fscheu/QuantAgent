from pathlib import Path

import click
import pytest
import yaml

from quantagent.fabrica.hipotesis import cargar_hipotesis

FIXTURE = Path(__file__).parent / "fixtures" / "hipotesis" / "rsi-valida.yaml"


def _variante(tmp_path, **cambios):
    data = {**yaml.safe_load(FIXTURE.read_text()), **cambios}
    p = tmp_path / "h.yaml"
    p.write_text(yaml.safe_dump({k: v for k, v in data.items() if v is not None}))
    return p


def test_valida_carga_y_producto_de_listas():
    h = cargar_hipotesis(FIXTURE)
    assert h.id == "rsi-sobreventa-spy" and h.activos == ("SPY",)
    assert len(h.combinaciones()) == 2 * 3
    assert {"oversold_threshold": 30.0, "rsi_period": 14} in h.combinaciones()


def test_falta_justificacion_nombra_el_campo(tmp_path):
    with pytest.raises(ValueError, match="justificacion"):
        cargar_hipotesis(_variante(tmp_path, justificacion=None))


@pytest.mark.parametrize("id_", ["RSI-Mayus", "con espacios"])
def test_id_invalido(tmp_path, id_):
    with pytest.raises(ValueError, match="id"):
        cargar_hipotesis(_variante(tmp_path, id=id_))


def test_parametro_desconocido_falla_en_validar_grilla(tmp_path):
    with pytest.raises(click.ClickException, match="desconocido"):
        cargar_hipotesis(_variante(tmp_path, parametros={"no_existe": [1, 2]}))
