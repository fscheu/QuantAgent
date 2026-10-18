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


def test_falta_universo_nombra_el_campo(tmp_path):
    with pytest.raises(ValueError, match="universo"):
        cargar_hipotesis(_variante(tmp_path, universo=None))


def test_formato_viejo_carga_como_una_pierna():
    h = cargar_hipotesis(FIXTURE)
    assert len(h.piernas) == 1
    assert h.piernas[0].clase == "rsi"
    assert h.clase == "rsi"


def test_yaml_dos_piernas_carga_con_combinaciones(tmp_path):
    data = {
        "id": "cartera-dos-piernas",
        "autor": "fede",
        "justificacion": "Combinar tendencia y momentum diversifica el riesgo en ETFs.",
        "timeframe": "1d",
        "piernas": [
            {
                "clase": "rsi",
                "parametros": {"rsi_period": [7, 14], "oversold_threshold": [30.0]},
                "activos": ["SPY"],
                "universo": "Mercado accionario EEUU",
            },
            {
                "clase": "sma-cross",
                "parametros": {"fast": [20, 50], "slow": [200]},
                "activos": ["TLT"],
                "universo": "Bonos del tesoro largo plazo",
            },
        ],
    }
    p = tmp_path / "dos_piernas.yaml"
    p.write_text(yaml.safe_dump(data))
    h = cargar_hipotesis(p)
    assert len(h.piernas) == 2
    assert len(h.piernas[0].combinaciones()) == 2
    assert len(h.piernas[1].combinaciones()) == 2


@pytest.mark.parametrize("id_", ["RSI-Mayus", "con espacios"])
def test_id_invalido(tmp_path, id_):
    with pytest.raises(ValueError, match="id"):
        cargar_hipotesis(_variante(tmp_path, id=id_))


def test_parametro_desconocido_falla_en_validar_grilla(tmp_path):
    with pytest.raises(click.ClickException, match="desconocido"):
        cargar_hipotesis(_variante(tmp_path, parametros={"no_existe": [1, 2]}))


def test_cli_cartera_dos_piernas_sin_guarda_c05(tmp_path):
    from click.testing import CliRunner

    from quantagent.cli.backtest import backtest_group

    data = {
        "id": "dos-piernas",
        "autor": "fede",
        "justificacion": "Combinar tendencia y momentum diversifica el riesgo en ETFs.",
        "timeframe": "1d",
        "piernas": [
            {"clase": "rsi", "parametros": {"rsi_period": [7], "oversold_threshold": [30.0]}, "activos": ["SPY"], "universo": "EEUU"},
            {"clase": "sma-cross", "parametros": {"fast": [20], "slow": [200]}, "activos": ["TLT"], "universo": "Bonos"},
        ],
    }
    archivo = tmp_path / "h.yaml"
    archivo.write_text(yaml.safe_dump(data))
    runner = CliRunner()
    for cmd in (["hipotesis", "filtrar", str(archivo), "--snapshot", "sintetico"],
                ["hipotesis", "correr", str(archivo), "--snapshot", "sintetico", "--carpeta", str(tmp_path / "c")],
                ["hipotesis", "tanda", str(tmp_path), "--snapshot", "sintetico", "--carpeta", str(tmp_path / "t")]):
        res = runner.invoke(backtest_group, cmd)
        assert "falta C05" not in res.output
