from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from quantagent.cli.hipotesis import correr_command, hipotesis_group
from quantagent.fabrica.estado import registrar_estado


@pytest.mark.parametrize("todas_cortan", [False, True])
def test_tanda_continua_y_escribe_informes_logs_y_resumen(tmp_path, monkeypatch, todas_cortan):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path / "datos"))
    origen = tmp_path / "yaml"
    origen.mkdir()
    plantilla = Path("hipotesis/rsi-reversion-corta.yaml").read_text()
    from quantagent.fabrica.hipotesis import cargar_hipotesis
    id_original = cargar_hipotesis("hipotesis/rsi-reversion-corta.yaml").id
    for nombre in ("c", "a", "b"):
        (origen / f"{nombre}.yaml").write_text(plantilla.replace(id_original, f"tanda-{nombre}"))
    (origen / "ignorar.txt").write_text("no es YAML")
    (origen / "0-invalido.yaml").write_text("id: [")
    registrar_estado("anterior", "0", "CORTA", "fuera de tanda")
    llamadas = []

    def correr(archivo, snapshot_name, carpeta):
        nombre = Path(archivo).stem
        llamadas.append(nombre)
        assert snapshot_name == "snapshot-test"
        assert Path(carpeta) == tmp_path / "salida" / f"tanda-{nombre}"
        click.echo(f"procesando {nombre}")
        if nombre == "b" and not todas_cortan:
            raise RuntimeError("fallo deliberado")
        registrar_estado(f"tanda-{nombre}", "1", "CORTA", "corte deliberado")
        raise click.exceptions.Exit(1)

    monkeypatch.setattr(correr_command, "callback", correr)
    resultado = CliRunner().invoke(hipotesis_group, ["tanda", str(origen), "--snapshot", "snapshot-test",
                                                   "--carpeta", str(tmp_path / "salida")])
    assert resultado.exit_code == 0, resultado.output
    assert llamadas == ["a", "b", "c"]
    assert "ParserError" in (tmp_path / "salida" / "0-invalido.log").read_text()
    for nombre in llamadas:
        salida = tmp_path / "salida" / f"tanda-{nombre}"
        assert f"procesando {nombre}" in (salida / "corrida.log").read_text()
        if nombre == "b" and not todas_cortan:
            assert "RuntimeError: fallo deliberado" in (salida / "corrida.log").read_text()
        else:
            assert "Veredicto: CORTA" in (salida / "informe.html").read_text()
    resumen = (tmp_path / "salida" / "resumen.md").read_text()
    assert f"| 1 | CORTA | {3 if todas_cortan else 2} |" in resumen
    assert "fuera de tanda" not in resumen
