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
    diagnosticos = []
    from quantagent.fabrica import informe as modulo_informe
    monkeypatch.setattr(modulo_informe, "diagnostico_exceso", lambda h, carpeta, snapshot: (
        diagnosticos.append((carpeta, snapshot)) or {"disponible": False, "motivo": "sin curva"}))

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
    assert diagnosticos == [(str(tmp_path / "salida" / f"tanda-{n}" / "equity"), "snapshot-test")
                            for n in llamadas]
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


@pytest.mark.parametrize("estados", [
    [("1", "CORTA"), ("2", "ALARMA"), ("3", "ERROR"), None],
    [("3", "PASA"), ("1", "PASA"), ("0", "PASA"), ("2", "PASA")],
    [("0", "CORTA"), ("3", "CORTA"), ("0", "ALARMA"), ("3", "ALARMA")],
])
def test_tanda_solo_corre_estados_no_finales(tmp_path, monkeypatch, estados):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path / "datos"))
    origen = tmp_path / "yaml"
    origen.mkdir()
    from quantagent.fabrica.hipotesis import cargar_hipotesis
    plantilla = Path("hipotesis/rsi-reversion-corta.yaml").read_text()
    id_original = cargar_hipotesis("hipotesis/rsi-reversion-corta.yaml").id
    finales = []
    for i, estado in enumerate(estados):
        nombre = f"tanda-{i}"
        (origen / f"{nombre}.yaml").write_text(plantilla.replace(id_original, nombre))
        if estado:
            registrar_estado(nombre, "0", "PASA", "estado anterior")
            registrar_estado(nombre, *estado, "último estado")
        if estado and (estado[1] in ("CORTA", "ALARMA") or estado == ("3", "PASA")):
            finales.append(nombre)
    registros = (tmp_path / "datos" / "hipotesis.csv").read_bytes()
    llamadas = []

    def correr(archivo, snapshot_name, carpeta):
        llamadas.append(Path(archivo).stem)
        click.echo("ejecutada")

    monkeypatch.setattr(correr_command, "callback", correr)
    resultado = CliRunner().invoke(hipotesis_group, ["tanda", str(origen), "--snapshot", "sin-lectura",
                                                   "--carpeta", str(tmp_path / "salida")])
    assert resultado.exit_code == 0, resultado.output
    assert llamadas == [f"tanda-{i}" for i in range(4) if f"tanda-{i}" not in finales]
    assert (tmp_path / "datos" / "hipotesis.csv").read_bytes() == registros
    assert not (tmp_path / "datos" / "intentos.csv").exists()
    for i, estado in enumerate(estados):
        salida = tmp_path / "salida" / f"tanda-{i}"
        log = (salida / "corrida.log").read_text()
        if f"tanda-{i}" in finales:
            assert f"ya evaluada: etapa {estado[0]} {estado[1]}; se saltea" in log
            assert f"Veredicto: {estado[1]}" in (salida / "informe.html").read_text()
        else:
            assert "ejecutada" in log
