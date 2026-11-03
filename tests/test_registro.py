import json
import re
import csv
from pathlib import Path

import pytest

from quantagent.fabrica import registro


@pytest.fixture
def raiz(tmp_path, monkeypatch):
    monkeypatch.setattr("quantagent.fabrica.registro.snapshot_root", lambda: tmp_path)
    return tmp_path / "registro-hipotesis.jsonl"


def test_dos_llamadas_agregan_lineas_y_preservan_la_primera(raiz):
    registro.registrar_evento("H1", "admision", motivo="entra")
    primera = raiz.read_bytes().splitlines(keepends=True)[0]
    registro.registrar_evento("H1", "etapa", "0", "PASA")

    lineas = raiz.read_bytes().splitlines(keepends=True)
    assert len(lineas) == 2
    assert lineas[0] == primera
    assert json.loads(lineas[0])["version_esquema"] == 1


def test_estado_actual_es_ultima_etapa_e_ignora_otros_eventos(raiz):
    registro.registrar_evento("H1", "etapa", "0", "PASA")
    registro.registrar_evento("H1", "intento", datos={"snapshot": "s1"})
    registro.registrar_evento("H1", "etapa", "1", "CORTA", "sin edge")
    registro.registrar_evento("H1", "ciclo", ciclo="2026-11")

    assert registro.estado_actual("H1")["etapa"] == "1"
    assert registro.estado_actual("H1")["veredicto"] == "CORTA"


def test_sin_eventos_retorna_vacios(raiz):
    assert registro.historial("H1") == []
    assert registro.estado_actual("H1") is None
    assert registro.estados_actuales() == {}


@pytest.mark.parametrize(
    "kwargs, mensaje",
    [
        ({"etapa": "4", "veredicto": "PASA"}, "etapa inválida: '4'; debe ser una de ('0', '1', '2', '3')"),
        ({"etapa": "0", "veredicto": "OK"}, "veredicto inválido: 'OK'; debe ser uno de ('PASA', 'CORTA', 'ALARMA', 'ERROR')"),
    ],
)
def test_etapa_y_veredicto_invalidos(kwargs, mensaje, raiz):
    with pytest.raises(ValueError, match=re.escape(mensaje)):
        registro.registrar_evento("H1", "etapa", **kwargs)


def test_ciclo_invalido(raiz):
    with pytest.raises(ValueError, match="ciclo inválido"):
        registro.registrar_evento("H1", "ciclo", ciclo="2026-13")


def test_linea_corrupta_en_medio_se_ignora_y_cuenta(raiz):
    registro.registrar_evento("H1", "admision")
    with raiz.open("a", encoding="utf-8") as f:
        f.write("{mal json}\n[]\n")
    registro.registrar_evento("H2", "etapa", "0", "PASA")

    assert len(registro.historial("H1")) == 1
    assert registro.estado_actual("H2")["veredicto"] == "PASA"
    assert registro.lineas_corruptas() == 2


def test_estados_actuales_con_dos_hipotesis(raiz):
    registro.registrar_evento("H1", "etapa", "0", "PASA")
    registro.registrar_evento("H2", "etapa", "1", "CORTA")
    registro.registrar_evento("H1", "etapa", "2", "PASA")

    assert {k: v["etapa"] for k, v in registro.estados_actuales().items()} == {
        "H1": "2",
        "H2": "1",
    }


def test_abrir_ciclo_limita_a_cinco_hipotesis(raiz):
    with pytest.raises(ValueError, match="máximo 5"):
        registro.abrir_ciclo("2026-11", ["H1", "H2", "H3", "H4", "H5", "H6"])

    registro.abrir_ciclo("2026-11", ["H1", "H2", "H3", "H4", "H5"])
    assert json.loads(raiz.read_text().splitlines()[0])["datos"]["hipotesis"] == ["H1", "H2", "H3", "H4", "H5"]


def test_cerrar_sin_promovidas_guarda_recibo_y_es_inmutable(raiz):
    registro.abrir_ciclo("2026-11", ["H1"])
    registro.registrar_evento("H1", "etapa", "0", "CORTA", "filtro", ciclo="2026-11")
    recibo = registro.cerrar_ciclo("2026-11")

    assert recibo["promovidas"] == 0
    assert recibo["etapas"] == {"0": 1, "1": 0, "2": 0, "3": 0}
    assert json.loads(raiz.read_text().splitlines()[-1])["motivo"]
    with pytest.raises(ValueError, match="ya fue cerrado"):
        registro.cerrar_ciclo("2026-11")


def test_recibo_se_reconstruye_solo_desde_jsonl(raiz, tmp_path):
    registro.abrir_ciclo("2026-11", ["H1"])
    registro.registrar_evento("H1", "intento", ciclo="2026-11", datos={"snapshot": "ausente"})
    registro.registrar_evento("H1", "etapa", "3", "PASA", ciclo="2026-11")
    esperado = registro.cerrar_ciclo("2026-11")
    (tmp_path / "hipotesis.csv").unlink(missing_ok=True)
    (tmp_path / "intentos.csv").unlink(missing_ok=True)

    assert registro.recibo_ciclo("2026-11") == esperado
    assert esperado["intentos"] == {"H1": 1}
    assert esperado["promovidas"] == 1
    assert esperado["hashes_manifiesto"] == {"ausente": None}


def test_ciclos_sin_etapa3_cuenta_cierres_consecutivos_y_ignora_abiertos(raiz):
    for mes in range(1, 6):
        ciclo = f"2026-{mes:02d}"
        registro.abrir_ciclo(ciclo, [f"H{mes}"])
        registro.cerrar_ciclo(ciclo)
    registro.abrir_ciclo("2026-06", ["H6"])

    assert registro.ciclos_sin_etapa3() == 5

    registro.cerrar_ciclo("2026-06")
    assert registro.ciclos_sin_etapa3() == 6


def test_ciclo_con_pase_a_etapa3_reinicia_cuenta(raiz):
    for mes in range(1, 5):
        ciclo = f"2026-{mes:02d}"
        registro.abrir_ciclo(ciclo, [f"H{mes}"])
        registro.cerrar_ciclo(ciclo)
    registro.abrir_ciclo("2026-05", ["H5"])
    registro.registrar_evento("H5", "etapa", "3", "PASA", ciclo="2026-05")
    registro.cerrar_ciclo("2026-05")
    registro.abrir_ciclo("2026-06", ["H6"])
    registro.cerrar_ciclo("2026-06")

    assert registro.ciclos_sin_etapa3() == 1


def test_calendario_cruza_anio_y_usa_dias_configurados(raiz):
    from quantagent import default_config

    ciclos = registro.calendario_ciclos("2026-12", 2)
    assert [ciclo["ciclo"] for ciclo in ciclos] == ["2026-12", "2027-01"]
    assert ciclos[0]["propuesta"] == f"2026-12-{default_config.DIA_PROPUESTA_HIPOTESIS:02d}"
    assert ciclos[0]["embudo"] == f"2026-12-{default_config.DIA_INICIO_EMBUDO:02d}"
    assert ciclos[0]["revision_paper"] == f"2026-12-{default_config.DIA_REVISION_PAPER:02d}"
    assert ciclos[1]["propuesta"] == f"2027-01-{default_config.DIA_PROPUESTA_HIPOTESIS:02d}"


def test_calendario_informa_estados_del_registro(raiz):
    registro.abrir_ciclo("2026-01", ["H1"])
    registro.abrir_ciclo("2026-02", ["H2"])
    registro.registrar_evento("H2", "etapa", "3", "PASA", ciclo="2026-02")
    registro.cerrar_ciclo("2026-02")

    ciclos = registro.calendario_ciclos("2026-01", 3)
    assert ciclos[0]["estado"] == "abierto"
    assert ciclos[1]["estado"] == "cerrado"
    assert ciclos[1]["promovidas"] == 1
    assert ciclos[2]["estado"] == "pendiente"


def test_calendario_rechaza_desde_invalido(raiz):
    with pytest.raises(ValueError, match="ciclo inválido"):
        registro.calendario_ciclos("2026-13")


def test_cli_corte_informa_umbral_y_referencia():
    from click.testing import CliRunner
    from quantagent.cli.__main__ import cli

    ayuda = CliRunner().invoke(cli, ["registro", "corte", "--help"])
    assert ayuda.exit_code == 0
    assert "Show consecutive closed cycles without a stage 3 pass." in ayuda.output
    assert "Example: quantagent registro corte" in ayuda.output
    referencia = Path("docs/manual/reference/cli/registro.md").read_text()
    assert "quantagent registro corte" in referencia


def test_cli_ciclo_expone_help_epilog_y_referencia():
    from click.testing import CliRunner
    from quantagent.cli.__main__ import cli

    ayuda = CliRunner().invoke(cli, ["registro", "ciclo", "abrir", "--help"])
    assert ayuda.exit_code == 0
    assert "Open a cycle with up to five hypothesis IDs." in ayuda.output
    assert "Example: quantagent registro ciclo abrir 2026-11 H1 H2" in ayuda.output
    referencia = Path("docs/manual/reference/cli/registro/ciclo.md").read_text()
    assert "Example: quantagent registro ciclo cerrar 2026-11" in referencia


def _csv_importables(tmp_path):
    with (tmp_path / "hipotesis.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fecha_utc", "hipotesis", "etapa", "veredicto", "motivo"])
        for i, h in enumerate(("H1", "H2", "H3"), 1):
            writer.writerow([f"2026-01-0{i}T00:00:00+00:00", h, "0", "PASA", "inicio"])
            writer.writerow([f"2026-01-0{i}T01:00:00+00:00", h, "1", "CORTA", "final"])
    with (tmp_path / "intentos.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fecha_utc", "estrategia", "activo", "snapshot", "desde", "hasta", "costos", "tamano", "params", "trades", "pnl", "sharpe", "hipotesis", "etapa", "sharpe_base", "estado"])
        writer.writerow(["2026-01-01T00:30:00+00:00", "s", "BTC", "snap", "", "", "", "", '{"x": 1}', "2", "3.5", "0.7", "H1", "0", "", ""])
        writer.writerow(["2026-01-02T00:30:00+00:00", "cartera", "cartera", "snap", "", "", "", "", "{}", "4", "", "0.9", "H2", "1", "", ""])


def test_importar_csv_a_db_preserva_fuentes_y_es_idempotente(db_temporal, tmp_path):
    from sqlalchemy import select
    from quantagent.database import SessionLocal
    from quantagent.fabrica import registro_db
    from quantagent.models import Cycle, HypothesisAttempt, HypothesisEvent

    _csv_importables(tmp_path)
    with (tmp_path / "registro-hipotesis.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"fecha_utc":"2026-02-01T00:00:00Z", "hipotesis":"H9", "evento":"admision",
            "etapa":None,"veredicto":None,"motivo":"", "ciclo":None,"datos":{"origen":"jsonl"}})+"\n")
        f.write(json.dumps({"fecha_utc":"2026-02-02T00:00:00Z", "hipotesis":None, "evento":"ciclo",
            "etapa":None,"veredicto":None,"motivo":"", "ciclo":"2026-02",
            "datos":{"accion":"cerrar","recibo":{"hipotesis":["H9"],"promovidas":0}}})+"\n")
    result = registro.importar_csv(datos=tmp_path)
    assert result["etapas"] == {"leidos": 6, "agregados": 6}
    assert result["intentos"] == {"leidos": 2, "agregados": 2}
    assert result["eventos"] == {"leidos": 2, "agregados": 2}
    with SessionLocal() as db:
        assert len(db.scalars(select(HypothesisEvent)).all()) == 8
        attempts = db.scalars(select(HypothesisAttempt)).all()
        assert len(attempts) == 2
        assert attempts[0].extra == {"desde":"", "hasta":"", "costos":"", "tamano":None, "sharpe_base":""}
        cycle = db.get(Cycle, "2026-02")
        assert cycle.closed_at is not None and cycle.receipt["hipotesis"] == ["H9"]
    assert len(registro_db.intentos(valida_solo=False)) == 2
    assert registro_db.recibo_ciclo("2026-02")["promovidas"] == 0
    again = registro.importar_csv(datos=tmp_path)
    assert all(v["agregados"] == 0 for k, v in again.items() if isinstance(v, dict))


def test_importar_csv_dry_run_y_filas_invalidas(db_temporal, tmp_path):
    from sqlalchemy import func, select
    from click.testing import CliRunner
    from quantagent.cli.__main__ import cli
    from quantagent.database import SessionLocal
    from quantagent.models import HypothesisAttempt, HypothesisEvent

    _csv_importables(tmp_path)
    with (tmp_path / "intentos.csv").open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["bad-date", "s", "BTC", "snap", "d", "h", "fees", "0.4", "{bad", "1", "0", "0", "", "0", "base", "invalida_bug"])
    result = CliRunner().invoke(cli, ["registro", "importar", "--dry-run", "--datos", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "(dry-run) agregaría etapas 6 de 6" in result.output
    with SessionLocal() as db:
        assert db.scalar(select(func.count()).select_from(HypothesisEvent)) == 0
        assert db.scalar(select(func.count()).select_from(HypothesisAttempt)) == 0
    with (tmp_path / "intentos.csv").open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["2026-01-03T00:00:00Z", "s", "BTC", "snap", "d", "h", "fees", "0.4", "{}", "1", "2", "0.5", "", "0", "base", "invalida_bug"])
    imported = registro.importar_csv(datos=tmp_path)
    assert imported["omitidas"] == 1
    assert imported["intentos"]["leidos"] == 3
    rows = __import__("quantagent.fabrica.registro_db", fromlist=["intentos"]).intentos(valida_solo=False)
    assert sum(r["estado"] == "invalida_bug" for r in rows) == 1
    assert any(r["hipotesis"] == "" for r in rows)

def test_cli_calendario_muestra_ciclos_y_corte(raiz, db_temporal):
    from click.testing import CliRunner
    from quantagent.cli.__main__ import cli
    from quantagent.fabrica import registro_db

    for mes in range(1, 7):
        ciclo = f"2026-{mes:02d}"
        registro_db.abrir_ciclo(ciclo, [f"H{mes}"])
        registro_db.cerrar_ciclo(ciclo)

    resultado = CliRunner().invoke(cli, ["registro", "calendario", "2026-12"])
    assert resultado.exit_code == 0
    assert "2026-12: propuesta 2026-12-01; embudo 2026-12-08; revisión paper 2026-12-25; pendiente" in resultado.output
    assert "Corte: 6/6 ciclos cerrados consecutivos sin pasar etapa 3." in resultado.output
    assert "Toca revisar el enfoque." in resultado.output



def test_cli_ver_lista_eventos_ordenados_y_filtra_hipotesis(raiz, db_temporal):
    from click.testing import CliRunner
    from quantagent.cli.__main__ import cli
    from quantagent.fabrica import registro_db

    registro_db.registrar_evento("H1", "etapa", "1", "PASA", fecha_utc="2026-02-02T00:00:00+00:00")
    registro_db.registrar_evento("H2", "admision", fecha_utc="2026-02-01T00:00:00+00:00")
    registro_db.registrar_evento("H1", "etapa", "0", "CORTA", fecha_utc="2026-02-01T00:00:00+00:00")
    resultado = CliRunner().invoke(cli, ["registro", "ver", "--hipotesis", "H1"])
    assert resultado.exit_code == 0
    assert [line.split(" | ")[0] for line in resultado.output.splitlines()] == [
        "2026-02-01T00:00:00+00:00", "2026-02-02T00:00:00+00:00"]
    todos = CliRunner().invoke(cli, ["registro", "ver"])
    assert todos.exit_code == 0
    assert todos.output.index("H2") < todos.output.index("H1")
