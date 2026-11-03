import json
import re
from datetime import datetime

import pytest

from quantagent.fabrica.campos_intentos import CAMPOS_INTENTOS


@pytest.fixture(params=["jsonl", "db"])
def registro_modulo(request, tmp_path, monkeypatch):
    if request.param == "jsonl":
        from quantagent.fabrica import registro
        monkeypatch.setattr("quantagent.fabrica.registro.snapshot_root", lambda: tmp_path)
        monkeypatch.setattr(registro, "snapshot_manifest_hash", lambda _: None)
        yield registro, tmp_path / "registro-hipotesis.jsonl"
        return
    from quantagent import database
    url = f"sqlite:///{tmp_path / 'registro.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    from quantagent import settings
    monkeypatch.setattr(settings, "DATABASE_URL", url, raising=False)
    database._engine = None
    database._SessionLocal = None
    database.init_db()
    from quantagent.fabrica import registro_db
    monkeypatch.setattr(registro_db, "snapshot_manifest_hash", lambda _: None)
    yield registro_db, None
    database._engine.dispose()
    database._engine = None
    database._SessionLocal = None


def test_eventos_se_apilan_y_estado_actual(registro_modulo):
    registro = registro_modulo[0]
    registro.registrar_evento("H1", "admision", motivo="entra")
    registro.registrar_evento("H1", "etapa", "0", "PASA")
    registro.registrar_evento("H1", "etapa", "1", "CORTA", "sin edge")
    assert [e["evento"] for e in _eventos(registro)] == ["admision", "etapa", "etapa"]
    assert registro.estado_actual("H1")["etapa"] == "1"
    assert registro.historial("H1")[0]["motivo"] == "entra"


def _eventos(registro):
    return registro.eventos() if hasattr(registro, "eventos") else registro._leer_eventos()[0]


# Se dejan en test_registro.py los casos ligados al JSONL, reconstrucción/importación
# CSV y CLI: registro_db no usa ese archivo ni implementa esa interfaz.


@pytest.mark.parametrize("kwargs,mensaje", [
    ({"etapa": "4", "veredicto": "PASA"}, "etapa inválida: '4'; debe ser una de ('0', '1', '2', '3')"),
    ({"etapa": "0", "veredicto": "OK"}, "veredicto inválido: 'OK'; debe ser uno de ('PASA', 'CORTA', 'ALARMA', 'ERROR')"),
])
def test_validaciones_idénticas(registro_modulo, kwargs, mensaje):
    with pytest.raises(ValueError, match=re.escape(mensaje)):
        registro_modulo[0].registrar_evento("H1", "etapa", **kwargs)


def test_ciclos_y_calendario(registro_modulo):
    registro = registro_modulo[0]
    registro.abrir_ciclo("2026-11", ["H1"])
    registro.registrar_evento("H1", "etapa", "3", "PASA", ciclo="2026-11")
    receipt = registro.cerrar_ciclo("2026-11")
    assert receipt["promovidas"] == 1
    assert registro.recibo_ciclo("2026-11") == receipt
    assert registro.calendario_ciclos("2026-11", 1)[0]["estado"] == "cerrado"
    with pytest.raises(ValueError, match="ya fue cerrado"):
        registro.cerrar_ciclo("2026-11")


def test_intentos_preservan_claves_csv_y_fechas_utc(registro_modulo):
    from quantagent.fabrica import registro_db
    if registro_modulo[0] is not registro_db:
        pytest.skip("intentos() pertenece al registro SQL")
    registro_db.registrar_intento("s", "BTC", "snap", "fees", 0.5,
        {"x": 1}, {"desde": "2026-01-01", "hasta": "2026-02-01", "trades": 2,
                   "pnl": 3.5, "sharpe": 0.7}, "H1", "0", "tasa_libre")
    row = registro_db.intentos()[0]
    assert list(row) == CAMPOS_INTENTOS
    assert (row["desde"], row["hasta"], row["costos"], row["tamano"], row["sharpe_base"]) == (
        "2026-01-01", "2026-02-01", "fees", 0.5, "tasa_libre")
    assert datetime.fromisoformat(row["fecha_utc"]).utcoffset().total_seconds() == 0


def test_invalidar_intentos_no_borra_y_valida_solo_filtra(registro_modulo):
    from quantagent.fabrica import registro_db
    if registro_modulo[0] is not registro_db:
        pytest.skip("invalidar_intentos pertenece al registro SQL")
    for snapshot in ("snap", "sintetico"):
        registro_db.registrar_intento("s", "BTC", snapshot, None, None, {},
            {"desde": "2026-01-01", "hasta": "2026-01-02", "trades": 1, "pnl": 0, "sharpe": 0}, "H1")
    before = len(registro_db.intentos(valida_solo=False))
    marked = registro_db.invalidar_intentos({"hipotesis": "H1", "desde": "2026-01-01T00:00:00+00:00"})
    assert marked == before
    assert len(registro_db.intentos(valida_solo=False)) == before
    assert all(r["estado"] == "invalida_bug" for r in registro_db.intentos(valida_solo=False))
    assert registro_db.intentos() == []


def test_intentos_expone_fechas_de_extra_como_strings(registro_modulo):
    registro_db = registro_modulo[0]
    from quantagent.fabrica import registro_db as modulo_db
    if registro_db is not modulo_db:
        pytest.skip("intentos() pertenece al registro SQL")
    registro_db.registrar_intento("s", "BTC", "snap", None, None, {"x": 1},
        {"desde": "2026-01-01", "hasta": "2026-02-01", "trades": 1, "pnl": 0, "sharpe": 0})
    row = registro_db.intentos()[0]
    assert row["desde"] == "2026-01-01" and row["hasta"] == "2026-02-01"


def test_intentos_son_compatibles_con_verdict_y_excluyen_invalidos(registro_modulo):
    registro_db = registro_modulo[0]
    from quantagent.fabrica import registro_db as modulo_db
    if registro_db is not modulo_db:
        pytest.skip("intentos() pertenece al registro SQL")
    from quantagent.backtesting.verdict import filas_validas, n_efectivo

    args = ("s", "BTC", "snap", None, None)
    row = {"desde": "2026-01-01", "hasta": "2026-02-01", "trades": 1, "pnl": 0, "sharpe": 0}
    for combo, snapshot in (({"x": 1}, "snap"), ({"x": 2}, "snap"),
                            ({"x": 1}, "snap"), ({"x": 3}, "sintetico")):
        registro_db.registrar_intento(args[0], args[1], snapshot, args[3], args[4], combo, row, "H1")
    registro_db.registrar_intento(*args, {"x": 4}, row, "H2")
    registro_db.invalidar_intentos({"hipotesis": "H2"})
    rows = registro_db.intentos(valida_solo=False)
    assert all(isinstance(r["params"], str) for r in rows)
    assert json.loads(rows[0]["params"]) == {"x": 1}
    assert len(filas_validas(rows)) == 3
    assert n_efectivo(rows) == 2


def test_paridad_n_efectivo_e_informe_con_filas_fijas(registro_modulo):
    from quantagent.fabrica import registro_db as modulo_db
    if registro_modulo[0] is not modulo_db:
        pytest.skip("el conteo SQL se valida en la variante db")

    from quantagent.backtesting.verdict import filas_validas, n_efectivo
    from quantagent.fabrica.informe import datos_informe
    from quantagent.fabrica.registro_db import registrar_evento

    resultado = {"desde": "2020-01-01", "hasta": "2021-01-01", "trades": 3, "pnl": 2.5, "sharpe": 0.8}
    for params, activo, etapa in (({"p": 1}, "SPY", "1"), ({"p": 1}, "SPY", "2"),
                                  ({"p": 2}, "SPY", "1")):
        modulo_db.registrar_intento("rsi", activo, "real", None, None, params, resultado,
                                    "H-paridad", etapa)
    registrar_evento("H-paridad", "etapa", "3", "PASA", "n_efectivo=2")
    todas = modulo_db.intentos(valida_solo=False)
    # Las tres filas son válidas; las dos primeras comparten clave efectiva.
    assert len(filas_validas(todas)) == 3
    assert n_efectivo(todas) == 2
    datos = datos_informe("H-paridad")
    assert datos["n_total"] == 3
    assert datos["n_efectivo"] == 2


def test_error_database_url_faltante_es_explicito(monkeypatch):
    from quantagent import database, settings
    from quantagent.fabrica.registro_db import intentos

    if database._engine is not None:
        database._engine.dispose()
    database._engine = None
    database._SessionLocal = None
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setattr(settings, "DATABASE_URL", "", raising=False)
    with pytest.raises(ValueError, match="DATABASE_URL no está configurada.*URL SQLAlchemy"):
        intentos()


def test_sin_eventos_retorna_vacios(registro_modulo):
    registro = registro_modulo[0]
    assert registro.historial("H1") == []
    assert registro.estado_actual("H1") is None
    assert registro.estados_actuales() == {}


def test_estados_actuales_con_dos_hipotesis(registro_modulo):
    registro = registro_modulo[0]
    registro.registrar_evento("H1", "etapa", "0", "PASA")
    registro.registrar_evento("H2", "etapa", "1", "CORTA")
    registro.registrar_evento("H1", "etapa", "2", "PASA")
    assert {k: v["etapa"] for k, v in registro.estados_actuales().items()} == {"H1": "2", "H2": "1"}


def test_eventos_etapa_ordena_y_filtra(registro_modulo):
    registro = registro_modulo[0]
    registro.registrar_evento("H1", "etapa", "1", "PASA", fecha_utc="2026-02-02T00:00:00+00:00")
    registro.registrar_evento("H2", "admision", fecha_utc="2026-02-01T00:00:00+00:00")
    registro.registrar_evento("H1", "etapa", "0", "CORTA", fecha_utc="2026-02-01T00:00:00+00:00")
    assert [e["etapa"] for e in registro.eventos_etapa("H1")] == ["0", "1"]
    assert [e["hipotesis"] for e in registro.eventos_etapa()] == ["H1", "H1"]


def test_ciclo_invalido(registro_modulo):
    with pytest.raises(ValueError, match="ciclo inválido"):
        registro_modulo[0].registrar_evento("H1", "ciclo", ciclo="2026-13")


def test_abrir_ciclo_limita_a_cinco_hipotesis(registro_modulo):
    registro = registro_modulo[0]
    with pytest.raises(ValueError, match="máximo 5"):
        registro.abrir_ciclo("2026-11", ["H1", "H2", "H3", "H4", "H5", "H6"])
    registro.abrir_ciclo("2026-11", ["H1", "H2", "H3", "H4", "H5"])
    assert next(e for e in _eventos(registro) if e["evento"] == "ciclo")["datos"]["hipotesis"] == ["H1", "H2", "H3", "H4", "H5"]


def test_cerrar_ciclo_sin_promovidas_guarda_recibo_inmutable(registro_modulo):
    registro = registro_modulo[0]
    registro.abrir_ciclo("2026-11", ["H1"])
    registro.registrar_evento("H1", "etapa", "0", "CORTA", "filtro", ciclo="2026-11")
    receipt = registro.cerrar_ciclo("2026-11")
    assert receipt["promovidas"] == 0
    assert receipt["etapas"] == {"0": 1, "1": 0, "2": 0, "3": 0}
    with pytest.raises(ValueError, match="ya fue cerrado"):
        registro.cerrar_ciclo("2026-11")


def test_ciclos_sin_etapa3_cuenta_cierres_y_ignora_abiertos(registro_modulo):
    registro = registro_modulo[0]
    for mes in range(1, 6):
        ciclo = f"2026-{mes:02d}"
        registro.abrir_ciclo(ciclo, [f"H{mes}"])
        registro.cerrar_ciclo(ciclo)
    registro.abrir_ciclo("2026-06", ["H6"])
    assert registro.ciclos_sin_etapa3() == 5
    registro.cerrar_ciclo("2026-06")
    assert registro.ciclos_sin_etapa3() == 6


def test_ciclo_con_pase_a_etapa3_reinicia_cuenta(registro_modulo):
    registro = registro_modulo[0]
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


def test_calendario_cruza_anio_y_usa_dias_configurados(registro_modulo):
    from quantagent import default_config
    ciclos = registro_modulo[0].calendario_ciclos("2026-12", 2)
    assert [c["ciclo"] for c in ciclos] == ["2026-12", "2027-01"]
    assert ciclos[0]["propuesta"] == f"2026-12-{default_config.DIA_PROPUESTA_HIPOTESIS:02d}"
    assert ciclos[0]["embudo"] == f"2026-12-{default_config.DIA_INICIO_EMBUDO:02d}"
    assert ciclos[0]["revision_paper"] == f"2026-12-{default_config.DIA_REVISION_PAPER:02d}"
    assert ciclos[1]["propuesta"] == f"2027-01-{default_config.DIA_PROPUESTA_HIPOTESIS:02d}"


def test_calendario_informa_estados(registro_modulo):
    registro = registro_modulo[0]
    registro.abrir_ciclo("2026-01", ["H1"])
    registro.abrir_ciclo("2026-02", ["H2"])
    registro.registrar_evento("H2", "etapa", "3", "PASA", ciclo="2026-02")
    registro.cerrar_ciclo("2026-02")
    ciclos = registro.calendario_ciclos("2026-01", 3)
    assert ciclos[0]["estado"] == "abierto"
    assert (ciclos[1]["estado"], ciclos[1]["promovidas"]) == ("cerrado", 1)
    assert ciclos[2]["estado"] == "pendiente"


def test_calendario_rechaza_desde_invalido(registro_modulo):
    with pytest.raises(ValueError, match="ciclo inválido"):
        registro_modulo[0].calendario_ciclos("2026-13")
