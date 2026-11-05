import pytest
from quantagent.fabrica.registro_db import leer_recibo

@pytest.fixture
def registro_modulo(tmp_path, monkeypatch):
    from quantagent import database, settings
    url = f"sqlite:///{tmp_path / 'registro.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setattr(settings, "DATABASE_URL", url, raising=False)
    database._engine = database._SessionLocal = None
    database.init_db()
    from quantagent.fabrica import registro_db
    monkeypatch.setattr(registro_db, "snapshot_manifest_hash", lambda _: None)
    yield registro_db, None
    database._engine.dispose()
    database._engine = database._SessionLocal = None
def test_recibo_v2_resume_estados_y_etapas(registro_modulo):
    registro = registro_modulo[0]
    registro.abrir_ciclo("2026-10", ["HP", "HC", "HE", "HA", "HN"])
    for h, etapa, veredicto in (("HP", "3", "PASA"), ("HC", "1", "CORTA"),
                                ("HE", "2", "ERROR"), ("HA", "2", "ALARMA")):
        registro.registrar_evento(h, "etapa", etapa, veredicto, "motivo", ciclo="2026-10")
    receipt = registro.cerrar_ciclo("2026-10")
    assert receipt["version"] == 2
    assert {k: receipt["por_hipotesis"][k]["estado_final"] for k in ("HP", "HC", "HE", "HA", "HN")} == {
        "HP": "promovida", "HC": "cortada", "HE": "detenida", "HA": "detenida", "HN": "en_curso"}
    assert receipt["por_hipotesis"]["HN"]["etapa"] is None
    assert receipt["por_etapa"]["2"] == {"arrived": 2, "passed": 0, "cut": 0, "error": 1, "alarm": 1}
    assert set(("ciclo", "hipotesis", "etapas", "promovidas", "caidas", "intentos", "snapshots", "hashes_manifiesto")) <= receipt.keys()
    before = registro.recibo_ciclo("2026-10")
    registro.abrir_ciclo("2026-11", [])
    registro.cerrar_ciclo("2026-11")
    assert registro.recibo_ciclo("2026-10") == before
    registro.abrir_ciclo("2026-12", ["HX"])
    registro.cerrar_ciclo("2026-12")
    assert registro.recibo_ciclo("2026-12")["promovidas"] == 0
def test_leer_recibo_v1_preserva_compatibilidad_y_campos_inderivables():
    old = {"ciclo": "2026-01", "hipotesis": ["H1", "H2"], "etapas": {"0": 2, "1": 1, "2": 0, "3": 0},
           "promovidas": 0, "caidas": {"H1": {"etapa": "1", "veredicto": "CORTA", "motivo": "x"}},
           "intentos": {}, "snapshots": [], "hashes_manifiesto": {}}
    result = leer_recibo(old)
    assert result["version"] == 1 and result["por_etapa"]["0"]["arrived"] == 2
    assert result["por_etapa"]["0"]["passed"] is None
    assert result["por_hipotesis"]["H1"]["estado_final"] == "cortada"
    assert "version" not in old
