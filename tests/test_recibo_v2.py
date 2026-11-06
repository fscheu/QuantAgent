import hashlib
import json

import pytest

from quantagent.fabrica import registro_db


def test_cierre_sin_promovida_guarda_motivo_y_resumen_por_etapa():
    registro_db.abrir_ciclo("2026-10", ["HC"])
    registro_db.registrar_evento("HC", "etapa", "0", "PASA", ciclo="2026-10")
    receipt = registro_db.cerrar_ciclo("2026-10")

    assert receipt["promovidas"] == 0
    assert receipt["por_etapa"]["0"] == {
        "arrived": 1,
        "passed": 1,
        "cut": 0,
        "error": 0,
        "alarm": 0,
    }
    close_event = next(
        event
        for event in registro_db.eventos()
        if event["evento"] == "ciclo"
        and event["ciclo"] == "2026-10"
        and event["datos"].get("accion") == "cerrar"
    )
    assert close_event["motivo"].startswith("ninguna hipótesis promovida")


def test_error_y_alarma_en_distintas_hipotesis_son_detenidas():
    registro_db.abrir_ciclo("2026-10", ["HE", "HA"])
    registro_db.registrar_evento("HE", "etapa", "2", "ERROR", ciclo="2026-10")
    registro_db.registrar_evento("HA", "etapa", "2", "ALARMA", ciclo="2026-10")

    receipt = registro_db.cerrar_ciclo("2026-10")

    assert receipt["por_etapa"]["2"] == {
        "arrived": 2,
        "passed": 0,
        "cut": 0,
        "error": 1,
        "alarm": 1,
    }
    assert receipt["por_hipotesis"]["HE"]["estado_final"] == "detenida"
    assert receipt["por_hipotesis"]["HA"]["estado_final"] == "detenida"
    assert receipt["promovidas"] == 0
    assert not any(
        item["estado_final"] == "promovida"
        for item in receipt["por_hipotesis"].values()
    )


def test_hipotesis_sin_eventos_queda_en_curso_y_no_cuenta_como_arrived():
    registro_db.abrir_ciclo("2026-10", ["HN"])

    receipt = registro_db.cerrar_ciclo("2026-10")

    assert receipt["por_hipotesis"]["HN"]["estado_final"] == "en_curso"
    assert receipt["por_hipotesis"]["HN"]["etapa"] is None
    assert all(stage["arrived"] == 0 for stage in receipt["por_etapa"].values())


def test_recibo_y_hash_no_cambian_con_eventos_posteriores_ni_otro_ciclo():
    registro_db.abrir_ciclo("2026-10", ["HP"])
    registro_db.registrar_evento("HP", "etapa", "3", "PASA", ciclo="2026-10")
    receipt = registro_db.cerrar_ciclo("2026-10")
    before = registro_db.recibo_ciclo("2026-10")
    without_hash = {key: value for key, value in receipt.items() if key != "hash"}
    canonical = json.dumps(
        without_hash,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert receipt["hash"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    registro_db.registrar_evento("HP", "etapa", "3", "CORTA", ciclo="2026-10")
    with pytest.raises(ValueError, match="ya fue cerrado"):
        registro_db.cerrar_ciclo("2026-10")
    registro_db.abrir_ciclo("2026-11", [])
    registro_db.cerrar_ciclo("2026-11")

    assert registro_db.recibo_ciclo("2026-10") == before
    assert registro_db.recibo_ciclo("2026-10")["hash"] == receipt["hash"]


def test_cierre_con_hipotesis_promovida():
    registro_db.abrir_ciclo("2026-10", ["HP"])
    registro_db.registrar_evento("HP", "etapa", "3", "PASA", ciclo="2026-10")

    receipt = registro_db.cerrar_ciclo("2026-10")

    assert receipt["promovidas"] == 1
    assert receipt["por_hipotesis"]["HP"]["estado_final"] == "promovida"


def test_leer_recibo_v1_preserva_compatibilidad_y_hash_desconocido():
    old = {
        "ciclo": "2026-01",
        "hipotesis": ["H1", "H2"],
        "etapas": {"0": 2, "1": 1, "2": 0, "3": 0},
        "promovidas": 0,
        "caidas": {"H1": {"etapa": "1", "veredicto": "CORTA", "motivo": "x"}},
        "intentos": {},
        "snapshots": [],
        "hashes_manifiesto": {},
    }

    result = registro_db.leer_recibo(old)

    assert result["version"] == 1
    assert result["hash"] is None
    assert result["por_etapa"]["0"]["arrived"] == 2
    assert result["por_etapa"]["0"]["passed"] is None
    assert result["por_hipotesis"]["H1"]["estado_final"] == "cortada"
    assert "version" not in old
