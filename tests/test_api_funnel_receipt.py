import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.app import app
from quantagent import database, settings
from quantagent.fabrica import registro_db


def _registro(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'registro.db'}"
    monkeypatch.setenv("QUANTAGENT_REGISTRO_URL", url)
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setattr(settings, "DATABASE_URL", url, raising=False)
    if database._engine is not None:
        database._engine.dispose()
    database._engine = None
    database._SessionLocal = None
    database.init_db()
    return url, TestClient(app)


def _keys(interface):
    source = Path("docs/03_design/web-design-system/api-types.ts").read_text()
    match = re.search(rf"interface {interface}(?: extends (\w+))? \{{([^\n]*)(?:\n(.*?)\n\}})?", source, re.S)
    body = match.group(2) if "}" in match.group(2) else match.group(2) + (match.group(3) or "")
    keys = (set(re.findall(r"^\s{2}(\w+):", body, re.M)) if "\n" in body
            else set(re.findall(r"(?:^|;)\s*(\w+):", body)))
    return keys | (_keys(match.group(1)) if match.group(1) else set())


def test_closed_receipt_promoted_hypothesis_and_stage_counts(tmp_path, monkeypatch):
    _, client = _registro(tmp_path, monkeypatch)
    registro_db.abrir_ciclo("2026-01", ["HP"])
    for stage in ("0", "1", "2", "3"):
        registro_db.registrar_evento("HP", "etapa", stage, "PASA", ciclo="2026-01")
    registro_db.cerrar_ciclo("2026-01")

    receipt = client.get("/api/funnel/cycles/2026-01/receipt").json()

    assert receipt["partial"] is False
    assert receipt["id"] == "2026-01"
    assert re.fullmatch(r"[0-9a-f]{64}", receipt["hash"])
    assert receipt["counts"]["promoted"] == 1
    assert [(stage["passed"], stage["cut"]) for stage in receipt["stages"]] == [
        (1, 0), (1, 0), (1, 0), (1, 0)
    ]


def test_closed_receipt_counts_error_and_alarm_as_held_drops(tmp_path, monkeypatch):
    _, client = _registro(tmp_path, monkeypatch)
    registro_db.abrir_ciclo("2026-02", ["HE", "HA"])
    registro_db.registrar_evento(
        "HE", "etapa", "2", "ERROR", "falló el cálculo", ciclo="2026-02",
        datos={"measure": {"metric": "trades", "value": 1, "threshold": 5,
                           "unit": "count", "comparator": ">="}},
    )
    registro_db.registrar_evento("HA", "etapa", "2", "ALARMA", "riesgo", ciclo="2026-02")
    registro_db.cerrar_ciclo("2026-02")

    receipt = client.get("/api/funnel/cycles/2026-02/receipt").json()

    assert receipt["counts"]["held"] == 2
    assert receipt["stages"][2]["error"] == 1
    assert receipt["stages"][2]["alarm"] == 1
    assert receipt["stages"][2]["cut"] == 0
    drops = {drop["hypothesis_id"]: drop for drop in receipt["drops"]}
    assert drops["HE"]["verdict"] == "ERROR"
    assert drops["HE"]["measure"] == {
        "metric": "trades", "value": 1, "threshold": 5, "unit": "count", "comparator": ">="
    }
    assert drops["HA"]["verdict"] == "ALARMA"


def test_closed_receipt_is_stable_after_another_cycle_closes(tmp_path, monkeypatch):
    _, client = _registro(tmp_path, monkeypatch)
    registro_db.abrir_ciclo("2026-03", ["H1"])
    registro_db.registrar_evento("H1", "etapa", "0", "PASA", ciclo="2026-03")
    registro_db.cerrar_ciclo("2026-03")

    first = client.get("/api/funnel/cycles/2026-03/receipt").json()
    second = client.get("/api/funnel/cycles/2026-03/receipt").json()
    assert first == second
    first_json = json.dumps(first, sort_keys=True, separators=(",", ":"))

    registro_db.abrir_ciclo("2026-04", [])
    registro_db.cerrar_ciclo("2026-04")

    third = client.get("/api/funnel/cycles/2026-03/receipt").json()
    assert third["hash"] == first["hash"]
    assert json.dumps(third, sort_keys=True, separators=(",", ":")) == first_json


def test_receipt_matches_typescript_contract_keys(tmp_path, monkeypatch):
    _, client = _registro(tmp_path, monkeypatch)
    registro_db.abrir_ciclo("2026-05", ["HC"])
    registro_db.registrar_evento(
        "HC", "etapa", "1", "CORTA", "motivo", ciclo="2026-05",
        datos={"measure": {"metric": "sharpe", "value": 0.2, "threshold": 0.5,
                           "unit": "ratio", "comparator": ">="}},
    )
    registro_db.cerrar_ciclo("2026-05")

    receipt = client.get("/api/funnel/cycles/2026-05/receipt").json()

    assert set(receipt) == _keys("CycleReceipt")
    assert set(receipt["counts"]) == {"entered", "active", "cut", "held", "promoted"}
    assert all(set(stage) == _keys("StageCounts") | {"pending"} for stage in receipt["stages"])
    assert all(set(drop) == {"hypothesis_id", "name", "stage", "verdict", "reason", "measure"}
               for drop in receipt["drops"])
    assert set(receipt["drops"][0]["measure"]) == _keys("Measure")
    assert all(set(attempt) == {"hypothesis_id", "name", "valid", "invalid"}
               for attempt in receipt["attempts"])
    assert all(set(snapshot) == _keys("Snapshot") for snapshot in receipt["snapshots"])
