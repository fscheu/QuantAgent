"""Tests for scripts/loop/pr_por_base.py: a run only adopts a PR that targets its own base branch."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "loop" / "pr_por_base.py"


def _filtrar(prs, base):
    out = subprocess.run([sys.executable, str(SCRIPT), base], input=json.dumps(prs), capture_output=True,
                         text=True, check=True).stdout
    return json.loads(out)


def test_ignora_el_pr_mas_nuevo_de_otro_lote():
    prs = [{"number": 367, "baseRefName": "lote/m4-prep-curva-puv2"},
           {"number": 368, "baseRefName": "lote/medicion-y-generador"},
           {"number": 364, "baseRefName": "lote/medicion-y-generador"}]
    assert [p["number"] for p in _filtrar(prs, "lote/medicion-y-generador")] == [368, 364]


def test_sin_pr_de_la_base_queda_vacio():
    assert _filtrar([{"number": 367, "baseRefName": "lote/m4-prep-curva-puv2"}], "lote/medicion-y-generador") == []
