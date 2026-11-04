import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "deploy/web/registro"


def test_registro_operations_files_are_read_only_and_scripts_parse(tmp_path):
    sql = (OPS / "crear_usuario_lectura.sql").read_text().upper()
    for table in ("HYPOTHESES", "HYPOTHESIS_EVENTS", "HYPOTHESIS_ATTEMPTS", "CYCLES"):
        assert table in sql
    for forbidden in ("INSERT", "UPDATE", "DELETE", "ALL"):
        assert forbidden not in sql
    backup = OPS / "backup_registro.sh"
    for script in (backup, OPS / "probar_restauracion.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)
    env = os.environ.copy()
    env.pop("QUANTAGENT_REGISTRO_ADMIN_URL", None)
    env.pop("QUANTAGENT_REGISTRO_CONTAINER", None)
    result = subprocess.run([str(backup)], env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "QUANTAGENT_REGISTRO_ADMIN_URL" in result.stderr
