import os
import subprocess
import sqlite3
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from apps.api.app import app

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "web-preview.sh"


def test_health_identifies_branch(monkeypatch):
    monkeypatch.delenv("QUANTAGENT_PREVIEW_BRANCH", raising=False)
    assert TestClient(app).get("/health").json() == {"status": "ok"}

    monkeypatch.setenv("QUANTAGENT_PREVIEW_BRANCH", "lote/web-qa")
    assert TestClient(app).get("/health").json() == {"status": "ok", "branch": "lote/web-qa"}

    monkeypatch.setenv("QUANTAGENT_PREVIEW_BRANCH", "801")
    assert TestClient(app).get("/health").json() == {"status": "ok", "branch": "801"}


def _setup_dbs(src_dir: Path) -> Path:
    src_dir.mkdir(parents=True, exist_ok=True)
    cartera_db = src_dir / "cartera.db"
    with sqlite3.connect(cartera_db) as conn:
        conn.execute("CREATE TABLE orders (id INT, symbol TEXT)")
        conn.execute("INSERT INTO orders VALUES (1, 'SPY')")

    maestro_db = src_dir / "maestro.db"
    with sqlite3.connect(maestro_db) as conn:
        conn.execute("CREATE TABLE accounts (id TEXT PRIMARY KEY, database_url TEXT)")
        conn.execute("INSERT INTO accounts VALUES ('m4', ?)", (f"sqlite:///{cartera_db.resolve()}",))
    return maestro_db


def test_preview_prepares_branch_and_isolated_data(tmp_path):
    preview_dir = tmp_path / "preview"
    worktree_dir = preview_dir / "worktree"
    maestro_db = _setup_dbs(tmp_path / "prod")

    env = os.environ.copy()
    env.update({
        "QUANTAGENT_PREVIEW_DIR": str(preview_dir),
        "QUANTAGENT_PREVIEW_SOURCE_MAESTRO": str(maestro_db),
        "QUANTAGENT_PREVIEW_SKIP_BUILD": "1",
        "QUANTAGENT_PREVIEW_SKIP_SYSTEMD": "1",
    })

    try:
        proc = subprocess.run([str(SCRIPT), "HEAD"], env=env, capture_output=True, text=True, check=True)
        assert proc.returncode == 0
        assert worktree_dir.is_dir()
        assert (preview_dir / "data" / "maestro.db").is_file()

        env_content = (preview_dir / "preview.env").read_text()
        assert "QUANTAGENT_PREVIEW_BRANCH=HEAD" in env_content
        assert f"QUANTAGENT_DATABASE_URL=sqlite:///{preview_dir}/data/maestro.db" in env_content
        assert "PORT=8012" in env_content

        # Copia aislada: modificar la copia no altera el original
        with sqlite3.connect(preview_dir / "data" / "cartera.db") as conn:
            conn.execute("INSERT INTO orders VALUES (2, 'QQQ')")
        with sqlite3.connect(tmp_path / "prod" / "cartera.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1
    finally:
        subprocess.run(["git", "-C", str(REPO_ROOT), "worktree", "remove", "--force", str(worktree_dir)],
                       capture_output=True)


def test_preview_resolves_pr_reference(tmp_path):
    preview_dir = tmp_path / "preview"
    worktree_dir = preview_dir / "worktree"
    maestro_db = _setup_dbs(tmp_path / "prod")

    subprocess.run(["git", "-C", str(REPO_ROOT), "update-ref", "refs/pull/777/head", "HEAD"], check=True)
    env = os.environ.copy()
    env.update({
        "QUANTAGENT_PREVIEW_DIR": str(preview_dir),
        "QUANTAGENT_PREVIEW_SOURCE_MAESTRO": str(maestro_db),
        "QUANTAGENT_PREVIEW_SKIP_BUILD": "1",
        "QUANTAGENT_PREVIEW_SKIP_SYSTEMD": "1",
    })

    try:
        proc = subprocess.run([str(SCRIPT), "777"], env=env, capture_output=True, text=True, check=True)
        assert proc.returncode == 0
        env_content = (preview_dir / "preview.env").read_text()
        assert "QUANTAGENT_PREVIEW_BRANCH=777" in env_content
    finally:
        subprocess.run(["git", "-C", str(REPO_ROOT), "update-ref", "-d", "refs/pull/777/head"], capture_output=True)
        subprocess.run(["git", "-C", str(REPO_ROOT), "worktree", "remove", "--force", str(worktree_dir)],
                       capture_output=True)


def test_preview_unknown_ref_fails(tmp_path):
    preview_dir = tmp_path / "preview"
    maestro_db = _setup_dbs(tmp_path / "prod")
    env = os.environ.copy()
    env.update({
        "QUANTAGENT_PREVIEW_DIR": str(preview_dir),
        "QUANTAGENT_PREVIEW_SOURCE_MAESTRO": str(maestro_db),
        "QUANTAGENT_PREVIEW_SKIP_BUILD": "1",
        "QUANTAGENT_PREVIEW_SKIP_SYSTEMD": "1",
    })
    proc = subprocess.run([str(SCRIPT), "nonexistent-branch-xyz"], env=env, capture_output=True, text=True)
    assert proc.returncode != 0
    assert not (preview_dir / "preview.env").exists()


def test_dev_service_unit_file():
    service_path = REPO_ROOT / "deploy" / "web" / "quantagent-api-dev.service"
    assert service_path.is_file()
    content = service_path.read_text()
    assert "--port 8012" in content
    assert "--host 127.0.0.1" in content
    assert "WorkingDirectory=/tmp/quantagent-web-preview/worktree" in content
    assert "EnvironmentFile=-/tmp/quantagent-web-preview/preview.env" in content
    assert "quantagent.cli api serve" in content
    assert "paper-ops" not in content
    assert "APCA" not in content
    assert "timer" not in content.lower()


def test_preview_stop_shuts_down_unit_and_cleans_dirs(tmp_path):
    preview_dir = tmp_path / "preview"
    worktree_dir = preview_dir / "worktree"
    data_dir = preview_dir / "data"
    worktree_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    (preview_dir / "preview.env").write_text("PORT=8012\n")

    mock_systemctl = tmp_path / "mock_systemctl.sh"
    log_file = tmp_path / "systemctl.log"
    mock_systemctl.write_text(f'#!/bin/sh\necho "$@" >> "{log_file}"\nexit 0\n')
    mock_systemctl.chmod(0o755)

    env = os.environ.copy()
    env.update({
        "QUANTAGENT_PREVIEW_DIR": str(preview_dir),
        "QUANTAGENT_PREVIEW_SYSTEMCTL": str(mock_systemctl),
    })

    proc = subprocess.run([str(SCRIPT), "stop"], env=env, capture_output=True, text=True, check=True)
    assert proc.returncode == 0
    assert not worktree_dir.exists()
    assert not data_dir.exists()
    assert not (preview_dir / "preview.env").exists()

    log_content = log_file.read_text()
    assert "--user stop quantagent-api-dev.service" in log_content


def test_preview_start_installs_unit_and_restarts(tmp_path):
    preview_dir = tmp_path / "preview"
    worktree_dir = preview_dir / "worktree"
    user_units_dir = tmp_path / "systemd"
    maestro_db = _setup_dbs(tmp_path / "prod")

    mock_systemctl = tmp_path / "mock_systemctl.sh"
    log_file = tmp_path / "systemctl.log"
    mock_systemctl.write_text(f'#!/bin/sh\necho "$@" >> "{log_file}"\nexit 0\n')
    mock_systemctl.chmod(0o755)

    env = os.environ.copy()
    env.update({
        "QUANTAGENT_PREVIEW_DIR": str(preview_dir),
        "QUANTAGENT_PREVIEW_SOURCE_MAESTRO": str(maestro_db),
        "QUANTAGENT_PREVIEW_SKIP_BUILD": "1",
        "QUANTAGENT_PREVIEW_SYSTEMCTL": str(mock_systemctl),
        "QUANTAGENT_SYSTEMD_USER_DIR": str(user_units_dir),
    })

    try:
        proc = subprocess.run([str(SCRIPT), "HEAD"], env=env, capture_output=True, text=True, check=True)
        assert proc.returncode == 0
        installed_service = user_units_dir / "quantagent-api-dev.service"
        assert installed_service.is_file()
        svc_content = installed_service.read_text()
        assert f"WorkingDirectory={worktree_dir}" in svc_content
        assert f"EnvironmentFile=-{preview_dir}/preview.env" in svc_content

        log_content = log_file.read_text()
        assert "--user daemon-reload" in log_content
        assert "--user restart quantagent-api-dev.service" in log_content
    finally:
        subprocess.run(["git", "-C", str(REPO_ROOT), "worktree", "remove", "--force", str(worktree_dir)],
                       capture_output=True)
