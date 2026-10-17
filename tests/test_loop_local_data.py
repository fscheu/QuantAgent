"""Exercise the wrapper's actual permission block and Codex command without launching an agent."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("agent,enabled", [("codex", True), ("codex", False), ("agy", True), ("claude", True)])
def test_local_data_one_shot(tmp_path, agent, enabled):
    wrapper = Path("scripts/loop/run_nightly.sh").read_text()
    setup = wrapper.split('export QUANTAGENT_SNAPSHOT_DIR=', 1)[1].split('\npython -m alembic', 1)[0]
    setup = 'export QUANTAGENT_SNAPSHOT_DIR=' + setup
    launch = wrapper.split('    timeout "${LOOP_TIMEOUT_MIN}m" "${LOOP_CODEX_BIN:-codex}"', 1)[1]
    launch = 'timeout "${LOOP_TIMEOUT_MIN}m" "${LOOP_CODEX_BIN:-codex}"' + launch.split('\n    LOOP_GITHUB_REPO=', 1)[0]
    if agent != "codex":
        launch = wrapper.rsplit(f"  {agent})\n", 1)[1].split(";;", 1)[0]
    flag = tmp_path / "next-local-data"
    if enabled:
        flag.touch()
    fake = tmp_path / agent
    fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    fake.chmod(0o755)
    log = tmp_path / "run.log"
    snapshot = str(tmp_path / "local data")
    env = dict(os.environ, LOOP_LOG_DIR=str(tmp_path), LOG=str(log), AGENT=agent,
               LOOP_SNAPSHOT_DIR=snapshot, LOOP_CODEX_BIN=str(fake), LOOP_TIMEOUT_MIN="1",
               MODEL="test", REPO=str(tmp_path), PROMPT="test", SELF_DIR="scripts/loop",
               PATH=str(tmp_path) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(["bash", "-uc", setup + '\n' + launch], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not flag.exists()
    expected = ["exec", "-m", "test", "-s", "workspace-write", "-c",
                "sandbox_workspace_write.network_access=true", "-c", "shell_environment_policy.inherit=all",
                "--add-dir", str(tmp_path / ".beads")]
    if enabled and agent == "codex":
        expected += ["--add-dir", snapshot]
        assert f"Codex: --add-dir {snapshot}" in log.read_text()
    elif enabled:
        assert f"next-local-data ignorado por {agent}" in log.read_text()
        assert "--add-dir" not in log.read_text()
        assert log.read_text().splitlines()[1:3] == ["-p", "test"]
    if agent == "codex":
        assert log.read_text().splitlines()[-len(expected) - 1:] == expected + ["-"]
