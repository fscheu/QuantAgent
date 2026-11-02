"""Run the actual generated e2e/report blocks, isolating only external commands."""
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("changed,status", [(0, 0), (1, 0), (1, 7)])
def test_web_e2e_report(tmp_path, changed, status):
    source = Path("scripts/loop/codex_review.sh").read_text()
    block = source.split("# E2E runs through try.sh;", 1)[1].split("\ntimeout '${LOOP_REVIEW_TIMEOUT_MIN}m' '${LOOP_CODEX_BIN", 1)[0]
    report = source[source.index("printf ", source.index("sed -i -E")):].split("\n[ -n '${LOOP_REVIEW_DRY", 1)[0]
    block = "# E2E runs through try.sh;" + block + "\n" + report
    script = tmp_path / "scripts/loop/try.sh"
    script.parent.mkdir(parents=True)
    script.write_text('printf "%s\\n" "$1" > called\nbash -c "$3"\n')
    web = tmp_path / "web"
    web.mkdir()
    npm = tmp_path / "npm"
    npm.write_text('#!/bin/bash\nif [ "$1" = ci ]; then exit 0; fi\nmkdir -p e2e-artifacts/390px\necho screenshot > e2e-artifacts/390px/screen.png\necho "e2e exit $TEST_STATUS"\nexit "$TEST_STATUS"\n')
    npm.chmod(0o755)
    env = dict(os.environ, WT=str(tmp_path), STATE=str(tmp_path), PR="42", WEB_CHANGED=str(changed),
               BRANCH="loop/test", LOOP_REVIEW_TIMEOUT_MIN="1", OUT=str(tmp_path / "report"),
               TEST_STATUS=str(status), PATH=str(tmp_path) + os.pathsep + os.environ["PATH"])
    rendered = subprocess.run(["bash", "-uc", "cat <<RUN\n" + block + "\nRUN"], env=env, text=True, capture_output=True, check=True)
    subprocess.run(["bash", "-uc", rendered.stdout], cwd=tmp_path, env=env, check=True)
    output = (tmp_path / "report").read_text()
    if changed:
        assert f"exit {status}" in output
        assert (tmp_path / "called").read_text().strip() == "loop/test"
        assert (tmp_path / "review-42-e2e-artifacts/390px/screen.png").read_text() == "screenshot\n"
        assert "Mirá los screenshots de 390px" in (tmp_path / ".review-prompt.txt").read_text()
    else:
        assert "No aplica" in output
        assert not (tmp_path / "called").exists()
