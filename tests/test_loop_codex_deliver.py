"""Tests for scripts/loop/codex_deliver.sh (QuantAgent-dhlu): the wrapper-side git step for Codex.

Real git against a local bare remote; `gh` is a stub that records its arguments. A regression here means
a Codex run ends without a PR, or commits the .loop-delivery/ scratch folder into the delivery.
"""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "loop" / "codex_deliver.sh"


@pytest.fixture
def wt(tmp_path):
    bindir, remote, work = tmp_path / "bin", tmp_path / "remote.git", tmp_path / "work"
    bindir.mkdir()
    (bindir / "gh").write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {tmp_path}/gh.args\n')
    (bindir / "gh").chmod(0o755)
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "LOOP_GITHUB_REPO": "o/r",
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.hooksPath",
           "GIT_CONFIG_VALUE_0": "/dev/null"}

    def run(*cmd, cwd=work):
        return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)

    run("git", "init", "-q", "--bare", str(remote), cwd=tmp_path)
    run("git", "clone", "-q", str(remote), str(work), cwd=tmp_path)
    run("git", "checkout", "-q", "-b", "lote/x")
    run("git", "commit", "-q", "--allow-empty", "-m", "base")
    run("git", "push", "-q", "origin", "lote/x")
    run("git", "checkout", "-q", "--detach")
    (work / "nuevo.py").write_text("x = 1\n")
    d = work / ".loop-delivery"
    d.mkdir()
    (d / "ticket").write_text("QuantAgent-ab1c\n")
    (d / "commit-msg").write_text("feat(QuantAgent-ab1c): algo\n")
    (d / "pr-body.md").write_text("Cambió: algo\n")
    (d / "labels").write_text("bloqueo\nmain\n")
    return run, tmp_path


def test_new_delivery_is_committed_pushed_and_opened_as_pr(wt):
    run, tmp = wt
    r = run("bash", str(SCRIPT), "lote/x", "nuevo")
    assert r.returncode == 0, r.stdout + r.stderr
    files = run("git", "ls-tree", "-r", "--name-only", "loop/QuantAgent-ab1c", cwd=tmp / "remote.git").stdout.split()
    assert files == ["nuevo.py"]  # the scratch folder is not part of the delivery
    args = (tmp / "gh.args").read_text().splitlines()
    assert args[:2] == ["pr", "create"] and args[args.index("--base") + 1] == "lote/x"
    assert args[args.index("--label") + 1] == "loop,bloqueo"  # only known labels pass
    assert args[args.index("--title") + 1] == "feat(QuantAgent-ab1c): algo"


@pytest.mark.parametrize("breaker", ["commit-msg", "ticket"])
def test_incomplete_or_invalid_delivery_pushes_nothing(wt, breaker):
    run, tmp = wt
    target = tmp / "work" / ".loop-delivery" / breaker
    target.unlink() if breaker == "commit-msg" else target.write_text("main; rm -rf x\n")
    r = run("bash", str(SCRIPT), "lote/x", "nuevo")
    assert r.returncode == 1
    assert run("git", "branch", "-r", "--list", "origin/loop/*").stdout.strip() == ""
    assert not (tmp / "gh.args").exists()


def test_missing_repo_variable_fails_before_pushing(wt):
    run, tmp = wt
    r = run("env", "-u", "LOOP_GITHUB_REPO", "bash", str(SCRIPT), "lote/x", "nuevo")
    assert r.returncode != 0
    assert run("git", "branch", "-r", "--list", "origin/loop/*").stdout.strip() == ""
