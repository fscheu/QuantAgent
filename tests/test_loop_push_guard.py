"""Tests for the loop's pre-push guard (scripts/loop/push_guard.sh, QuantAgent-l4n).

Each test does a real `git push` against a local bare repo with the guard installed the same way
run_nightly.sh installs it (core.hooksPath through GIT_CONFIG_* env). A regression here means a loop
agent could push to main, delete a branch or force-push, since main has no branch protection.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[1] / "scripts" / "loop" / "push_guard.sh"


@pytest.fixture
def repo(tmp_path):
    """Clone of a bare remote with one commit on main; returns (run, remote_sha) helpers."""
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    shutil.copy(GUARD, hooks / "pre-push")
    (hooks / "pre-push").chmod(0o755)
    env = {**os.environ, "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.hooksPath",
           "GIT_CONFIG_VALUE_0": str(hooks), "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    env.pop("LOOP_ORIG_HOOKS", None)
    remote, work = tmp_path / "remote.git", tmp_path / "work"

    def run(*args, cwd=work):
        return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True)

    assert run("init", "-q", "--bare", "-b", "main", str(remote), cwd=tmp_path).returncode == 0
    assert run("clone", "-q", str(remote), str(work), cwd=tmp_path).returncode == 0
    run("checkout", "-q", "-b", "main")
    run("commit", "-q", "--allow-empty", "-m", "base")
    # Seed main on the remote without the guard: the guard itself must reject any push to main.
    seed = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "push", "-q", "origin", "main"],
                          cwd=work, env={k: v for k, v in env.items() if not k.startswith("GIT_CONFIG")},
                          capture_output=True, text=True)
    assert seed.returncode == 0, seed.stderr

    def remote_sha(ref):
        return run("rev-parse", "-q", "--verify", ref, cwd=remote).stdout.strip()

    return run, remote_sha


def _commit(run, msg):
    assert run("commit", "-q", "--allow-empty", "-m", msg).returncode == 0


@pytest.mark.parametrize("refspec", ["main", "HEAD:main", "loop/x:main"])
def test_push_to_main_is_rejected_and_remote_main_does_not_move(repo, refspec):
    run, remote_sha = repo
    before = remote_sha("refs/heads/main")
    run("checkout", "-q", "-b", "loop/x")
    _commit(run, "work")
    run("branch", "-f", "main", "loop/x")
    res = run("push", "origin", refspec)
    assert res.returncode != 0
    assert "push a main bloqueado" in res.stderr
    assert remote_sha("refs/heads/main") == before


def test_new_branch_and_fast_forward_pushes_pass(repo):
    run, remote_sha = repo
    run("checkout", "-q", "-b", "loop/x")
    _commit(run, "first")
    assert run("push", "-u", "origin", "loop/x").returncode == 0
    _commit(run, "second")
    assert run("push", "origin", "loop/x").returncode == 0
    assert remote_sha("refs/heads/loop/x") == run("rev-parse", "HEAD").stdout.strip()


def test_force_push_of_rewritten_history_is_rejected(repo):
    run, remote_sha = repo
    run("checkout", "-q", "-b", "loop/x")
    _commit(run, "first")
    assert run("push", "-u", "origin", "loop/x").returncode == 0
    pushed = remote_sha("refs/heads/loop/x")
    assert run("commit", "-q", "--amend", "--allow-empty", "-m", "rewritten").returncode == 0
    res = run("push", "--force", "origin", "loop/x")
    assert res.returncode != 0
    assert "push forzado" in res.stderr
    assert remote_sha("refs/heads/loop/x") == pushed


def test_remote_branch_deletion_is_rejected(repo):
    run, remote_sha = repo
    run("checkout", "-q", "-b", "loop/x")
    _commit(run, "first")
    assert run("push", "-u", "origin", "loop/x").returncode == 0
    res = run("push", "origin", "--delete", "loop/x")
    assert res.returncode != 0
    assert "borrar ramas" in res.stderr
    assert remote_sha("refs/heads/loop/x") != ""


def test_guard_chains_to_the_original_pre_push_hook(repo, tmp_path):
    """The repo's own pre-push hook (beads) must still run, and its failure must block the push."""
    run, remote_sha = repo
    orig = tmp_path / "orig-hooks"
    orig.mkdir()
    marker = tmp_path / "orig-ran"
    (orig / "pre-push").write_text(f"#!/usr/bin/env bash\ncat > {marker}\nexit 3\n")
    (orig / "pre-push").chmod(0o755)
    run("checkout", "-q", "-b", "loop/x")
    _commit(run, "first")
    res = subprocess.run(["git", "push", "-u", "origin", "loop/x"], cwd=tmp_path / "work",
                         capture_output=True, text=True,
                         env={**os.environ, "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.hooksPath",
                              "GIT_CONFIG_VALUE_0": str(tmp_path / "hooks"), "LOOP_ORIG_HOOKS": str(orig)})
    assert res.returncode != 0
    assert "refs/heads/loop/x" in marker.read_text()
    assert remote_sha("refs/heads/loop/x") == ""
