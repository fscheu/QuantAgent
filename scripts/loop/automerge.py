#!/usr/bin/env python3
"""Merge reviewed loop deliveries into lot branches (PLAN-CONTINUACION.md §3.7)."""

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _date(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _latest_pm(comments, reviewer, ticket):
    candidates = [
        c for c in comments
        if c.get("user", {}).get("login") == reviewer
        and re.match(r"^PM:\s*" + re.escape(ticket) + r"(?=\s|$)", c.get("body") or "")
    ]
    return max(candidates, key=lambda c: (_date(c["created_at"]), c["id"]), default=None)


def decide(pr, comments, check_runs, reviewer, lot_prefix):
    """Return (allowed, reason) for REST API data; pr includes last_commit_at."""
    try:
        base = pr["base"]["ref"]
        if not lot_prefix or not base.startswith("lote/") or not base.startswith(lot_prefix):
            return False, "la base no es una rama de lote"
        if not re.fullmatch(r"loop/QuantAgent-[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*", pr["head"]["ref"]):
            return False, "el head no es loop/QuantAgent-<id>"
        if pr["head"].get("repo", {}).get("full_name") != pr["base"].get("repo", {}).get("full_name"):
            return False, "el head viene de un fork"
        if pr["state"] != "open":
            return False, "el PR no está abierto"
        if pr["draft"] is not False:
            return False, "el PR es borrador"
        if pr["mergeable"] is not True or pr.get("mergeable_state") in {"dirty", "CONFLICTING"}:
            return False, "mergeabilidad en conflicto o indeterminada"
        ticket = pr["head"]["ref"].removeprefix("loop/")
        pm = _latest_pm(comments, reviewer, ticket)
        if pm is None:
            return False, "falta comentario PM del reviewer para este ticket"
        decisions = re.findall(r"decido:\s*(\w+)", pm["body"])
        if not decisions or decisions[-1] != "merge":
            return False, "la última decisión PM no es merge"
        if _date(pm["created_at"]) <= _date(pr["last_commit_at"]):
            return False, "el comentario PM no es posterior al último commit"
        if not any(c["name"] == "CI (Lint + Tests)" and c["status"] == "completed"
                   and c["conclusion"] == "success" for c in check_runs):
            return False, "CI (Lint + Tests) no está en success"
        for check in check_runs:
            if check["status"] != "completed" or check["conclusion"] not in {"success", "skipped", "neutral"}:
                return False, f"check sin aprobar: {check['name']}"
        return True, f"PM autoriza merge: {pm['html_url']}; checks aprobados"
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        return False, f"datos incompletos o inválidos: {exc}"


def _api(endpoint, method="GET", payload=None, paginate=False):
    command = ["gh", "api", "-X", method, endpoint]
    if paginate:
        command += ["--paginate", "--slurp"]
    if payload is not None:
        command += ["--input", "-"]
    return json.loads(subprocess.run(
        command, input=json.dumps(payload) if payload is not None else None,
        check=True, capture_output=True, text=True,
    ).stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo")
    args = parser.parse_args()
    try:
        config = {}
        for line in (Path(__file__).resolve().parents[2] / "loop/config.env").read_text().splitlines():
            tokens = shlex.split(line, comments=True)
            if tokens:
                key, value = tokens[0].split("=", 1)
                config[key] = value
        repo = args.repo or config["LOOP_GITHUB_REPO"]
        endpoint = f"repos/{repo}/pulls/{args.pr}"
        pr = _api(endpoint)
        sha = pr["head"]["sha"]
        commits = [c for page in _api(endpoint + "/commits?per_page=100", paginate=True) for c in page]
        pr["last_commit_at"] = next(c["commit"]["committer"]["date"] for c in commits if c["sha"] == sha)
        deadline = time.monotonic() + 300
        while True:
            checks = [c for page in _api(f"repos/{repo}/commits/{sha}/check-runs?per_page=100", paginate=True)
                      for c in page["check_runs"]]
            ci_seen = any(c["name"] == "CI (Lint + Tests)" for c in checks)
            if (ci_seen and not any(c.get("status") != "completed" for c in checks)) or time.monotonic() >= deadline:
                break
            time.sleep(min(20, max(0, deadline - time.monotonic())))
        # Refresh mutable PR state and comments after polling; never authorize a different head.
        fresh = _api(endpoint)
        for _ in range(5):
            if fresh.get("mergeable") is not None:
                break
            time.sleep(5)  # GitHub computes mergeability lazily
            fresh = _api(endpoint)
        if fresh["head"]["sha"] != sha:
            print("No merge: el head cambió durante la evaluación")
            return 0
        fresh["last_commit_at"] = pr["last_commit_at"]
        comments = [c for page in _api(f"repos/{repo}/issues/{args.pr}/comments?per_page=100", paginate=True)
                    for c in page]
        allowed, reason = decide(fresh, comments, checks, config["LOOP_REVIEWER"], config["LOOP_LOT_PREFIX"])
        print(f"{'Merge' if allowed else 'No merge'}: {reason}" + (" (dry-run)" if args.dry_run else ""))
        if not allowed or args.dry_run:
            return 0
        result = _api(endpoint + "/merge", "PUT", {"merge_method": "merge", "sha": sha})
        if not result.get("merged"):
            raise RuntimeError(f"GitHub no mergeó: {result.get('message')}")
        pm = _latest_pm(comments, config["LOOP_REVIEWER"], fresh["head"]["ref"].removeprefix("loop/"))
        summary = "\n".join(f"- {c['name']}: {c['conclusion']}" for c in checks)
        _api(f"repos/{repo}/issues/{args.pr}/comments", "POST", {
            "body": "Mergeado por la Action de auto-merge (§3.7)\n\n"
                    f"PM usado: {pm['html_url']}\n\nChecks del head `{sha}`:\n{summary}",
        })
        return 0
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError, StopIteration, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
