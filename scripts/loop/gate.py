#!/usr/bin/env python3
"""Review gate for the QuantAgent nightly loop (PLAN-CONTINUACION.md §3.1 and §5).

Prints one JSON object and exits 0 (PASS: the loop may work) or 10 (WAIT: the last delivery is
not reviewed yet). Exit 2 on errors. WAIT never invokes the model.
"""

import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

EXIT_PASS, EXIT_WAIT, EXIT_ERROR = 0, 10, 2

R_LINE = re.compile(r"^R:\s*(\S+)\s+vi:\s*(.*?)\s*decido:\s*(merge|cambio|descarto)\b\s*(.*)$", re.IGNORECASE)
EMPTY_WORDS = {
    "ok", "bien", "todo", "listo", "nada", "revisado", "visto", "vi", "si", "que", "con", "los", "las",
    "del", "por", "para", "una", "uno", "sin", "muy", "esta", "este", "como", "mas", "the", "and",
}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)?", _norm(text))


def _bare_ticket(ticket: str) -> str:
    return _norm(ticket).removeprefix("quantagent-")


def parse_review(body: str, ticket: str, evidence: str) -> tuple[dict | None, str | None]:
    """Validate an `R:` line against the delivery it reviews. Returns (review, None) or (None, reason)."""
    # Read the whole comment as one line: reviewers paste command output in the middle of `vi:`.
    first = " ".join(body.split())
    match = R_LINE.match(first)
    if not match:
        return None, "no cumple el formato 'R: <ticket> vi: <...> decido: merge|cambio|descarto'"
    r_ticket, seen, decision, detail = match.groups()
    decision = decision.lower()
    if _bare_ticket(r_ticket) != _bare_ticket(ticket):
        return None, f"la R: nombra '{r_ticket}' pero la entrega es '{ticket}'"
    words = _tokens(seen)
    if len(words) < 3:
        return None, "'vi:' tiene menos de 3 palabras"
    meaningful = [w for w in words if w not in EMPTY_WORDS and (len(w) >= 3 or any(c.isdigit() for c in w))]
    present = set(_tokens(evidence))
    if not any(w in present for w in meaningful):
        return None, "'vi:' no menciona nada que aparezca en la entrega"
    if decision in ("cambio", "descarto") and not detail.strip():
        return None, f"'decido: {decision}' necesita decir qué o por qué"
    return {"ticket": r_ticket, "vi": seen.strip(), "decision": decision, "detail": detail.strip(), "line": first}, None


def evaluate(prs: list[dict], reviewer: str, get_detail) -> tuple[int, dict]:
    """Decide PASS/WAIT from the loop PRs. `get_detail(number)` returns commits, comments, reviews, body, diff."""
    if not prs:
        return EXIT_PASS, {"result": "PASS", "mode": "nuevo", "message": "sin entregas previas"}
    pr = max(prs, key=lambda p: p["createdAt"])
    detail = get_detail(pr["number"])
    branch = pr.get("headRefName", "")
    ticket = branch.removeprefix("loop/") if branch.startswith("loop/") else branch
    base = {"pr": pr["number"], "url": pr.get("url", ""), "ticket": ticket, "branch": branch, "state": pr["state"]}
    last_commit = max((c["committedDate"] for c in detail["commits"]), default=pr["createdAt"])

    candidates = [
        c for c in detail["comments"] + detail["reviews"]
        if (c.get("author") or {}).get("login") == reviewer
        and c.get("createdAt", c.get("submittedAt", "")) > last_commit
        and re.match(r"^\s*R:", c.get("body") or "")
    ]
    if not candidates:
        return EXIT_WAIT, {**base, "result": "WAIT", "message": f"⏸ esperando revisión: {base['url']} (desde {last_commit[:10]})"}

    latest = max(candidates, key=lambda c: c.get("createdAt", c.get("submittedAt", "")))
    evidence = "\n".join([pr.get("title", ""), detail.get("body", ""), detail.get("diff", "")])
    review, reason = parse_review(latest["body"], ticket, evidence)
    if review is None:
        return EXIT_WAIT, {**base, "result": "WAIT", "message": f"⏸ R inválida en {base['url']}: {reason}"}

    state, decision = pr["state"], review["decision"]
    out = {**base, "review": review}
    if decision == "merge" and state == "MERGED":
        return EXIT_PASS, {**out, "result": "PASS", "mode": "nuevo", "previous": "cerrar"}
    if decision == "descarto" and state == "CLOSED":
        return EXIT_PASS, {**out, "result": "PASS", "mode": "nuevo", "previous": "descartar"}
    if decision == "cambio" and state == "OPEN":
        return EXIT_PASS, {**out, "result": "PASS", "mode": "cambio", "previous": "cambio"}
    if decision == "merge" and state == "OPEN":
        return EXIT_WAIT, {**out, "result": "WAIT", "message": f"⏸ la R dice merge pero falta mergear {base['url']}"}
    return EXIT_WAIT, {**out, "result": "WAIT", "message": f"⏸ combinación sin regla: decido {decision} con PR {state} ({base['url']})"}


def _gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def _load_config() -> dict:
    cfg = {}
    for line in (Path(__file__).resolve().parents[2] / "loop" / "config.env").read_text().splitlines():
        m = re.match(r'^([A-Z_]+)="?(.*?)"?$', line.strip())
        if m:
            cfg[m.group(1)] = m.group(2)
    return cfg


def main() -> int:
    try:
        cfg = _load_config()
        fields = "number,state,createdAt,headRefName,url,title"
        repo = ("-R", cfg["LOOP_GITHUB_REPO"])  # explicit: the launcher runs the gate outside any git checkout
        prs = json.loads(_gh("pr", "list", *repo, "--label", cfg["LOOP_LABEL"], "--state", "all", "--limit", "30",
                             "--json", fields))

        def get_detail(number: int) -> dict:
            d = json.loads(_gh("pr", "view", str(number), *repo, "--json", "commits,comments,reviews,body"))
            d["diff"] = _gh("pr", "diff", str(number), *repo)
            return d

        code, result = evaluate(prs, cfg["LOOP_REVIEWER"], get_detail)
    except Exception as exc:  # noqa: BLE001 - any failure must stop the loop, never let it run
        print(json.dumps({"result": "ERROR", "message": f"⚠ gate error: {exc}"}, ensure_ascii=False))
        return EXIT_ERROR
    print(json.dumps(result, ensure_ascii=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
