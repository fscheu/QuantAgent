"""Tests for the nightly loop review gate (scripts/loop/gate.py, PLAN-CONTINUACION.md §3.1 and §5).

Each test builds the GitHub data the gate would read and checks PASS/WAIT. A regression here means
the loop could start work on an unreviewed delivery, or stay blocked after a valid review.
"""

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

_spec = importlib.util.spec_from_file_location(
    "loop_gate", Path(__file__).resolve().parents[1] / "scripts" / "loop" / "gate.py"
)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

REVIEWER = "fscheu"
BODY = "Cambió: la salida del CLI queda en 5 líneas\nObtenido: Trades: 227 Sharpe 0.40"
DIFF = "+    logger.debug('Insufficient data')\n"


def _pr(state="MERGED", number=7, created="2026-10-01T02:00:00Z", branch="loop/QuantAgent-bv8", base="main"):
    return {"number": number, "state": state, "createdAt": created, "headRefName": branch, "baseRefName": base,
            "url": f"https://github.com/x/y/pull/{number}", "title": "QuantAgent-bv8: stdout limpio"}


def _comment(body, at="2026-10-01T12:00:00Z", login=REVIEWER):
    return {"author": {"login": login}, "body": body, "createdAt": at}


def _detail(comments=(), commit_at="2026-10-01T02:30:00Z"):
    return lambda n: {"commits": [{"committedDate": commit_at}], "comments": list(comments),
                      "reviews": [], "body": BODY, "diff": DIFF}


def test_no_loop_prs_lets_the_loop_start():
    code, out = gate.evaluate([], REVIEWER, _detail())
    assert (code, out["mode"]) == (gate.EXIT_PASS, "nuevo")


def test_delivery_without_review_waits():
    code, out = gate.evaluate([_pr("OPEN")], REVIEWER, _detail())
    assert code == gate.EXIT_WAIT
    assert "esperando revisión" in out["message"]


def test_valid_merge_on_merged_pr_passes_and_closes_previous():
    r = _comment("R: bv8 vi: en la VM salieron 5 lineas, sharpe 0.40 decido: merge")
    code, out = gate.evaluate([_pr("MERGED")], REVIEWER, _detail([r]))
    assert (code, out["mode"], out["previous"]) == (gate.EXIT_PASS, "nuevo", "cerrar")


def test_review_written_before_the_last_commit_does_not_count():
    r = _comment("R: bv8 vi: en la VM salieron 5 lineas decido: merge", at="2026-10-01T02:10:00Z")
    code, _ = gate.evaluate([_pr("MERGED")], REVIEWER, _detail([r]))
    assert code == gate.EXIT_WAIT


def test_review_from_someone_else_does_not_count():
    r = _comment("R: bv8 vi: en la VM salieron 5 lineas decido: merge", login="someone-else")
    code, _ = gate.evaluate([_pr("MERGED")], REVIEWER, _detail([r]))
    assert code == gate.EXIT_WAIT


@pytest.mark.parametrize("line, reason", [
    ("R: bv8 vi: decido: merge", "menos de 3 palabras"),
    ("R: bv8 vi: todo bien ok decido: merge", "no menciona nada"),
    ("R: bv8 vi: miré el código nuevo decido: merge", "no menciona nada"),
    ("R: e35 vi: salieron 5 lineas decido: merge", "nombra 'e35'"),
    ("R: bv8 salieron 5 lineas, merge", "formato"),
    ("R: bv8 vi: salieron 5 lineas decido: descarto", "necesita decir"),
])
def test_invalid_review_lines_keep_the_loop_waiting(line, reason):
    code, out = gate.evaluate([_pr("CLOSED")], REVIEWER, _detail([_comment(line)]))
    assert code == gate.EXIT_WAIT
    assert reason in out["message"]


def test_plain_comment_without_r_prefix_is_not_a_review():
    code, out = gate.evaluate([_pr("MERGED")], REVIEWER, _detail([_comment("revisado")]))
    assert code == gate.EXIT_WAIT and "esperando revisión" in out["message"]


def test_real_line_from_pr1_is_rejected_for_empty_vi():
    review, reason = gate.parse_review("R: cierre vi: decido: merge", "cierre", "Trades: 227")
    assert review is None and "menos de 3 palabras" in reason


def test_real_line_from_pr2_is_valid_against_its_own_delivery():
    review, reason = gate.parse_review(
        "R: hyt vi: líneas comentadas ok decido: merge", "QuantAgent-hyt",
        "cada línea comentada cuenta dos veces. No hay líneas nuevas")
    assert reason is None and review["decision"] == "merge"


def test_real_multiline_line_from_pr7_with_pasted_output_is_valid():
    body = ("R: fdi vi: testeado en la VM, resultado ok: \u2500\u2500 loop/QuantAgent-fdi @ 5965aa1c \u2500\u2500\r\n"
            "Trades: 7\r\nWin rate: 71.43%\r\nProfit factor: 1.44\r\nSharpe ratio: 2.10\r\nTotal PnL: 94.02 decido: merge")
    review, reason = gate.parse_review(body, "QuantAgent-fdi", "Obtenido por el loop: Trades: 7 Win rate: 71.43%")
    assert reason is None
    assert review["decision"] == "merge" and "71.43" in review["vi"]


def test_multiline_comment_without_decision_is_still_rejected():
    review, reason = gate.parse_review("R: fdi vi: salieron 7 trades\nTotal PnL: 94.02", "fdi", "Trades: 7")
    assert review is None and "formato" in reason


def test_merge_decision_on_open_pr_waits_for_the_merge():
    r = _comment("R: bv8 vi: salieron 5 lineas decido: merge")
    code, out = gate.evaluate([_pr("OPEN")], REVIEWER, _detail([r]))
    assert code == gate.EXIT_WAIT and "falta mergear" in out["message"]


def test_change_request_on_open_pr_resumes_same_branch():
    r = _comment("R: bv8 vi: salieron 5 lineas decido: cambio dejar tambien el warning de sqlalchemy")
    code, out = gate.evaluate([_pr("OPEN")], REVIEWER, _detail([r]))
    assert (code, out["mode"], out["branch"]) == (gate.EXIT_PASS, "cambio", "loop/QuantAgent-bv8")
    assert out["review"]["detail"].startswith("dejar")


def test_discard_on_closed_pr_moves_on():
    r = _comment("R: bv8 vi: salieron 5 lineas decido: descarto prefiero un flag --quiet")
    code, out = gate.evaluate([_pr("CLOSED")], REVIEWER, _detail([r]))
    assert (code, out["previous"]) == (gate.EXIT_PASS, "descartar")


def test_only_the_most_recent_loop_pr_is_gated():
    old = _pr("OPEN", number=3, created="2026-09-01T02:00:00Z")
    new = _pr("MERGED", number=7)
    r = _comment("R: bv8 vi: salieron 5 lineas decido: merge")
    code, out = gate.evaluate([old, new], REVIEWER, _detail([r]))
    assert (code, out["pr"]) == (gate.EXIT_PASS, 7)


LOT = "lote/metricas-auditadas"


def test_pm_line_passes_on_a_pr_into_a_lot_branch():
    r = _comment("PM: bv8 vi: corrí el comando y salieron 5 lineas decido: merge")
    code, out = gate.evaluate([_pr("MERGED", base=LOT)], REVIEWER, _detail([r]))
    assert (code, out["previous"], out["review"]["by"]) == (gate.EXIT_PASS, "cerrar", "PM")


def test_pm_line_never_counts_on_a_pr_into_main():
    r = _comment("PM: bv8 vi: corrí el comando y salieron 5 lineas decido: merge")
    code, out = gate.evaluate([_pr("MERGED", base="main")], REVIEWER, _detail([r]))
    assert code == gate.EXIT_WAIT and "hace falta la R: de Fede" in out["message"]


def test_pm_line_is_held_to_the_same_evidence_rule():
    r = _comment("PM: bv8 vi: todo bien ok decido: merge")
    code, out = gate.evaluate([_pr("MERGED", base=LOT)], REVIEWER, _detail([r]))
    assert code == gate.EXIT_WAIT and "no menciona nada" in out["message"]


def test_pm_escalation_leaves_the_pr_open_and_lets_the_lot_continue():
    r = _comment("PM: bv8 vi: salieron 5 lineas decido: escalo Fede elige entre 252 y 365 periodos")
    code, out = gate.evaluate([_pr("OPEN", base=LOT)], REVIEWER, _detail([r]))
    assert (code, out["mode"], out["previous"]) == (gate.EXIT_PASS, "nuevo", "escalar")


@pytest.mark.parametrize("line, reason", [
    ("R: bv8 vi: salieron 5 lineas decido: escalo que decida otro", "solo para la línea PM:"),
    ("PM: bv8 vi: salieron 5 lineas decido: escalo", "necesita decir"),
])
def test_invalid_escalations_wait(line, reason):
    code, out = gate.evaluate([_pr("OPEN", base=LOT)], REVIEWER, _detail([_comment(line)]))
    assert code == gate.EXIT_WAIT and reason in out["message"]


def test_fede_r_line_still_decides_on_a_pr_into_a_lot_branch():
    r = _comment("R: bv8 vi: salieron 5 lineas decido: cambio dejar el warning")
    code, out = gate.evaluate([_pr("OPEN", base=LOT)], REVIEWER, _detail([r]))
    assert (code, out["mode"]) == (gate.EXIT_PASS, "cambio")


def test_without_an_open_lot_the_loop_waits():
    lot, wait = gate.active_lot([])
    assert lot is None and "sin lote abierto" in wait


def test_a_lot_marked_ready_waits_for_fede_merge():
    lot, wait = gate.active_lot([{"number": 30, "headRefName": LOT, "isDraft": False, "url": "u"}])
    assert lot is None and "falta que Fede lo mergee" in wait


def test_two_open_lots_wait():
    prs = [{"number": 30, "headRefName": LOT, "isDraft": True}, {"number": 31, "headRefName": "lote/b", "isDraft": True}]
    lot, wait = gate.active_lot(prs)
    assert lot is None and "más de un lote" in wait


def test_a_draft_lot_is_the_base_of_the_next_delivery():
    lot, wait = gate.active_lot([{"number": 30, "headRefName": LOT, "isDraft": True}])
    assert wait is None and lot["headRefName"] == LOT


def test_active_lot_with_target_lot():
    prs = [{"number": 30, "headRefName": LOT, "isDraft": True}, {"number": 31, "headRefName": "lote/b", "isDraft": True}]
    assert gate.active_lot(prs)[0] is None
    assert gate.active_lot(prs, target_lot="lote/b")[0]["headRefName"] == "lote/b"


@pytest.mark.parametrize("bd, ticket, msg", [
    ({"status": "closed", "labels": [], "dependencies": []}, "QuantAgent-d4cq.76", "no está open"),
    ({"status": "open", "labels": [], "dependencies": [{"status": "open", "dependency_type": "blocks"}]}, "QuantAgent-d4cq.76", "blockers abiertos"),
    ({"status": "open", "labels": ["loop-descartado"], "dependencies": []}, "QuantAgent-d4cq.76", "label de exclusión"),
    ({"status": "open", "labels": [], "dependencies": []}, "QuantAgent-e35", "pertenece al lote L1"),
])
def test_ticket_not_eligible(bd, ticket, msg):
    q = "| L12 | `lote/cartera` | ...\n| L1 | `lote/m` | ...\n| C11 | `QuantAgent-d4cq.76` | o | a | l | L12 |\n| T01 | `QuantAgent-e35` | o | a | l | L1 |\n"
    assert msg in gate.check_ticket_eligible(ticket, "lote/cartera", q, bd)[1]
    assert gate.check_ticket_eligible("QuantAgent-d4cq.76", "lote/cartera", q, {"status": "open", "labels": [], "dependencies": []})[0] is True


def test_gate_with_ticket_reads_queue_from_lot_branch(tmp_path, monkeypatch, capsys):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "--allow-empty", "--no-verify", "-m", "init"], cwd=tmp_path, check=True)
    (tmp_path / "PLAN-CONTINUACION.md").write_text("| L12 | `lote/cartera` |\n| C11 | `QuantAgent-d4cq.76` | o | a | l | L12 |\n")
    subprocess.run(["git", "add", "PLAN-CONTINUACION.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "--no-verify", "-m", "queue in lot"], cwd=tmp_path, check=True)
    subprocess.run(["git", "update-ref", "refs/remotes/origin/lote/cartera", "HEAD"], cwd=tmp_path, check=True)
    (tmp_path / "PLAN-CONTINUACION.md").unlink()
    subprocess.run(["git", "commit", "-a", "--no-verify", "-m", "remove queue from main"], cwd=tmp_path, check=True)

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GIT_DIR", raising=False)
    monkeypatch.setattr(gate, "_gh", lambda *a: json.dumps(
        [{"number": 42, "url": "https://github.com/x/y/pull/42", "headRefName": "lote/cartera", "isDraft": True}] if "--label" in a and a[a.index("--label") + 1] == "lote" else []
    ))
    orig_run = gate.subprocess.run
    monkeypatch.setattr(gate.subprocess, "run", lambda cmd, *a, **kw: subprocess.CompletedProcess(cmd, 0, stdout=json.dumps([{"status": "open", "labels": [], "dependencies": []}])) if cmd[:2] == ["bd", "show"] else orig_run(cmd, *a, **kw))
    monkeypatch.setattr(gate.sys, "argv", ["gate.py", "--ticket", "QuantAgent-d4cq.76"])

    rc = gate.main()
    out = json.loads(capsys.readouterr().out)
    assert rc == gate.EXIT_PASS
    assert out["result"] == "PASS" and out["ticket_asignado"] == "QuantAgent-d4cq.76"


def test_load_lot_queue_failure_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GIT_DIR", raising=False)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    with pytest.raises(RuntimeError, match="no se pudo leer PLAN-CONTINUACION.md de origin/lote/inexistente"):
        gate.load_lot_queue("lote/inexistente")


def test_filter_loop_prs():
    prs = [_pr(number=1, base="lote/a"), _pr(number=2, base="lote/b")]
    assert gate.filter_loop_prs(prs, None) == prs
    assert gate.filter_loop_prs(prs, "") == prs
    assert [p["number"] for p in gate.filter_loop_prs(prs, "lote/a")] == [1]
    assert [p["number"] for p in gate.filter_loop_prs(prs, "lote/b")] == [2]
    assert gate.filter_loop_prs(prs, "lote/c") == []


def test_gate_with_loop_lote_evaluates_only_matching_base(monkeypatch, capsys):
    pr_a = _pr("MERGED", number=10, created="2026-10-01T01:00:00Z", branch="loop/QuantAgent-a", base="lote/a")
    pr_b = _pr("OPEN", number=11, created="2026-10-01T02:00:00Z", branch="loop/QuantAgent-b", base="lote/b")
    r_a = _comment("PM: a vi: salida ok 5 lineas decido: merge")

    def fake_gh(*args):
        if "--label" in args and args[args.index("--label") + 1] in ("lote", "carril-b"):
            return json.dumps([{"number": 100, "url": "u", "headRefName": "lote/a", "isDraft": True}])
        if "--label" in args and args[args.index("--label") + 1] == "loop":
            return json.dumps([pr_a, pr_b])
        if "view" in args:
            return json.dumps({"commits": [{"committedDate": "2026-10-01T01:30:00Z"}], "comments": [r_a], "reviews": [], "body": BODY})
        if "diff" in args:
            return DIFF
        return ""

    monkeypatch.setattr(gate, "_gh", fake_gh)
    monkeypatch.setattr(gate.sys, "argv", ["gate.py"])
    monkeypatch.setenv("LOOP_LOTE", "lote/a")

    rc = gate.main()
    out = json.loads(capsys.readouterr().out)
    assert rc == gate.EXIT_PASS
    assert out["result"] == "PASS" and out["pr"] == 10 and out["mode"] == "nuevo"


def test_gate_with_loop_lote_ignores_cambio_on_other_lot(monkeypatch, capsys):
    pr_a = _pr("MERGED", number=20, created="2026-10-01T01:00:00Z", branch="loop/QuantAgent-a", base="lote/a")
    pr_b = _pr("OPEN", number=21, created="2026-10-01T02:00:00Z", branch="loop/QuantAgent-b", base="lote/b")
    r_a = _comment("PM: a vi: salida ok 5 lineas decido: merge")
    r_b = _comment("PM: b vi: salida ok 5 lineas decido: cambio ajustar x")

    def fake_gh(*args):
        if "--label" in args and args[args.index("--label") + 1] in ("lote", "carril-b"):
            return json.dumps([{"number": 100, "url": "u", "headRefName": "lote/a", "isDraft": True}])
        if "--label" in args and args[args.index("--label") + 1] == "loop":
            return json.dumps([pr_a, pr_b])
        if "view" in args:
            num = int(args[args.index("view") + 1])
            comments = [r_a] if num == 20 else [r_b]
            return json.dumps({"commits": [{"committedDate": "2026-10-01T01:30:00Z"}], "comments": comments, "reviews": [], "body": BODY})
        if "diff" in args:
            return DIFF
        return ""

    monkeypatch.setattr(gate, "_gh", fake_gh)
    monkeypatch.setattr(gate.sys, "argv", ["gate.py"])
    monkeypatch.setenv("LOOP_LOTE", "lote/a")

    rc = gate.main()
    out = json.loads(capsys.readouterr().out)
    assert rc == gate.EXIT_PASS
    assert out["result"] == "PASS" and out["pr"] == 20 and out["mode"] == "nuevo"


