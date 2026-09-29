"""Tests for the nightly loop review gate (scripts/loop/gate.py, PLAN-CONTINUACION.md §3.1 and §5).

Each test builds the GitHub data the gate would read and checks PASS/WAIT. A regression here means
the loop could start work on an unreviewed delivery, or stay blocked after a valid review.
"""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "loop_gate", Path(__file__).resolve().parents[1] / "scripts" / "loop" / "gate.py"
)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

REVIEWER = "fscheu"
BODY = "Cambió: la salida del CLI queda en 5 líneas\nObtenido: Trades: 227 Sharpe 0.40"
DIFF = "+    logger.debug('Insufficient data')\n"


def _pr(state="MERGED", number=7, created="2026-10-01T02:00:00Z", branch="loop/QuantAgent-bv8"):
    return {"number": number, "state": state, "createdAt": created, "headRefName": branch,
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
