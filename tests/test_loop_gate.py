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
