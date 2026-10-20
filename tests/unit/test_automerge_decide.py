"""Each refusal guards a separate condition of the lot auto-merge policy."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "loop_automerge", Path(__file__).resolve().parents[2] / "scripts/loop/automerge.py"
)
automerge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(automerge)


def _comment(body="PM: QuantAgent-4148 vi: 25 tests verdes decido: merge", **changes):
    return {"id": 1, "user": {"login": "fscheu"}, "body": body,
            "created_at": "2026-10-19T12:00:00Z", "html_url": "https://github.com/x/y/issues/1#issuecomment-1",
            **changes}


@pytest.fixture
def data():
    return ({"base": {"ref": "lote/test", "repo": {"full_name": "o/r"}},
             "head": {"ref": "loop/QuantAgent-4148", "repo": {"full_name": "o/r"}},
             "state": "open", "draft": False, "mergeable": True,
             "last_commit_at": "2026-10-19T11:00:00Z"},
            [_comment()], [{"name": "CI (Lint + Tests)", "status": "completed", "conclusion": "success"}])


def _decide(data):
    return automerge.decide(*data, "fscheu", "lote/")


def test_happy_path(data):
    allowed, reason = _decide(data)
    assert allowed
    assert data[1][0]["html_url"] in reason


def test_main_is_never_allowed(data):
    data[0]["base"]["ref"] = "main"
    assert not _decide(data)[0]


def test_other_base_is_never_allowed_even_with_changed_config(data):
    data[0]["base"]["ref"] = "feature/test"
    assert not automerge.decide(*data, "fscheu", "feature/")[0]


def test_configured_lot_prefix_is_required(data):
    assert not automerge.decide(*data, "fscheu", "lote/other")[0]


@pytest.mark.parametrize("head", ["feature/QuantAgent-4148", "loop/4148", "loop/QuantAgent-4148/extra"])
def test_head_must_identify_a_loop_ticket(data, head):
    data[0]["head"]["ref"] = head
    assert not _decide(data)[0]


def test_closed_pr(data):
    data[0]["state"] = "closed"
    assert not _decide(data)[0]


def test_draft(data):
    data[0]["draft"] = True
    assert not _decide(data)[0]


def test_other_author(data):
    data[1][0]["user"]["login"] = "someone-else"
    assert not _decide(data)[0]


@pytest.mark.parametrize("ticket", ["QuantAgent-other", "QuantAgent-41480", "4148"])
def test_other_ticket(data, ticket):
    data[1][0]["body"] = f"PM: {ticket} vi: tests verdes decido: merge"
    assert not _decide(data)[0]


@pytest.mark.parametrize("prefix", ["R:", "texto PM:", " PM:"])
def test_requires_pm_at_start(data, prefix):
    data[1][0]["body"] = f"{prefix} QuantAgent-4148 vi: tests verdes decido: merge"
    assert not _decide(data)[0]


@pytest.mark.parametrize("at", ["2026-10-19T10:00:00Z", "2026-10-19T11:00:00Z"])
def test_pm_must_be_strictly_after_commit(data, at):
    data[1][0]["created_at"] = at
    assert not _decide(data)[0]


@pytest.mark.parametrize("decision", ["cambio corregir", "descarto motivo", "escalo pregunta", "mergex", ""])
def test_decision_must_be_merge(data, decision):
    data[1][0]["body"] = f"PM: QuantAgent-4148 vi: tests verdes decido: {decision}"
    assert not _decide(data)[0]


def test_last_decision_merge_wins(data):
    data[1][0]["body"] += " decido: cambio corregir decido: merge"
    assert _decide(data)[0]


def test_last_decision_change_wins(data):
    data[1][0]["body"] += " decido: cambio corregir"
    assert not _decide(data)[0]


def test_latest_pm_comment_wins_even_when_unsorted(data):
    data[1].insert(0, _comment("PM: QuantAgent-4148 decido: cambio corregir", id=2,
                             created_at="2026-10-19T13:00:00Z"))
    assert not _decide(data)[0]


def test_newer_unrelated_comment_does_not_override_pm(data):
    data[1].append(_comment("R: QuantAgent-4148 decido: cambio", id=2,
                            created_at="2026-10-19T13:00:00Z"))
    assert _decide(data)[0]


def test_red_ci(data):
    data[2][0]["conclusion"] = "failure"
    assert not _decide(data)[0]


def test_pending_ci(data):
    data[2][0].update(status="in_progress", conclusion=None)
    assert not _decide(data)[0]


def test_missing_required_ci(data):
    data[2][0]["name"] = "Other CI"
    assert not _decide(data)[0]


@pytest.mark.parametrize("status,conclusion", [
    ("completed", "failure"), ("completed", "cancelled"), ("completed", "timed_out"),
    ("queued", None), ("in_progress", None), ("completed", None),
])
def test_other_checks_must_also_pass(data, status, conclusion):
    data[2].append({"name": "Other check", "status": status, "conclusion": conclusion})
    assert not _decide(data)[0]


@pytest.mark.parametrize("conclusion", ["skipped", "neutral"])
def test_other_checks_may_be_skipped_or_neutral(data, conclusion):
    data[2].append({"name": "Other check", "status": "completed", "conclusion": conclusion})
    assert _decide(data)[0]


@pytest.mark.parametrize("mergeable", [False, None, "CONFLICTING"])
def test_not_mergeable(data, mergeable):
    data[0]["mergeable"] = mergeable
    assert not _decide(data)[0]


def test_conflicting_mergeable_state(data):
    data[0]["mergeable_state"] = "CONFLICTING"
    assert not _decide(data)[0]


def test_incomplete_data_is_refused(data):
    del data[0]["last_commit_at"]
    assert not _decide(data)[0]


def test_fork_head_is_refused(data):
    data[0]["head"]["repo"]["full_name"] = "evil/r"
    assert not _decide(data)[0]
