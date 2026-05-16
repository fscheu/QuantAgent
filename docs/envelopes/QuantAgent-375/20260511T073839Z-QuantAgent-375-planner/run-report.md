---
run_id: "20260511T073839Z-QuantAgent-375-planner"
phase: "planner"
executor: "claude-code"
status: "SUCCESS"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-375"
workdir: "/home/azureuser/repos/projects/QuantAgent"
finished_at: "2026-05-11T08:00:00.000000+00:00"
exit_code: 0
---

# Run Report — QuantAgent-375 — planner

**Run-ID:** 20260511T073839Z-QuantAgent-375-planner  
**Phase:** planner  
**Issue:** QuantAgent-375 — Scope replay signal lookup to selected source run

---

## Summary

Planning phase completed for QuantAgent-375. The root cause of the replay signal contamination
bug was confirmed: `Backtest.run_replay()` queries signals by `(symbol, timeframe, date, environment)`
only, with no link to the source `BacktestRun`. The fix requires adding a `backtest_run_id` FK
to `Signal` (one new nullable column + migration) and updating both signal creation and replay
query paths.

Four planning documents were produced and four README files were updated.

---

## Root Cause (confirmed)

`Signal` has no `backtest_run_id`. When two backtest runs share symbol/timeframe/dates,
the `signal_map` built in `run_replay()` is overwritten by whichever signal is loaded last.
This was proven in the QuantAgent-3o8 tech-lead integration review on 2026-05-10.

---

## Docs Produced

| File | Type |
|------|------|
| `docs/01_requirements/QuantAgent-375-RQ-scope-replay-signal-lookup.md` | Requirements |
| `docs/02_planning/QuantAgent-375-PL-scope-replay-signal-lookup.md` | Planning |
| `docs/03_design/QuantAgent-375-DS-scope-replay-signal-lookup.md` | Design |
| `docs/05_acceptance_tests/QuantAgent-375-AC-scope-replay-signal-lookup.md` | Acceptance Tests |

README.md updated in: `01_requirements/`, `02_planning/`, `03_design/`, `05_acceptance_tests/`

---

## Quality Gates

| Gate | Result |
|------|--------|
| `git status --short` | PASS — 4 new doc files, 4 README updates, no unexpected dirty state |
| issue ID in docs paths | PASS — QuantAgent-375 present in all 4 doc filenames |
| acceptance criteria testable | PASS — 7 Given/When/Then test cases with explicit testability notes |
| `python -m compileall -q` | PASS — no code changes in planner phase |

---

## Risks

1. **3o8 branch rebase:** The feature branch for QuantAgent-3o8 is 207 commits behind `main`.
   After QuantAgent-375 lands on main, the 3o8 changes must be rebased; the `run_replay()`
   query update (Step 4) is part of the 3o8 branch, not 375.

2. **Pre-migration signals:** Existing `Signal` rows will have `backtest_run_id = NULL`.
   A `run_replay()` call against an old run will return zero signals. Implementer must add
   an explicit `ValueError` guard for this case.

---

## Next Step

**Implementer phase.** Hand off:
- `docs/02_planning/QuantAgent-375-PL-scope-replay-signal-lookup.md` (step-by-step)
- `docs/03_design/QuantAgent-375-DS-scope-replay-signal-lookup.md` (schema + code deltas)
- `docs/05_acceptance_tests/QuantAgent-375-AC-scope-replay-signal-lookup.md` (test cases)

Branch base: `main` (fresh branch `feature/QuantAgent-375-scope-replay-signal-lookup`)
