# Run Report — QuantAgent-3o8 Tech Lead Review

- **Run ID:** 20260510T214304Z-QuantAgent-3o8-tech-lead
- **Timestamp (ART):** 2026-05-10 18:43:02 -03
- **Issue:** QuantAgent-3o8
- **Mode:** tech lead integration review / patrol-assisted verification
- **Status:** FAIL
- **Primary failure:** IMPLEMENTATION_INCOMPLETE / replay_signal_provenance_collision
- **Secondary failure:** GIT_PREFLIGHT_DIRTY_OR_DIVERGED / stale_feature_branch

## Summary
- Repo root had pre-existing untracked envelope dirs but no active merge/rebase/cherry-pick state; work continued in isolated worktrees.
- Ready queue contained 4 tickets; only `QuantAgent-3o8` was design-approved and integration-relevant. P3 `design_pending` tickets (`e4k`, `69d`, `um8`) were skipped this run.
- Feature branch `feature/QuantAgent-3o8-implement-replay-execution-mode-reuse-an` remains **207 commits behind** `origin/main`.
- Direct replay verification proved a correctness bug: replay for source run 1 consumed signal ID 2 from another overlapping run because `run_replay()` builds `signal_map[(symbol, generated_at)]` without source-run scoping.
- Created blocker issue `QuantAgent-375` and moved `QuantAgent-3o8` to `blocked`.

## Patrol / Preflight
### Repo root
- Branch: `main`
- Status: dirty only via pre-existing untracked envelope dirs:
  - `docs/envelopes/QuantAgent-3o8/`
  - `docs/envelopes/QuantAgent-4fm/`
  - `docs/envelopes/QuantAgent-l8r/20260508T124108Z-QuantAgent-l8r-planner/`
- No `MERGE_HEAD`, `CHERRY_PICK_HEAD`, `rebase-merge`, or `rebase-apply` state detected.

### Ready queue inspected
- `QuantAgent-3o8` — P2, design approved, selected
- `QuantAgent-e4k` — P3, design_pending, skipped
- `QuantAgent-69d` — P3, design_pending, skipped
- `QuantAgent-um8` — P3, design_pending, skipped

## Source of Truth Reviewed
- `AGENTS.md`, `CLAUDE.md`
- `docs/01_requirements/QuantAgent-3o8-RQ-replay-execution.md`
- `docs/03_design/QuantAgent-3o8-DS-replay-execution.md`
- `docs/05_acceptance_tests/QuantAgent-3o8-AC-replay-execution.md`
- Beads issue/comments for `QuantAgent-3o8`

## Verification Findings
### 1. Stale branch preflight
- Merge-base: `69e24bdd4afcd709bbd902f64f7d3ce77a0897b0`
- Commits behind `origin/main`: `207`
- `git merge-tree` preview reported overlapping changes (`changed in both`) before any integration attempt.

### 2. Correctness failure: replay signal provenance collision
Acceptance TC5/TC11 require replay to use the same source-run signals and keep provenance traceable.

Observed behavior from direct proof run:
- Source run ID: `1`
- Source signal ID: `1`
- Overlapping other-run signal ID: `2`
- `run_replay(source_run_id=1)` used signal ID `2` (`other-run-b`) for timestamp `2024-01-01T00:00:00`

Proof output:
```python
{'source_run_id': 1, 'source_signal_id': 1, 'other_signal_id': 2, 'used': [('BTC', '2024-01-01T00:00:00', 2, 'other-run-b'), ('BTC', '2024-01-01T01:00:00', None, None)]}
```

Root cause:
- `Backtest.run_replay()` loads signals only with:
  - symbol in source assets
  - timeframe == source timeframe
  - generated_at within source date range
  - environment == backtest
- It then collapses them into `signal_map[(symbol, generated_at)]`, so overlapping runs overwrite each other.
- There is no robust signal-to-source-run linkage in the query path.

## Decision
- **Do not merge.**
- `QuantAgent-3o8` is not integration-ready because it fails a core replay/provenance acceptance requirement and the branch is also severely stale.
- New blocker issue created: `QuantAgent-375` — scope replay signal lookup to the selected source run.

## Beads / Routing Actions
- Created `QuantAgent-375` (P1 bug, discovered from `QuantAgent-3o8`)
- Updated `QuantAgent-3o8` status to `blocked`
- Final Beads comment added to `QuantAgent-3o8`

## Merge / Deploy / User Manual
- Merge: not attempted
- Push to `main`: none
- Deploy: not applicable
- User manual: skipped (no merge, no `docs/user-manual/` present)

## Recommended Next Step
1. Implement blocker `QuantAgent-375` on a fresh branch/worktree from current `origin/main`.
2. Add deterministic source-run scoping for replay signal loading.
3. Add tests covering overlapping runs with identical symbol/timeframe/timestamp.
4. Re-run tester/integration only after the replay provenance bug is fixed and the candidate branch is rebased/fresh.
