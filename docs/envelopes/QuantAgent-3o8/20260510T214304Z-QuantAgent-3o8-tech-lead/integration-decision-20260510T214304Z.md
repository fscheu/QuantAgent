# Integration Decision — QuantAgent-3o8

- **Run ID:** 20260510T214304Z-QuantAgent-3o8-tech-lead
- **Ticket:** QuantAgent-3o8
- **Decision:** DO_NOT_MERGE
- **Status:** FAIL
- **Primary failure_class:** IMPLEMENTATION_INCOMPLETE
- **Primary failure_subclass:** replay_signal_provenance_collision
- **Secondary failure_class:** GIT_PREFLIGHT_DIRTY_OR_DIVERGED
- **Secondary failure_subclass:** stale_feature_branch
- **Executor lineage:** auto (implementer) -> tech-lead direct verification
- **Feature branch:** `feature/QuantAgent-3o8-implement-replay-execution-mode-reuse-an`
- **Feature commit under review:** `7886ad2b29f82603d5727ca624f9639f15c369f7`
- **Merge strategy:** none
- **Conflict status:** preflight overlap detected (`git merge-tree` reported `changed in both`)
- **Deploy status:** not_applicable
- **User manual:** skipped (no merge; no `docs/user-manual/` tree present)
- **Blocker issue:** `QuantAgent-375`

## Why merge was rejected
1. The branch is 207 commits behind `origin/main`, so integration risk is already high.
2. More importantly, replay correctness is broken:
   - `Backtest.run_replay()` queries signals by asset/timeframe/date/environment only.
   - With overlapping source runs, `signal_map[(symbol, generated_at)]` is overwritten by whichever signal is loaded last.
   - Direct proof run showed `source_run_id=1` consuming signal `2` from another run, violating acceptance TC5 / TC11.

## Evidence reviewed
- Issue docs on candidate branch:
  - `docs/01_requirements/QuantAgent-3o8-RQ-replay-execution.md`
  - `docs/03_design/QuantAgent-3o8-DS-replay-execution.md`
  - `docs/05_acceptance_tests/QuantAgent-3o8-AC-replay-execution.md`
- Implementer comment + commit `7886ad2b`
- `python3 -m compileall -q quantagent apps alembic` -> PASS
- Direct replay provenance proof -> FAIL (cross-run signal contamination)
- `git merge-tree` preflight -> overlap detected

## Follow-up required before re-integration
- Land blocker `QuantAgent-375` from a fresh branch based on current `origin/main`
- Add deterministic source-run signal scoping
- Add regression tests for overlapping runs
- Re-run tester + integration after fix
