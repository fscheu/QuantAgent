# Integration decision — QuantAgent-82t

- Timestamp: `20260508T214848Z`
- Ticket: `QuantAgent-82t`
- Decision: `NO_MERGE`
- Merge strategy: `none`
- Conflict status: `not_attempted`
- Failure taxonomy: `QUALITY_GATE_FAILED / pre_existing`

## Evidence reviewed
- Candidate integration branch: `integration/QuantAgent-82t-20260508T213950Z`
- Candidate diff vs main: `.github/workflows/main-ci-deploy.yml` plus issue docs only
- Current `origin/main` still has the pytest step commented in `.github/workflows/main-ci-deploy.yml` lines 67-75
- Exact gate command revalidated against a detached `origin/main` worktree with PostgreSQL local:

```bash
DATABASE_URL=postgresql://test:***@localhost:5432/quantagent_test \
/mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python -m pytest \
  tests/test_vje_scheduler_heartbeat_backend.py \
  tests/test_wait_sec_deprecation_removal.py \
  tests/trading/test_scheduler.py \
  -v --tb=short --maxfail=10 -m "not integration and not slow"
```

## Result summary
- Result on `origin/main`: `10 failed, 30 passed`
- Failures are pre-existing on current main, not introduced by the `QuantAgent-82t` workflow diff.
- Because `QuantAgent-82t` is a gate-enablement ticket, merging it now would knowingly turn `main` red.

## Concrete blockers extracted
1. `QuantAgent-uzq` — fix TradingScheduler heartbeat + scheduler unit-test regressions
   - 5 heartbeat backend failures
   - 3 scheduler unit-test failures caused by `DummySession.query()` returning a truthy `Mock`, which is treated as an active position and crashes in `update_candle_tracking()`
2. `QuantAgent-9t5` — fix stale hardcoded worktree path in wait_sec deprecation validation tests
   - 2 failures due to `FileNotFoundError` for historical path `/home/azureuser/repos/projects/QuantAgent/.worktrees/feature__QuantAgent-lmn-fix-deprecated-wait-sec-parameter-in-age`

## Integration ruling
`QuantAgent-82t` is **not merge-ready**. The workflow diff looks correct, but the re-enabled pytest gate still has reproducible failures on `main`. Per gate-enablement policy, this stays blocked until those blockers land.

## Beads actions taken
- Created blocker `QuantAgent-uzq`
- Created blocker `QuantAgent-9t5`
- Added real dependencies `QuantAgent-82t -> QuantAgent-uzq` and `QuantAgent-82t -> QuantAgent-9t5`
- Moved `QuantAgent-82t` to `blocked`

## Next route
- Route `QuantAgent-uzq` to planner/implementer/tester as the highest-value unblocker
- Route `QuantAgent-9t5` as a second narrow blocker
- Re-run the exact CI gate after both blockers merge, then re-evaluate integration of `QuantAgent-82t`

## User-manual impact
- `user_manual_skipped`: internal CI/workflow change only; no user-facing manual update required
