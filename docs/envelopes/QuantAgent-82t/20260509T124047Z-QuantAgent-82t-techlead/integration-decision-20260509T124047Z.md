# Integration decision — QuantAgent-82t

- **Run ID:** 20260509T124047Z-QuantAgent-82t-techlead
- **Issue:** QuantAgent-82t
- **Decision:** MERGE
- **Status:** SUCCESS
- **failure_class:** NO_FAILURE
- **failure_subclass:** none
- **tester_run_id:** detached-main-revalidation-20260509T123805Z
- **merge_strategy:** linear integration branch push to `origin/main`
- **conflict_status:** clean
- **candidate_change_commit:** `47dfcd46`
- **final_main_sha:** `FINAL_MAIN_SHA_PENDING`

## Why merge is now honest
This ticket re-enables a quality gate, so the diff alone is not enough. The exact gate command was rerun on a detached `origin/main` worktree after the last concrete blockers closed.

Command used:
`DATABASE_URL=postgresql://test:test@localhost:5432/quantagent_test /mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python -m pytest tests/ -v --tb=short --maxfail=10 -m "not integration and not slow"`

Result:
- exit code `0`
- `715 passed, 20 skipped, 34 deselected`
- duration `73.19s`

That means the newly re-enabled test step is now expected to pass on `main`, so the workflow diff is merge-ready.

## Scope review
- Included: `.github/workflows/main-ci-deploy.yml` test-step re-enable.
- Excluded: any test or production-code changes; no extra gate tweaks; no user-facing docs.
- Historical `feature/QuantAgent-82t-*` branches were not used because they had accumulated unrelated drift against current `main`.

## Preflight / safety
- Repo root was dirty due to unrelated untracked `docs/envelopes/QuantAgent-l8r/...`; integration used isolated worktrees only.
- `gh run list` showed no active `main` workflow at decision time.
- Runner headroom before push: `/` had ~13G free; Docker reclaimable images ~16.56G.

## Post-merge manual
- `user_manual_skipped`: internal CI/workflow change and `docs/user-manual/` does not exist.
