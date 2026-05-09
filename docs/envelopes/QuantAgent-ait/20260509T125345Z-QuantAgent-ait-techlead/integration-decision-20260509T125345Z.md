# Integration decision — QuantAgent-ait

- **Decision:** MERGE
- **Timestamp:** 2026-05-09 09:53:45 -0300
- **Mode:** correction
- **Tester evidence:** direct Tech Lead verification in isolated worktree
- **Conflict status:** clean
- **Merge strategy:** `--no-ff` into `main`
- **User manual:** skipped (`internal test-only fix`)

## Why
Main CI for commit `80f6a50d` failed because `tests/test_static_util.py` used deprecated uppercase hourly pandas aliases. The fix is limited to tests, reproduced against pandas 3.x behavior, and the relevant full gate now passes locally.
