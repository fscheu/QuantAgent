# Tester Run Report — QuantAgent-82t

**Run ID:** 20260504T175326Z-QuantAgent-82t-tester  
**Phase:** tester  
**Branch:** feature/QuantAgent-82t-re-enable-unit-tests-in-ci-pipeline-main  
**Date:** 2026-05-04  

---

## Summary

The implementer's workflow change (uncomment "Run unit tests" step in `main-ci-deploy.yml`) is **structurally correct** — all static ACs pass. However, a **pre-existing JSONB/SQLite incompatibility** in multiple test fixtures will cause CI failures even with PostgreSQL available, because those tests hardcode `sqlite:///:memory:` connections.

**Result: PARTIAL**

---

## Acceptance Criteria Verification

| AC | Description | Status | Evidence |
|----|-------------|--------|----------|
| AC-1 | Test step uncommented | ✅ PASS | `grep "Run unit tests"` returns active YAML (no `#`) |
| AC-2 | pytest command correct | ✅ PASS | `maxfail=10`, `-m "not integration and not slow"` |
| AC-3 | DATABASE_URL configured | ✅ PASS | `postgresql://test:test@localhost:5432/quantagent_test` |
| AC-4 | No `continue-on-error` | ✅ PASS | grep returns nothing for test step |
| NT-2 | `id: tests` present | ✅ PASS | Referenced correctly by status step |

---

## Test Suite Results (Local Run)

### CI Command: `pytest tests/ -v --tb=short --maxfail=10 -m "not integration and not slow"`

```
66 passed, 29 deselected, 10 errors in 4.94s
```

**10 errors hit `--maxfail`:** All from `test_backtest.py` — `ValueError: DATABASE_URL not configured`  
**Root cause:** No local PostgreSQL. In CI with `postgres:16` service, these 10 tests (marked `api`) **would pass**.

### Full Suite Run (without maxfail, excluding api + known JSONB)

```
23 failed, 381 passed, 21 skipped, 100 deselected, 56 errors in 44.81s
```

---

## Pre-existing Risk: JSONB/SQLite Incompatibility

**Problem:** `quantagent/models.py` contains `Column(JSONB)` (added via `feature/QuantAgent-yuk-logging` merge). Multiple test files hardcode `sqlite:///:memory:` as their database engine. SQLite cannot compile JSONB columns → `sqlalchemy.exc.CompileError`.

**Affected test files (4 files, ~50+ tests):**
- `tests/test_portfolio_manager.py` — hardcodes SQLite
- `tests/test_position_monitor.py` — hardcodes SQLite  
- `tests/test_position_monitor_constraints.py` — hardcodes SQLite (local fixture added on branch)
- `tests/test_r78_trade_pnl_calculation.py` — hardcodes SQLite

**Why PostgreSQL in CI doesn't fix this:** These tests create their own `create_engine("sqlite:///:memory:")` — they don't use `DATABASE_URL`. CI's PostgreSQL service is irrelevant to them.

**CI impact with current setup:**
- `test_backtest.py` (api): PASS in CI ✓
- JSONB tests hit alphabetically after `test_backtest.py`: FAIL in CI ✗  
- With `--maxfail=10`, CI job would fail → deploy blocked

**Is this introduced by QuantAgent-82t?** NO. The branch only changed `.github/workflows/main-ci-deploy.yml` (1 file, 7 lines). The JSONB issue predates this branch.

---

## Commands Run

```bash
git status --short
git branch --show-current
grep -A 7 "name: Run unit tests" .github/workflows/main-ci-deploy.yml
grep -A 2 "pytest tests/" .github/workflows/main-ci-deploy.yml
grep -A 1 "DATABASE_URL:" .github/workflows/main-ci-deploy.yml
grep -A 10 "name: Run unit tests" ... | grep "continue-on-error"  # (nothing)
grep "id: tests" .github/workflows/main-ci-deploy.yml
python -m compileall -q .  # PASS
pytest tests/ -v --tb=short --maxfail=10 -m "not integration and not slow"
pytest tests/ -v --tb=short -m "not integration and not slow and not api" --ignore=...
```

---

## Artifacts Produced

- `run-report.md` — this document
- `result.json` — machine-readable result
- `quality-gates.log` — gate results

---

## Next Step Recommended

1. Create a new issue to fix JSONB/SQLite test fixtures before merging QuantAgent-82t to main
2. Fix: replace `create_engine("sqlite:///:memory:")` with PostgreSQL URL (from env var) in the 4 affected test files, or add `pytest.importorskip` / SQLite-compatible type overrides
3. Alternative short-term: add `tests/test_portfolio_manager.py`, `tests/test_position_monitor.py` to an `integration` marker so the CI command (`-m "not integration and not slow"`) excludes them
4. Once tests pass cleanly in CI, merge QuantAgent-82t

---

## Workflow Change Summary

```diff
-      # Unit tests temporarily disabled (tracked for later deep dive)
-      # - name: Run unit tests
-      #   id: tests
-      #   env:
-      #     DATABASE_URL: postgresql://test:test@localhost:5432/quantagent_test
-      #   run: |
-      #     pytest tests/ -v --tb=short --maxfail=5 \
-      #       -m "not integration and not slow"
-      #   continue-on-error: true
+      - name: Run unit tests
+        id: tests
+        env:
+          DATABASE_URL: postgresql://test:test@localhost:5432/quantagent_test
+        run: |
+          pytest tests/ -v --tb=short --maxfail=10 \
+            -m "not integration and not slow"
```
