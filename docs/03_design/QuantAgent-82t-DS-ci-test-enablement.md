# QuantAgent-82t — Design: Re-enable Unit Tests in CI Pipeline

**Issue ID:** QuantAgent-82t  
**Title:** Re-enable unit tests in CI pipeline (main-ci-deploy.yml)  
**Type:** Task  
**Updated:** 2026-05-04 (line numbers corrected; blockers confirmed resolved)

---

## Design Overview

Uncomment and configure the existing "Run unit tests" step in `.github/workflows/main-ci-deploy.yml` to restore test execution in the CI pipeline. The PostgreSQL service container is already configured; we only need to activate the test step and ensure it properly blocks deployment on failure.

---

## Affected Components

### Modified
- `.github/workflows/main-ci-deploy.yml` — Uncomment test step (lines 67-75)

### Not Modified
- `.github/workflows/main-ci.yml` — Different workflow (handled by QuantAgent-ng1)
- `pytest.ini` — Configuration already correct
- `tests/` — Test files unchanged
- PostgreSQL service configuration — Already configured correctly

---

## Technical Changes

### Change 1: Uncomment Test Step

**Current (lines 67-75):**
```yaml
# Unit tests temporarily disabled (tracked for later deep dive)
# - name: Run unit tests
#   id: tests
#   env:
#     DATABASE_URL: postgresql://test:test@localhost:5432/quantagent_test
#   run: |
#     pytest tests/ -v --tb=short --maxfail=5 \
#       -m "not integration and not slow"
#   continue-on-error: true
```

**After (proposed):**
```yaml
- name: Run unit tests
  id: tests
  env:
    DATABASE_URL: postgresql://test:test@localhost:5432/quantagent_test
  run: |
    pytest tests/ -v --tb=short --maxfail=10 \
      -m "not integration and not slow"
```

**Key changes:**
1. Remove comment block (lines 67 and 68-75 prefixes)
2. Change `maxfail=5` → `maxfail=10` (see more failures before stopping)
3. **Remove** `continue-on-error: true` (block deployment on failure)

---

## Design Decisions

### Decision 1: Why Remove `continue-on-error: true`?

**Rationale:**
- Original intent: Allow CI to continue even if tests fail (during test suite stabilization)
- **Now:** Tests are fixed (QuantAgent-sfc + QuantAgent-4ch both CLOSED), should enforce quality gate
- **Benefit:** Prevents broken code from reaching QA

**Alternative considered:** Keep `continue-on-error: true`
- ❌ **Rejected:** Defeats the purpose of CI tests; broken code still deploys

### Decision 2: Why `maxfail=10` instead of `maxfail=5`?

**Rationale:**
- More context for debugging when failures occur
- 10 failures is still fast enough (tests fail-fast)
- Helps identify patterns in failures vs. single flaky test

**Alternative considered:** Remove `maxfail` entirely (run all tests)
- ❌ **Rejected:** Wastes CI minutes on known failures; fail-fast is better

### Decision 3: Why Keep Marker Filters `"not integration and not slow"`?

**Rationale:**
- **`not integration`:** Integration tests may require additional setup
- **`not slow`:** Keep CI fast (unit tests only)
- **Matches pytest.ini markers:** Already defined and in use

**Alternative considered:** Run all tests including integration/slow
- ❌ **Rejected:** CI would be slow; integration tests better suited for nightly/scheduled runs

### Decision 4: PostgreSQL Service Configuration

**Already configured correctly (lines 16-29):**
```yaml
services:
  postgres:
    image: postgres:16
    env:
      POSTGRES_USER: test       # ✓ Matches DATABASE_URL
      POSTGRES_PASSWORD: test   # ✓ Matches DATABASE_URL
      POSTGRES_DB: quantagent_test  # ✓ Matches DATABASE_URL
    ports:
      - 5432:5432              # ✓ Matches DATABASE_URL
    options: >-
      --health-cmd pg_isready
      --health-interval 10s
      --health-timeout 5s
      --health-retries 5
```

**No changes needed** — credentials and health checks already correct.

---

## Workflow Flow After Change

```
Push to main
  → Checkout code
  → Setup Python
  → Install dependencies
  → Lint with ruff
  → Run unit tests (RE-ENABLED)
      ↓ pass
  → Determine overall status
  → Notify Telegram (CI success)
  → Deploy to QA
```

**If tests fail:** Telegram CI failure notification; deploy-qa job blocked by `needs: ci`.

---

## Status Determination Logic

**Existing logic in "Determine overall status" step (no changes needed):**
```yaml
- name: Determine overall status
  id: status
  env:
    LINT_EXIT: ${{ steps.lint.outcome }}
    TESTS_EXIT: ${{ steps.tests.outcome || 'skipped' }}
  run: |
    if [[ "$LINT_EXIT" != "success" ]]; then
      echo "failed_step=Lint" >> $GITHUB_OUTPUT
      exit 1
    fi

    if [[ "$TESTS_EXIT" != "success" && "$TESTS_EXIT" != "skipped" ]]; then
      echo "failed_step=Unit tests" >> $GITHUB_OUTPUT
      exit 1
    fi

    echo "All checks passed"
```

**After uncommenting test step:**
- `TESTS_EXIT` will no longer be `'skipped'`
- If tests fail, `TESTS_EXIT` will be `'failure'`
- Status step will fail with `failed_step=Unit tests`
- Telegram notification will show "Failed step: Unit tests"

**No changes needed to this logic** — it already handles test failures correctly.

---

## Risk Assessment

### Risk 1: Flaky Tests
**Probability:** Low (QuantAgent-4ch fixed database setup)  
**Impact:** Medium (blocked deployments)  
**Mitigation:** Monitor first few runs; fail-fast with `maxfail=10`

### Risk 2: PostgreSQL Service Not Ready
**Probability:** Very Low (health checks configured)  
**Impact:** Medium  
**Mitigation:** Service has `--health-cmd pg_isready` with 5 retries

### Risk 3: YAML Syntax Error During Uncomment
**Probability:** Low  
**Impact:** Low (quick to fix)  
**Mitigation:** Validate YAML before commit; change is minimal (remove comment chars)

---

## Rollback Plan

If tests cause issues after re-enabling:

1. **Immediate revert:** `git revert HEAD && git push origin main`
2. **Re-comment test step** (hotfix): Add `#` back to lines 67-75, commit, push
3. **Fix forward:** Identify issue from GitHub Actions logs, fix in separate branch

---

## Open Questions

None — design is straightforward. Both blockers resolved.
