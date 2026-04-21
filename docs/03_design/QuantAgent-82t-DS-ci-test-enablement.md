# QuantAgent-82t — Design: Re-enable Unit Tests in CI Pipeline

**Issue ID:** QuantAgent-82t  
**Title:** Re-enable unit tests in CI pipeline (main-ci-deploy.yml)  
**Type:** Task

---

## Design Overview

Uncomment and configure the existing "Run unit tests" step in `.github/workflows/main-ci-deploy.yml` to restore test execution in the CI pipeline. The PostgreSQL service container is already configured; we only need to activate the test step and ensure it properly blocks deployment on failure.

---

## Affected Components

### Modified
- `.github/workflows/main-ci-deploy.yml` — Uncomment test step (lines 62-69)

### Not Modified
- `.github/workflows/main-ci.yml` — Different workflow (handled by QuantAgent-ng1)
- `pytest.ini` — Configuration already correct
- `tests/` — Test files unchanged
- PostgreSQL service configuration — Already configured correctly

---

## Technical Changes

### Change 1: Uncomment Test Step

**Current (lines 62-69):**
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
1. Remove all comment symbols (`#`)
2. Change `maxfail=5` → `maxfail=10` (see more failures before stopping)
3. **Remove** `continue-on-error: true` (block deployment on failure)

---

## Design Decisions

### Decision 1: Why Remove `continue-on-error: true`?

**Rationale:**
- Original intent: Allow CI to continue even if tests fail (during test suite stabilization)
- **Now:** Tests are fixed (QuantAgent-sfc + QuantAgent-4ch), should enforce quality gate
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
- **`not integration`:** Integration tests may require additional setup (external APIs, etc.)
- **`not slow`:** Keep CI fast (unit tests only)
- **Matches pytest.ini markers:** Already defined and in use

**Alternative considered:** Run all tests including integration/slow
- ❌ **Rejected:** CI would be slow; integration tests better suited for nightly/scheduled runs

### Decision 4: PostgreSQL Service Configuration

**Already configured correctly:**
```yaml
services:
  postgres:
    image: postgres:16
    env:
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
      POSTGRES_DB: quantagent_test
    ports:
      - 5432:5432
    options: >-
      --health-cmd pg_isready
      --health-interval 10s
      --health-timeout 5s
      --health-retries 5
```

**No changes needed:**
- ✅ Correct PostgreSQL version (16)
- ✅ Test database `quantagent_test` created
- ✅ Health checks ensure database is ready before tests run
- ✅ Credentials match `DATABASE_URL` in test step

---

## Workflow Flow After Change

```mermaid
graph TD
    A[Push to main] --> B[Checkout code]
    B --> C[Setup Python]
    C --> D[Install dependencies]
    D --> E[Lint with ruff]
    E --> F{Lint passed?}
    F -->|No| G[Notify Telegram - CI Fail]
    F -->|Yes| H[Run unit tests]
    H --> I{Tests passed?}
    I -->|No| J[Notify Telegram - CI Fail]
    I -->|Yes| K[Notify Telegram - CI Success]
    K --> L[Deploy to QA]
    L --> M[Health check]
    M --> N{Health OK?}
    N -->|Yes| O[Notify Telegram - Deploy Success]
    N -->|No| P[Notify Telegram - Deploy Fail]
```

**Key points:**
- Tests run **after** lint (existing order preserved)
- Test failure **blocks** deployment (new behavior)
- Telegram notifications triggered at each stage

---

## Status Determination Logic

**Existing logic in "Determine overall status" step:**
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

## Alternative Approaches Considered

### ❌ Approach 1: Add `continue-on-error: false` Explicitly
**Why rejected:**
- Default behavior is already to fail on error
- Explicit `false` is redundant
- Simpler to just remove the line

### ❌ Approach 2: Use Separate Job for Tests
**Why rejected:**
- More complex (requires artifact sharing, status checks)
- Current single-job approach is simpler
- No benefit for this use case

### ❌ Approach 3: Run Tests Before Lint
**Why rejected:**
- Lint is faster; fail fast on style issues first
- Tests require database setup; slower
- Existing order is sensible

---

## Testing Strategy

### Pre-Merge Testing
1. Create test branch with changes
2. Temporarily push to `main` (or use `workflow_dispatch`)
3. Verify CI runs tests
4. Verify test failures block deployment
5. Verify Telegram notifications sent

### Post-Merge Validation
1. First commit to `main` after merge should run tests
2. Monitor GitHub Actions UI
3. Confirm Telegram notifications received
4. Verify QA deployment occurs (if tests pass)

---

## Risk Assessment

### Risk 1: Flaky Tests
**Description:** Intermittent test failures cause false positives

**Mitigation:**
- QuantAgent-4ch fixed database setup (reduces flakiness)
- Monitor first few runs closely
- If flakiness detected, identify and fix root cause

**Probability:** Low (after QuantAgent-4ch fix)  
**Impact:** Medium (blocked deployments)

### Risk 2: PostgreSQL Service Not Ready
**Description:** Tests run before PostgreSQL is healthy

**Mitigation:**
- Service already has health checks configured
- GitHub Actions waits for healthy status before running tests
- Tested in existing configuration

**Probability:** Very Low  
**Impact:** Medium

### Risk 3: Test Suite Too Slow
**Description:** CI takes too long, slows down development

**Mitigation:**
- Only run unit tests (exclude `integration` and `slow`)
- Fail-fast with `maxfail=10`
- Monitor CI duration; optimize if needed

**Probability:** Low  
**Impact:** Low

---

## Rollback Plan

If tests cause issues after re-enabling:

1. **Immediate rollback:**
   ```bash
   git revert <commit-hash>
   git push origin main
   ```

2. **Re-comment test step** (hotfix):
   - Add `#` back to lines 62-69
   - Commit with message: "Hotfix: Temporarily disable tests (investigate issue)"
   - Push to `main`

3. **Investigation:**
   - Check GitHub Actions logs
   - Identify root cause
   - Fix issue in separate branch
   - Re-enable tests

---

## Success Metrics

### Immediate (Day 1)
- [ ] Test step runs on first push to `main`
- [ ] No errors in GitHub Actions logs
- [ ] Telegram notifications received correctly

### Short-term (Week 1)
- [ ] 5+ successful CI runs with tests
- [ ] Zero false positives (flaky tests)
- [ ] QA deployments only occur when tests pass

### Long-term (Month 1)
- [ ] Test suite catches at least 1 real bug before QA
- [ ] CI duration remains < 5 minutes
- [ ] Developer confidence in CI increases

---

## Open Questions

None — design is straightforward.
