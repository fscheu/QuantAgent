# QuantAgent-82t — Planning: Re-enable Unit Tests in CI Pipeline

**Issue ID:** QuantAgent-82t  
**Title:** Re-enable unit tests in CI pipeline (main-ci-deploy.yml)  
**Type:** Task  
**Priority:** 2

---

## Objective

Re-enable unit tests in the CI pipeline by uncommenting the test step in `main-ci-deploy.yml`, ensuring tests run before deployment to QA.

---

## Prerequisites (Blockers)

**Must be completed before starting:**
- ✅ **QuantAgent-sfc**: SQLite/JSONB compatibility fixes
- ✅ **QuantAgent-4ch**: Test database configuration (SQLite for local, PostgreSQL for CI)

**Verification:**
```bash
# Confirm blockers are merged to main
git log --oneline --grep="QuantAgent-sfc\|QuantAgent-4ch" origin/main

# Expected: Recent commits for both issues
```

---

## Tasks

### Task 1: Uncomment Test Step
**Estimate:** 0.25h

**What:**
- Edit `.github/workflows/main-ci-deploy.yml`
- Remove comment symbols (`#`) from lines 62-69
- Locate the section:
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
- Uncomment all lines (remove leading `#` and one space)

**How to validate:**
```bash
# Check that lines are uncommented
grep -A 8 "name: Run unit tests" .github/workflows/main-ci-deploy.yml | \
  grep -v "^#"

# Should show all YAML lines without comment syntax
```

**Dependencies:** None

---

### Task 2: Update Pytest Command Parameters
**Estimate:** 0.1h

**What:**
- Change `--maxfail=5` to `--maxfail=10`
- Keep other parameters unchanged:
  - `-v` (verbose)
  - `--tb=short` (short traceback)
  - `-m "not integration and not slow"` (marker filter)

**Why:**
- More context for debugging (see up to 10 failures)
- Still fail-fast (doesn't run all ~600 tests if failures occur)

**Exact change:**
```diff
-     pytest tests/ -v --tb=short --maxfail=5 \
+     pytest tests/ -v --tb=short --maxfail=10 \
```

**How to validate:**
```bash
grep "maxfail=" .github/workflows/main-ci-deploy.yml
# Expected: --maxfail=10
```

**Dependencies:** Task 1

---

### Task 3: Remove Error Tolerance
**Estimate:** 0.05h

**What:**
- Delete the line: `continue-on-error: true`
- This ensures test failures **block** the deployment

**Why:**
- Original purpose: Allow CI to proceed during test stabilization
- Now: Tests are stable, should enforce quality gate

**Exact change:**
```diff
      -m "not integration and not slow"
-   continue-on-error: true
```

**How to validate:**
```bash
# Verify continue-on-error is NOT present in test step
grep -A 8 "name: Run unit tests" .github/workflows/main-ci-deploy.yml | \
  grep "continue-on-error"

# Expected: (no output - pattern not found)
```

**Dependencies:** Task 1

---

### Task 4: Verify DATABASE_URL Configuration
**Estimate:** 0.1h

**What:**
- Confirm `DATABASE_URL` is present in the `env` section of the test step
- Verify it matches PostgreSQL service configuration:
  ```yaml
  env:
    DATABASE_URL: postgresql://test:test@localhost:5432/quantagent_test
  ```

**Why:**
- Tests require database connection
- Must match service container credentials and database name

**Cross-reference with service config (lines 14-29):**
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
```

**How to validate:**
```bash
# Extract DATABASE_URL from test step
grep -A 2 "DATABASE_URL:" .github/workflows/main-ci-deploy.yml | \
  grep "postgresql://"

# Expected: postgresql://test:test@localhost:5432/quantagent_test
```

**Dependencies:** Task 1

---

### Task 5: Test Locally (Pre-Push Validation)
**Estimate:** 0.5h

**What:**
- Run tests locally with PostgreSQL to simulate CI environment
- Use Docker Compose to match CI setup

**Commands:**
```bash
# Start PostgreSQL service (simulating CI)
docker run --name test-postgres -d \
  -e POSTGRES_USER=test \
  -e POSTGRES_PASSWORD=test \
  -e POSTGRES_DB=quantagent_test \
  -p 5432:5432 \
  postgres:16

# Wait for PostgreSQL to be ready
sleep 5

# Run tests (same command as CI)
export DATABASE_URL=postgresql://test:test@localhost:5432/quantagent_test
pytest tests/ -v --tb=short --maxfail=10 \
  -m "not integration and not slow"

# Cleanup
docker stop test-postgres
docker rm test-postgres
```

**Expected result:**
- All tests pass
- No database connection errors
- Exit code 0

**If tests fail:**
- Fix tests or investigate (likely issue with QuantAgent-4ch)
- Do **not** proceed to Task 6

**Dependencies:** Tasks 1-4

---

### Task 6: Commit and Push Changes
**Estimate:** 0.1h

**What:**
- Stage changes to workflow file
- Commit with descriptive message
- Push to main (or feature branch for review)

**Commands:**
```bash
# Stage changes
git add .github/workflows/main-ci-deploy.yml

# Commit
git commit -m "[QuantAgent-82t] Re-enable unit tests in CI pipeline

- Uncommented 'Run unit tests' step
- Updated maxfail=5 to maxfail=10
- Removed continue-on-error (tests now block deployment)
- Verified DATABASE_URL matches PostgreSQL service

Closes QuantAgent-82t"

# Push (to feature branch for PR, or directly to main if allowed)
git push origin feature/QuantAgent-82t-ci-test-enablement
# or
git push origin main
```

**Dependencies:** Task 5 (must pass)

---

### Task 7: Monitor First CI Run
**Estimate:** 0.25h

**What:**
- Observe GitHub Actions workflow execution
- Verify test step runs successfully
- Check Telegram notifications

**Steps:**
1. Navigate to GitHub Actions tab
2. Find the workflow run triggered by push
3. Expand "CI (Lint + Tests)" job
4. Verify steps execute in order:
   - ✅ Checkout code
   - ✅ Set up Python
   - ✅ Install dependencies
   - ✅ Lint
   - ✅ **Run unit tests** (newly enabled)
   - ✅ Determine overall status
   - ✅ Notify Telegram

5. Check Telegram for notifications:
   - "✅ CI passed" (if all tests pass)
   - "🚀 QA Deploy Success" (if deployment succeeds)

**If CI fails:**
- Check logs for failure reason
- If test-related: investigate test issue
- If config-related: review workflow file changes

**Dependencies:** Task 6

---

## Total Estimate

**1.35 hours** (7 tasks)

**Breakdown:**
- Configuration changes: 0.5h (Tasks 1-4)
- Validation: 0.6h (Task 5)
- Deployment: 0.25h (Tasks 6-7)

---

## Execution Order

1. **Task 1** (uncomment step) — Foundation
2. **Task 2** (update maxfail) — Parameter tuning
3. **Task 3** (remove continue-on-error) — Quality gate enforcement
4. **Task 4** (verify DATABASE_URL) — Validation
5. **Task 5** (test locally) — Pre-push validation ⚠️ **Critical**
6. **Task 6** (commit and push) — Deployment
7. **Task 7** (monitor CI) — Post-deployment validation

---

## Risks & Mitigations

### Risk 1: Tests Fail in CI (But Passed Locally)
**Description:** Environment differences cause CI-only failures

**Mitigation:**
- Task 5 uses Docker PostgreSQL to match CI environment
- QuantAgent-4ch fixed database setup (reduces env differences)
- If occurs: check GitHub Actions logs, compare env vars

**Probability:** Low  
**Impact:** Medium (blocked deployment until fixed)

---

### Risk 2: PostgreSQL Service Not Healthy
**Description:** Service container fails health checks, tests can't connect

**Mitigation:**
- Service config already has `--health-cmd pg_isready`
- Health checks have 5 retries with 10s interval (up to 50s wait)
- If occurs: check service logs in GitHub Actions

**Probability:** Very Low  
**Impact:** Medium

---

### Risk 3: Workflow Syntax Error
**Description:** YAML formatting mistake breaks workflow

**Mitigation:**
- Use YAML validator before commit:
  ```bash
  yamllint .github/workflows/main-ci-deploy.yml
  ```
- GitHub Actions will show syntax error immediately
- Easy to fix and re-push

**Probability:** Low (simple uncomment operation)  
**Impact:** Low (quick fix)

---

## Rollback Plan

If CI breaks after enabling tests:

### Option 1: Immediate Revert (< 5 min)
```bash
git revert HEAD
git push origin main
```

### Option 2: Re-Comment Test Step (< 2 min)
```bash
# Edit .github/workflows/main-ci-deploy.yml
# Add # back to lines 62-69
git add .github/workflows/main-ci-deploy.yml
git commit -m "Hotfix: Re-disable tests (investigate issue)"
git push origin main
```

### Option 3: Fix Forward (variable time)
- Identify issue from GitHub Actions logs
- Fix in new commit
- Push fix

**Choose Option 1 or 2 if:** Issue unclear, need time to investigate  
**Choose Option 3 if:** Issue obvious and quick to fix

---

## Success Criteria

- [ ] Test step uncommented in workflow file
- [ ] `maxfail=10` configured
- [ ] `continue-on-error: true` removed
- [ ] `DATABASE_URL` verified correct
- [ ] Local tests pass (Task 5)
- [ ] Changes committed and pushed
- [ ] First CI run successful
- [ ] Telegram notifications received
- [ ] QA deployment completed (if tests pass)

---

## Testing Strategy

### Pre-Merge (Task 5)
- ✅ Run tests with Docker PostgreSQL locally
- ✅ Verify all tests pass
- ✅ Confirm database connection works

### Post-Merge (Task 7)
- ✅ Monitor first GitHub Actions run
- ✅ Verify test step executes
- ✅ Check Telegram notifications
- ✅ Validate QA deployment

### Ongoing (Next Week)
- Monitor 5-10 CI runs
- Watch for flaky tests
- Track CI duration
- Ensure no regressions

---

## Next Steps After This Issue

1. **Monitor CI stability** (first week)
2. **QuantAgent-ng1**: Update `main-ci.yml` (separate workflow)
3. **Future enhancements**:
   - Add code coverage reporting
   - Run integration tests nightly
   - Optimize CI performance if needed

---

## Documentation Updates

**Files created:**
- `docs/01_requirements/QuantAgent-82t-RQ-ci-test-enablement.md` ✓
- `docs/03_design/QuantAgent-82t-DS-ci-test-enablement.md` ✓
- `docs/05_acceptance_tests/QuantAgent-82t-AC-ci-test-enablement.md` ✓
- `docs/02_planning/QuantAgent-82t-PL-ci-test-enablement.md` (this file) ✓

**Files modified:**
- `.github/workflows/main-ci-deploy.yml` (pending implementation)

---

## Final Checklist

Before starting implementation:
- [ ] QuantAgent-sfc merged to main
- [ ] QuantAgent-4ch merged to main
- [ ] Local environment has Docker installed (for Task 5)
- [ ] Telegram bot configured for notifications

During implementation:
- [ ] Complete Tasks 1-4 (workflow changes)
- [ ] Run Task 5 (local validation) ⚠️ **Do not skip**
- [ ] Complete Task 6 (commit and push)
- [ ] Complete Task 7 (monitor first run)

After implementation:
- [ ] Update Beads status to `test_done` or `merged`
- [ ] Add comment to QuantAgent-82t with results
- [ ] Monitor next 5 CI runs for stability
