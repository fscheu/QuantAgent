# QuantAgent-82t — Requirements: Re-enable Unit Tests in CI Pipeline

**Issue ID:** QuantAgent-82t  
**Title:** Re-enable unit tests in CI pipeline (main-ci-deploy.yml)  
**Type:** Task  
**Priority:** 2  
**Labels:** openclaw:design_approved ci  
**Blocked by:** ~~QuantAgent-sfc~~, ~~QuantAgent-4ch~~ (both CLOSED as of 2026-04-24)

---

## Objective

Re-enable the commented-out unit test step in `.github/workflows/main-ci-deploy.yml` to ensure the CI pipeline runs tests before deploying to QA, preventing broken code from reaching the QA environment.

---

## Background

Currently, the "Run unit tests" step in `main-ci-deploy.yml` is disabled (lines 67-75):
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

This means:
- Only linting runs in CI (ruff check)
- Tests are skipped, allowing broken code to deploy
- QA environment can receive untested changes

With **QuantAgent-sfc** (SQLite/JSONB fixes, CLOSED 2026-04-24) and **QuantAgent-4ch** (test database configuration, CLOSED 2026-04-28) resolved, the test suite is clean and ready to run in CI.

---

## Scope

### In Scope
- Uncomment the "Run unit tests" step in `main-ci-deploy.yml`
- Update pytest command to use correct parameters (`maxfail=10`)
- Remove `continue-on-error: true` (tests must block deployment on failure)
- Verify `DATABASE_URL` environment variable is configured correctly
- Ensure test failures trigger Telegram notifications

### Out of Scope
- Modifying test files or test logic
- Changes to `main-ci.yml` workflow (separate ticket: QuantAgent-ng1)
- Adding new test markers or pytest configuration
- Performance optimization of CI pipeline
- Adding code coverage reporting

---

## Current Behavior (Broken)

1. Developer pushes to `main` branch
2. GitHub Actions triggers `main-ci-deploy.yml`
3. CI runs:
   - ✅ Lint (ruff check)
   - ⚠️ Tests skipped (commented out)
4. If lint passes → deploys to QA
5. **Problem:** Broken tests don't block deployment

---

## Expected Behavior (Fixed)

1. Developer pushes to `main` branch
2. GitHub Actions triggers `main-ci-deploy.yml`
3. CI runs:
   - ✅ Lint (ruff check)
   - ✅ Unit tests (pytest with PostgreSQL service)
4. **If tests fail:**
   - ❌ Deployment to QA is blocked
   - 🚨 Telegram notification sent with failure details
5. **If all checks pass:**
   - ✅ Deploys to QA
   - ✅ Telegram notification confirms success

---

## Acceptance Criteria

### AC-1: Test Step Enabled
**Given** the `main-ci-deploy.yml` workflow  
**When** a commit is pushed to `main`  
**Then** the "Run unit tests" step executes (not commented out)

### AC-2: Correct Test Command
**Given** the test step is enabled  
**When** tests execute  
**Then** the command is:
```bash
pytest tests/ -v --tb=short --maxfail=10 -m "not integration and not slow"
```

### AC-3: Database Configuration
**Given** the test step is enabled  
**When** tests execute  
**Then** `DATABASE_URL` environment variable is set to:
```
postgresql://test:test@localhost:5432/quantagent_test
```

### AC-4: No Error Tolerance
**Given** the test step is enabled  
**When** tests fail  
**Then**:
- `continue-on-error: true` is **NOT** present
- CI job fails and blocks deployment
- GitHub Actions shows red ❌ status

### AC-5: Deployment Blocking
**Given** tests fail in CI  
**When** the CI job completes  
**Then**:
- `deploy-qa` job does not run (blocked by `needs: ci` dependency)
- No deployment to QA occurs
- Telegram notification indicates CI failure with failed step

### AC-6: Success Path Unchanged
**Given** all tests pass in CI  
**When** the CI job completes  
**Then**:
- Deployment to QA proceeds normally
- Telegram notifications sent for CI success and deploy success

---

## Constraints

- **No test modifications**: Tests themselves are not changed
- **PostgreSQL service already configured**: Workflow already has `postgres:16` service container
- **Marker compatibility**: Use existing pytest markers (`integration`, `slow`)
- **Backwards compatible**: No breaking changes to existing workflow structure

---

## Dependencies

### Blockers (Resolved)
- **QuantAgent-sfc**: ✅ CLOSED 2026-04-24 — SQLite/JSONB compatibility fixed
- **QuantAgent-4ch**: ✅ CLOSED 2026-04-28 — Test database configuration fixed

### Related Issues
- **QuantAgent-ng1**: Update `main-ci.yml` (separate file, different workflow)

---

## Definition of Done

- [ ] Test step uncommented in `main-ci-deploy.yml`
- [ ] Correct pytest command configured (`maxfail=10`)
- [ ] `DATABASE_URL` environment variable present
- [ ] `continue-on-error: true` removed
- [ ] Push to `main` with passing tests → deploys to QA
- [ ] Push to `main` with failing tests → blocks deployment + Telegram alert
- [ ] Documentation updated
