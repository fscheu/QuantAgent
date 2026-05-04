# QuantAgent-82t — Acceptance Criteria: Re-enable Unit Tests in CI Pipeline

**Issue ID:** QuantAgent-82t  
**Title:** Re-enable unit tests in CI pipeline (main-ci-deploy.yml)  
**Type:** Task  
**Updated:** 2026-05-04

---

## AC-1: Test Step Enabled

**Given** the `main-ci-deploy.yml` workflow file  
**When** viewing the file in the repository  
**Then**:
- Lines 67-75 are **not commented out**
- Step has `id: tests` identifier
- Step appears in the CI job steps list

**Verification:**
```bash
grep -A 7 "name: Run unit tests" .github/workflows/main-ci-deploy.yml | head -8
# Should show uncommented YAML (no leading # symbols)
```

---

## AC-2: Correct Test Command

**Given** the test step is enabled  
**When** examining the step configuration  
**Then** the `run` command is:
```yaml
run: |
  pytest tests/ -v --tb=short --maxfail=10 \
    -m "not integration and not slow"
```

**Verification:**
```bash
grep -A 2 "pytest tests/" .github/workflows/main-ci-deploy.yml
# Expected: pytest tests/ -v --tb=short --maxfail=10 \
```

---

## AC-3: Database Configuration

**Given** the test step is enabled  
**When** examining the step's `env` section  
**Then** `DATABASE_URL` is set to:
```yaml
env:
  DATABASE_URL: postgresql://test:test@localhost:5432/quantagent_test
```

**Verification:**
```bash
grep -A 1 "DATABASE_URL:" .github/workflows/main-ci-deploy.yml
# Expected: postgresql://test:test@localhost:5432/quantagent_test
```

---

## AC-4: No Error Tolerance

**Given** the test step is enabled  
**When** examining the step  
**Then**:
- `continue-on-error: true` is **NOT** present in the test step
- CI job fails and blocks deployment when tests fail

**Verification:**
```bash
grep -A 8 "name: Run unit tests" .github/workflows/main-ci-deploy.yml | grep "continue-on-error"
# Expected: (no output — pattern not found)
```

---

## AC-5: Deployment Blocking

**Given** tests fail in CI  
**When** the CI job completes  
**Then**:
- `deploy-qa` job does not run (blocked by `needs: ci` dependency)
- Telegram notification indicates CI failure with failed step

**Verification (manual):** Push a commit that intentionally breaks a test; confirm deploy-qa is skipped in GitHub Actions.

---

## AC-6: Success Path Unchanged

**Given** all tests pass in CI  
**When** the CI job completes  
**Then**:
- Deployment to QA proceeds normally
- Telegram notification: "✅ CI passed"
- Telegram notification: "🚀 QA Deploy Success"

---

## Edge Cases

### EC-1: PostgreSQL Not Ready
- Health checks (`--health-cmd pg_isready`, 5 retries) prevent test step from running before DB ready
- No action needed — already handled

### EC-2: Tests Timeout
- `maxfail=10` ensures tests fail fast; CI times out at GitHub's 6-hour default
- Expected: fast failure, not timeout

### EC-3: Only Slow/Integration Tests Failing
- Marker filter `-m "not integration and not slow"` excludes these
- Those failures won't block CI (correct behavior)

---

## Pre-Merge Checklist

- [ ] Lines 67-75 uncommented in `main-ci-deploy.yml`
- [ ] `maxfail=10` present
- [ ] `continue-on-error` absent from test step
- [ ] `DATABASE_URL` matches PostgreSQL service config
- [ ] Local test run with Docker PostgreSQL passes
- [ ] YAML syntax valid

## Post-Merge Checklist

- [ ] First GitHub Actions run completes with test step visible
- [ ] Telegram "✅ CI passed" received
- [ ] QA deployment proceeds
- [ ] Monitor 3+ runs for stability
