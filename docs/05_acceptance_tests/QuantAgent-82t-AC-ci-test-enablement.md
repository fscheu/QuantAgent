# QuantAgent-82t — Acceptance Criteria: Re-enable Unit Tests in CI Pipeline

**Issue ID:** QuantAgent-82t  
**Title:** Re-enable unit tests in CI pipeline (main-ci-deploy.yml)  
**Type:** Task

---

## AC-1: Test Step Enabled

**Given** the `main-ci-deploy.yml` workflow file  
**When** viewing the file in the repository  
**Then**:
- Lines containing "Run unit tests" step are **not commented out**
- Step has `id: tests` identifier
- Step appears in the CI job steps list

**Verification:**
```bash
# Check that test step is uncommented
grep -A 7 "name: Run unit tests" .github/workflows/main-ci-deploy.yml | head -8
# Should show uncommented YAML (no leading # symbols)

# Verify in GitHub UI
# Navigate to: .github/workflows/main-ci-deploy.yml
# Line ~62-69 should show active step (no comment syntax)
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
# Extract pytest command from workflow
grep -A 2 "pytest tests/" .github/workflows/main-ci-deploy.yml

# Expected output:
#     pytest tests/ -v --tb=short --maxfail=10 \
#       -m "not integration and not slow"
```

**Breakdown:**
- ✅ `tests/` — Run tests in tests directory
- ✅ `-v` — Verbose output
- ✅ `--tb=short` — Short traceback format
- ✅ `--maxfail=10` — Stop after 10 failures
- ✅ `-m "not integration and not slow"` — Exclude integration and slow tests

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
# Check DATABASE_URL in workflow
grep -A 1 "DATABASE_URL:" .github/workflows/main-ci-deploy.yml

# Expected:
#     DATABASE_URL: postgresql://test:test@localhost:5432/quantagent_test
```

**Component validation:**
- ✅ Protocol: `postgresql://`
- ✅ User: `test`
- ✅ Password: `test`
- ✅ Host: `localhost` (service container)
- ✅ Port: `5432` (mapped from service)
- ✅ Database: `quantagent_test` (matches service config)

---

## AC-4: No Error Tolerance

**Given** the test step configuration  
**When** examining the step for error handling  
**Then**:
- `continue-on-error: true` is **NOT present** in the step
- No other error suppression directives exist

**Verification:**
```bash
# Check that continue-on-error is NOT in test step
grep -A 10 "name: Run unit tests" .github/workflows/main-ci-deploy.yml | \
  grep "continue-on-error"

# Expected: (no output - grep finds nothing)
# Exit code: 1 (pattern not found)
```

**Negative test (anti-pattern detection):**
```bash
# Verify test step will fail on error
# (default GitHub Actions behavior when continue-on-error absent)
yq eval '.jobs.ci.steps[] | select(.name == "Run unit tests") | has("continue-on-error")' \
  .github/workflows/main-ci-deploy.yml

# Expected: false (or key doesn't exist)
```

---

## AC-5: Deployment Blocking

**Given** tests fail in CI  
**When** the CI job completes  
**Then**:
- CI job status is **failure** (red ❌)
- `deploy-qa` job does **not** run
- Telegram notification sent with `failed_step: Unit tests`

**Verification (simulated failure):**

1. **Introduce test failure:**
   ```bash
   # Temporarily break a test
   echo "def test_fail(): assert False" >> tests/test_temporary.py
   git add tests/test_temporary.py
   git commit -m "Test: Verify CI blocks on test failure"
   git push origin main
   ```

2. **Observe GitHub Actions:**
   - CI job shows ❌ red status
   - "Run unit tests" step failed
   - "Determine overall status" step failed with `failed_step=Unit tests`
   - `deploy-qa` job is **skipped** (not run)

3. **Check Telegram:**
   - Notification received with:
     ```
     🚨 CI Failure
     Failed step: Unit tests
     Logs: [link]
     ```

4. **Cleanup:**
   ```bash
   git rm tests/test_temporary.py
   git commit -m "Cleanup: Remove test failure"
   git push origin main
   ```

---

## AC-6: Success Path Unchanged

**Given** all tests pass in CI  
**When** the CI job completes  
**Then**:
- CI job status is **success** (green ✅)
- `deploy-qa` job runs (triggered by `needs: ci`)
- Telegram notifications for both CI success and deploy success

**Verification (normal operation):**

1. **Push valid change:**
   ```bash
   # Make trivial documentation change
   echo "# Test CI" >> docs/test_ci.md
   git add docs/test_ci.md
   git commit -m "Test: Verify CI passes and deploys"
   git push origin main
   ```

2. **Observe GitHub Actions:**
   - CI job shows ✅ green status
   - All steps passed (lint + tests)
   - `deploy-qa` job runs after CI completes
   - QA deployment succeeds

3. **Check Telegram:**
   - First notification:
     ```
     ✅ CI passed
     Commit: [hash]
     Message: Test: Verify CI passes and deploys
     ```
   - Second notification (after deploy):
     ```
     🚀 QA Deploy Success
     URL: https://qa.fedes.dev
     ```

4. **Cleanup:**
   ```bash
   git rm docs/test_ci.md
   git commit -m "Cleanup: Remove test file"
   git push origin main
   ```

---

## Edge Cases

### EC-1: Empty Test Suite
**Given** no tests exist in `tests/` directory  
**When** CI runs  
**Then** pytest reports "no tests collected" but **exits 0** (success)

**Expected behavior:** CI passes (no tests = no failures)

**Verification:**
```bash
# Check pytest behavior with no tests
pytest --collect-only tests/ | grep "no tests collected"
echo $?  # Should be 0 (success)
```

### EC-2: Database Service Failure
**Given** PostgreSQL service fails to start  
**When** test step attempts to run  
**Then**:
- Tests fail with connection error
- CI job fails
- Deployment blocked

**Expected behavior:** Health checks should prevent this (service has `--health-cmd pg_isready`)

**Monitoring:** Check GitHub Actions logs for PostgreSQL service health

### EC-3: Import Errors
**Given** tests have import errors (e.g., missing dependency)  
**When** pytest collects tests  
**Then**:
- Pytest reports collection error
- CI job fails
- Deployment blocked

**Expected behavior:** Same as test failure (blocks deployment)

### EC-4: Timeout (No Response)
**Given** a test hangs indefinitely  
**When** CI runs tests  
**Then**:
- GitHub Actions job timeout (default: 360 min, unlikely to hit)
- Job fails
- Deployment blocked

**Expected behavior:** `maxfail=10` ensures fail-fast, but no explicit timeout on test step

**Mitigation (if needed):** Add `timeout-minutes: 10` to test step

---

## Performance Criteria

### P-1: CI Duration
**Given** the full CI pipeline (lint + tests + deploy)  
**When** all steps run successfully  
**Then**:
- CI job completes in < 10 minutes
- Test step specifically < 5 minutes

**Measurement:**
- Check GitHub Actions duration in UI
- Monitor over 10 runs to establish baseline

**Acceptance threshold:**
- 🟢 < 5 min: Excellent
- 🟡 5-10 min: Acceptable
- 🔴 > 10 min: Investigate optimization

### P-2: Resource Usage
**Given** tests run with PostgreSQL service  
**When** monitoring GitHub Actions runner  
**Then**:
- Memory usage < 2 GB
- No OOM errors

**Expected:** Standard unit tests should be lightweight

---

## Negative Test Cases

### NT-1: Lint Failure Still Blocks
**Given** lint fails (ruff check finds errors)  
**When** CI runs  
**Then**:
- CI fails before running tests
- Tests are **not** run (fail-fast)
- Deployment blocked

**Verification:**
```bash
# Introduce lint error
echo "import os,sys" >> quantagent/temporary.py  # Multiple imports on one line
git add quantagent/temporary.py
git commit -m "Test: Verify lint failure blocks"
git push origin main

# Expected:
# - Lint step fails
# - Test step doesn't run
# - Deploy blocked
```

### NT-2: Test Step ID Correct
**Given** test step is enabled  
**When** examining step configuration  
**Then** `id: tests` is present

**Why it matters:** `steps.tests.outcome` is referenced in status determination

**Verification:**
```bash
grep "id: tests" .github/workflows/main-ci-deploy.yml
# Expected: line containing "id: tests" (uncommented)
```

---

## Integration with Existing Workflow

### Telegram Notifications
**Existing logic should work without modification:**

1. **CI Failure Notification:**
   ```bash
   Failed step: ${{ steps.status.outputs.failed_step || 'Unknown' }}
   ```
   - If tests fail: `failed_step=Unit tests`
   - If lint fails: `failed_step=Lint`

2. **CI Success Notification:**
   - Sent only if all steps pass
   - Existing logic unchanged

3. **Deploy Notifications:**
   - Only triggered if `deploy-qa` job runs
   - Blocked by CI failure

**No changes needed to notification logic** — already handles test failures.

---

## Manual Test Procedure

### Setup
```bash
# Clone repo and switch to feature branch
git clone <repo>
cd QuantAgent
git checkout feature/QuantAgent-82t-ci-test-enablement
```

### Test 1: Verify Uncommenting
```bash
# Check file diff
git diff main .github/workflows/main-ci-deploy.yml

# Should show:
# - Removed: # (comment symbols) from lines 62-69
# - Removed: continue-on-error: true
# - Changed: maxfail=5 → maxfail=10
```

### Test 2: Simulate Success
```bash
# Push to main (after blockers resolved)
git push origin main

# Monitor GitHub Actions:
# 1. CI job runs
# 2. Lint passes
# 3. Tests run and pass
# 4. Deploy job runs
# 5. QA deployment succeeds

# Check Telegram:
# - ✅ CI passed
# - 🚀 QA Deploy Success
```

### Test 3: Simulate Failure
```bash
# Create failing test
echo "def test_fail(): assert False" >> tests/test_ci_verification.py
git add tests/test_ci_verification.py
git commit -m "CI Test: Verify failure blocks deployment"
git push origin main

# Monitor GitHub Actions:
# 1. CI job runs
# 2. Lint passes
# 3. Tests run and FAIL
# 4. Deploy job SKIPPED

# Check Telegram:
# - 🚨 CI Failure
# - Failed step: Unit tests

# Cleanup
git rm tests/test_ci_verification.py
git commit -m "Cleanup: Remove test failure"
git push origin main
```

---

## Definition of Done (Testing Checklist)

- [ ] AC-1: Test step uncommented in workflow file
- [ ] AC-2: Correct pytest command (`maxfail=10`, markers)
- [ ] AC-3: DATABASE_URL configured correctly
- [ ] AC-4: No `continue-on-error: true` present
- [ ] AC-5: Test failure blocks deployment (verified)
- [ ] AC-6: Test success allows deployment (verified)
- [ ] EC-1: Empty test suite handled gracefully
- [ ] EC-2: Database service health checks working
- [ ] P-1: CI completes in < 10 minutes
- [ ] NT-1: Lint failure still blocks (not bypassed)
- [ ] NT-2: Test step ID is `tests` (for status check)
- [ ] Manual Test 2 (success path) completed
- [ ] Manual Test 3 (failure path) completed
- [ ] Telegram notifications received for all scenarios
