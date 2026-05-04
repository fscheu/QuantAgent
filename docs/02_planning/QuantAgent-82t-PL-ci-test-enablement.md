# QuantAgent-82t — Planning: Re-enable Unit Tests in CI Pipeline

**Issue ID:** QuantAgent-82t  
**Title:** Re-enable unit tests in CI pipeline (main-ci-deploy.yml)  
**Type:** Task  
**Priority:** 2  
**Updated:** 2026-05-04 (blockers resolved; ready for implementation)

---

## Objective

Re-enable unit tests in the CI pipeline by uncommenting the test step in `main-ci-deploy.yml`, ensuring tests run before deployment to QA.

---

## Prerequisites (Blockers — Both Resolved)

- ✅ **QuantAgent-sfc**: SQLite/JSONB compatibility fixes — CLOSED 2026-04-24
- ✅ **QuantAgent-4ch**: Test database configuration — CLOSED 2026-04-28

**Verification:**
```bash
# Confirm blockers are merged to main
git log --oneline --grep="QuantAgent-sfc\|QuantAgent-4ch" origin/main
```

---

## Tasks

### Task 1: Uncomment Test Step
**Estimate:** 0.25h

**What:**
Edit `.github/workflows/main-ci-deploy.yml`, lines 67-75.

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

**After:**
```yaml
- name: Run unit tests
  id: tests
  env:
    DATABASE_URL: postgresql://test:test@localhost:5432/quantagent_test
  run: |
    pytest tests/ -v --tb=short --maxfail=10 \
      -m "not integration and not slow"
```

**Validation:**
```bash
grep -A 7 "name: Run unit tests" .github/workflows/main-ci-deploy.yml
# Should show uncommented YAML (no leading # symbols)
```

---

### Task 2: Update Pytest Command Parameters
**Estimate:** 0.1h

**What:** Change `--maxfail=5` to `--maxfail=10` (included in Task 1).

**Validation:**
```bash
grep "maxfail=" .github/workflows/main-ci-deploy.yml
# Expected: --maxfail=10
```

---

### Task 3: Remove Error Tolerance
**Estimate:** 0.05h

**What:** Delete `continue-on-error: true` from the test step (included in Task 1).

**Validation:**
```bash
grep -A 8 "name: Run unit tests" .github/workflows/main-ci-deploy.yml | grep "continue-on-error"
# Expected: (no output)
```

---

### Task 4: Verify DATABASE_URL Configuration
**Estimate:** 0.1h

**What:** Confirm `DATABASE_URL` is correctly set in the test step env.

**Validation:**
```bash
grep -A 2 "DATABASE_URL:" .github/workflows/main-ci-deploy.yml | grep "postgresql://"
# Expected: postgresql://test:test@localhost:5432/quantagent_test
```

Cross-reference with service config (lines 16-29):
- `POSTGRES_USER: test` ✓
- `POSTGRES_PASSWORD: test` ✓
- `POSTGRES_DB: quantagent_test` ✓
- Port `5432` ✓

---

### Task 5: Test Locally (Pre-Push Validation)
**Estimate:** 0.5h

**What:** Run tests locally with PostgreSQL to simulate CI environment.

```bash
# Start PostgreSQL service (simulating CI)
docker run --name test-postgres -d \
  -e POSTGRES_USER=test \
  -e POSTGRES_PASSWORD=test \
  -e POSTGRES_DB=quantagent_test \
  -p 5432:5432 \
  postgres:16

sleep 5  # wait for postgres to be ready

# Run tests (same command as CI)
export DATABASE_URL=postgresql://test:test@localhost:5432/quantagent_test
cd /home/azureuser/repos/projects/QuantAgent
pytest tests/ -v --tb=short --maxfail=10 -m "not integration and not slow"

# Cleanup
docker stop test-postgres && docker rm test-postgres
```

**Expected:** All tests pass, exit code 0. If tests fail, do NOT proceed to Task 6.

---

### Task 6: Commit Changes
**Estimate:** 0.1h

```bash
git add .github/workflows/main-ci-deploy.yml
git commit -m "[QuantAgent-82t] Re-enable unit tests in CI pipeline

- Uncomment 'Run unit tests' step (lines 67-75)
- Update maxfail=5 to maxfail=10
- Remove continue-on-error (tests now block deployment)
- DATABASE_URL verified correct

Closes QuantAgent-82t"
```

---

### Task 7: Monitor First CI Run
**Estimate:** 0.25h

After push, verify in GitHub Actions:
1. "Run unit tests" step executes
2. Tests pass
3. Telegram receives "✅ CI passed"
4. QA deployment proceeds

---

## Total Estimate

**1.35 hours** (7 tasks)

| Phase | Tasks | Time |
|---|---|---|
| Configuration changes | 1-4 | 0.5h |
| Local validation | 5 | 0.5h |
| Deployment + monitoring | 6-7 | 0.35h |

---

## Execution Order

1. Task 1 (uncomment + maxfail + remove continue-on-error)
2. Task 2-3 (verify as part of Task 1)
3. Task 4 (verify DATABASE_URL)
4. Task 5 (local test with Docker PostgreSQL) ⚠️ **Critical — do not skip**
5. Task 6 (commit)
6. Task 7 (monitor CI)

---

## Rollback Plan

### Option 1: Immediate Revert (< 5 min)
```bash
git revert HEAD && git push origin main
```

### Option 2: Re-Comment Test Step (< 2 min)
```bash
# Add # back to test step in .github/workflows/main-ci-deploy.yml
git add .github/workflows/main-ci-deploy.yml
git commit -m "Hotfix: Re-disable tests (investigate issue)"
git push origin main
```

---

## Documentation Artifacts

- `docs/01_requirements/QuantAgent-82t-RQ-ci-test-enablement.md` ✓
- `docs/03_design/QuantAgent-82t-DS-ci-test-enablement.md` ✓
- `docs/05_acceptance_tests/QuantAgent-82t-AC-ci-test-enablement.md` ✓
- `docs/02_planning/QuantAgent-82t-PL-ci-test-enablement.md` (this file) ✓
- `.github/workflows/main-ci-deploy.yml` (pending implementation)
