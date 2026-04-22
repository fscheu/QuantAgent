# QuantAgent-ng1 — Design: Remove Redundant CI Workflow

**Issue ID:** QuantAgent-ng1  
**Title:** Remove redundant main-ci.yml workflow  
**Type:** Task

---

## Design Overview

Delete `.github/workflows/main-ci.yml` to eliminate redundant CI runs. The remaining `main-ci-deploy.yml` provides all needed functionality.

---

## Affected Components

### Deleted
- `.github/workflows/main-ci.yml` — Redundant CI-only workflow

### Not Modified
- `.github/workflows/main-ci-deploy.yml` — Comprehensive CI + deploy workflow (kept)
- `.github/workflows/deploy.yml` — GitHub Pages deployment (different trigger)

---

## Technical Analysis

### Current State (Problem)

**Two workflows trigger on `push: branches: [main]`:**

1. **`main-ci-deploy.yml`:**
   - Runs: lint, tests (disabled but configured), status checks
   - Sends: Telegram notifications (success/failure)
   - Then: Deploys to QA if CI passes
   - Sends: Telegram notifications for deploy (success/failure)

2. **`main-ci.yml`:**
   - Runs: lint, tests (disabled but configured), status checks
   - Sends: Telegram notifications (success/failure)
   - **Ends** (no deployment)

**Result:**
- Both workflows run in parallel on every push
- Both send Telegram notifications
- User receives 2 messages per push (one from each workflow)

### Comparison of Workflows

| Feature | main-ci.yml | main-ci-deploy.yml |
|---------|-------------|---------------------|
| Trigger | push: main | push: main |
| Python setup | ✓ | ✓ |
| TA-Lib install | ✓ | ✓ |
| Dependencies | ✓ | ✓ |
| Lint (ruff) | ✓ | ✓ |
| Tests | disabled | disabled |
| Telegram notify | ✓ | ✓ |
| Deploy to QA | ✗ | ✓ |
| Deploy notify | ✗ | ✓ |

**Conclusion:** `main-ci.yml` is a strict subset of `main-ci-deploy.yml` — it provides no unique value.

---

## Design Decision

### Chosen Approach: Delete Redundant Workflow

**Why delete `main-ci.yml` (not `main-ci-deploy.yml`):**
- `main-ci-deploy.yml` is more complete (includes deployment)
- `main-ci-deploy.yml` is the intended final workflow
- Deleting the simpler one is less risky

**Benefits:**
1. ✅ Eliminate duplicate runs (saves GitHub Actions minutes)
2. ✅ Eliminate duplicate Telegram notifications (cleaner communication)
3. ✅ Single source of truth for CI pipeline
4. ✅ Simpler maintenance (one workflow to manage)

**Risks:**
- ✅ **None** — `main-ci-deploy.yml` provides all functionality of `main-ci.yml` plus deployment

---

## Alternative Approaches Considered

### ❌ Approach 1: Keep Both, Add Different Triggers

**Idea:** Make `main-ci.yml` trigger on PRs, `main-ci-deploy.yml` on main

**Why rejected:**
- Would require workflow modifications (out of scope)
- No benefit — PRs should use `main-ci-deploy.yml` CI logic too
- Adds complexity instead of reducing it

---

### ❌ Approach 2: Delete `main-ci-deploy.yml` Instead

**Idea:** Keep simpler `main-ci.yml`, add deployment to it

**Why rejected:**
- More work (modify workflow, not just delete)
- `main-ci-deploy.yml` is already correct and complete
- Higher risk of breaking deployment

---

### ❌ Approach 3: Merge Workflows into One File

**Idea:** Consolidate both into a single new workflow

**Why rejected:**
- `main-ci-deploy.yml` already does everything needed
- Renaming/merging is unnecessary complexity
- No functional benefit

---

## Implementation

### Step 1: Delete File
```bash
git rm .github/workflows/main-ci.yml
```

### Step 2: Commit
```bash
git commit -m "[QuantAgent-ng1] Remove redundant main-ci.yml workflow

- Eliminates duplicate CI runs on push to main
- Reduces Telegram notifications from 2 to 1 per push
- main-ci-deploy.yml provides all needed functionality

Closes QuantAgent-ng1"
```

### Step 3: Push and Verify
```bash
git push origin main
# OR push to feature branch for PR
```

---

## Verification Plan

### Before Merge (Preview)
```bash
# Check which workflows would trigger
git diff --name-only HEAD origin/main | grep workflows
# Should show: deleted .github/workflows/main-ci.yml
```

### After Merge
1. **Push test commit to main**
2. **Check GitHub Actions:**
   - Navigate to Actions tab
   - Verify only ONE workflow run appears
   - Workflow name: "Main CI + Deploy QA"
3. **Check Telegram:**
   - Verify only ONE notification received
   - Message should be from main-ci-deploy.yml

---

## Rollback Plan

If issues arise (unlikely):

### Option 1: Revert Commit
```bash
git revert <commit-hash>
git push origin main
# Restores main-ci.yml
```

### Option 2: Restore File
```bash
git checkout <previous-commit> -- .github/workflows/main-ci.yml
git commit -m "Restore main-ci.yml (temporary)"
git push origin main
```

---

## Success Metrics

### Immediate (Day 1)
- [ ] Only 1 workflow run per push to main
- [ ] Only 1 Telegram message per push

### Short-term (Week 1)
- [ ] GitHub Actions usage reduced by ~50% for main branch
- [ ] Cleaner Telegram notification history

---

## Open Questions

None — this is a straightforward deletion with no ambiguity.
