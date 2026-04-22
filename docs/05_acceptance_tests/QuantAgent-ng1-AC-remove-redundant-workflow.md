# QuantAgent-ng1 — Acceptance Criteria: Remove Redundant Workflow

**Issue ID:** QuantAgent-ng1  
**Title:** Remove redundant main-ci.yml workflow  
**Type:** Task

---

## AC-1: Workflow File Removed

**Given** the repository after implementation  
**When** checking the `.github/workflows/` directory  
**Then** `main-ci.yml` does NOT exist

**Verification:**
```bash
# Check file does not exist in filesystem
ls .github/workflows/main-ci.yml
# Expected: No such file or directory

# Check file is not tracked by git
git ls-files .github/workflows/main-ci.yml
# Expected: (no output)

# Verify deletion in git history
git log --oneline --follow -- .github/workflows/main-ci.yml | head -1
# Expected: Shows commit that deleted the file
```

---

## AC-2: Single Workflow Run Per Push

**Given** a commit is pushed to `main` branch  
**When** viewing GitHub Actions runs  
**Then** exactly ONE workflow run is triggered

**Setup:**
```bash
# Create test commit
echo "# Test commit for QuantAgent-ng1" >> README.md
git add README.md
git commit -m "Test: Verify single workflow run"
git push origin main
```

**Verification:**
1. Navigate to: `https://github.com/[org]/[repo]/actions`
2. Find the commit with message "Test: Verify single workflow run"
3. Count workflow runs for that commit
4. **Expected:** 1 run
5. **Expected workflow name:** "Main CI + Deploy QA"

**Before fix:**
- 2 workflow runs:
  - "Main CI + Notifications"
  - "Main CI + Deploy QA"

**After fix:**
- 1 workflow run:
  - "Main CI + Deploy QA"

**Visual verification:**

Before:
```
Main CI + Notifications        ✓ (or ✗)
Main CI + Deploy QA            ✓ (or ✗)
```

After:
```
Main CI + Deploy QA            ✓ (or ✗)
```

---

## AC-3: Single Telegram Notification Per Push

**Given** a commit is pushed to `main` branch  
**When** workflows complete  
**Then** exactly ONE Telegram notification is sent

**Setup:**
Same test commit as AC-2

**Verification:**
1. Wait for workflow to complete
2. Check Telegram channel (ID: -1003835130753, thread: 208)
3. Count messages for the test commit
4. **Expected:** 1 message

**Before fix:**
- 2 messages:
  - "✅ CI passed" from main-ci.yml
  - "✅ CI passed" from main-ci-deploy.yml
  - (Plus deployment success message)

**After fix:**
- 1 message:
  - "✅ CI passed" from main-ci-deploy.yml
  - (Plus deployment success message if CI passes)

**Message format (expected):**
```
✅ CI passed

Commit: abc1234
Message: Test: Verify single workflow run
```

---

## Edge Cases

### EC-1: Workflow File History Preserved

**Given** the file was deleted  
**When** checking git history  
**Then** previous commits with `main-ci.yml` are still accessible

**Verification:**
```bash
# Show file content from previous commit
git show HEAD~1:.github/workflows/main-ci.yml
# Expected: Shows full file content (before deletion)

# Show file history
git log --oneline --follow -- .github/workflows/main-ci.yml
# Expected: Shows all commits that touched the file
```

---

### EC-2: Other Workflows Unaffected

**Given** the repository has other workflow files  
**When** `main-ci.yml` is deleted  
**Then** other workflows continue to function

**Verification:**
```bash
# List remaining workflows
ls .github/workflows/
# Expected: 
#   deploy.yml
#   main-ci-deploy.yml
#   (NO main-ci.yml)

# Verify deploy.yml still works (triggers on push to gh-pages)
# Verify main-ci-deploy.yml still works (tested in AC-2)
```

---

### EC-3: Manual Workflow Dispatch Still Works

**Given** `main-ci-deploy.yml` has `workflow_dispatch` trigger  
**When** manually triggering the workflow from GitHub UI  
**Then** workflow runs successfully

**Verification:**
1. Navigate to: Actions → "Main CI + Deploy QA" → Run workflow
2. Click "Run workflow" button
3. Select branch: main
4. Click green "Run workflow" button
5. **Expected:** Workflow runs successfully

---

## Negative Test Cases

### NT-1: Deleted Workflow Does Not Trigger

**Given** `main-ci.yml` is deleted  
**When** pushing to main  
**Then** "Main CI + Notifications" workflow does NOT run

**Verification:**
- Push test commit
- Check Actions tab
- **Expected:** No workflow run with name "Main CI + Notifications"

---

### NT-2: No Extra Notifications

**Given** only `main-ci-deploy.yml` remains  
**When** CI fails  
**Then** exactly ONE failure notification is sent (not two)

**Setup:**
```bash
# Introduce lint error (temporarily)
echo "import os,sys" >> quantagent/models.py  # Bad: multiple imports on one line
git add quantagent/models.py
git commit -m "Test: Verify single failure notification"
git push origin main
```

**Verification:**
1. Wait for workflow to fail (lint error)
2. Check Telegram
3. **Expected:** 1 failure message:
   ```
   🚨 CI Failure
   
   Commit: xyz7890
   Message: Test: Verify single failure notification
   Failed step: Lint
   ```

**Cleanup:**
```bash
git revert HEAD
git push origin main
```

---

## Performance Verification

### PV-1: GitHub Actions Minutes Saved

**Given** the redundant workflow is removed  
**When** measuring workflow run times  
**Then** approximately 50% reduction in Actions minutes for main branch

**Measurement:**
1. Before: Check Actions usage for 1 week prior
2. After: Check Actions usage for 1 week after
3. Compare: main branch workflow minutes

**Expected reduction:**
- If each workflow takes ~5 minutes
- Before: 10 minutes per push (2 workflows × 5 min)
- After: 5 minutes per push (1 workflow × 5 min)
- Savings: 50%

---

### PV-2: Faster Notification Delivery

**Given** only one workflow runs  
**When** pushing to main  
**Then** notifications arrive faster (no queueing)

**Before fix:**
- Both workflows queue simultaneously
- Both send notifications (can arrive out of order)

**After fix:**
- Single workflow runs
- Single notification (predictable timing)

---

## Manual Test Procedure

### Prerequisites
1. Ensure repository has both workflows currently
2. Have access to GitHub Actions tab
3. Have access to Telegram channel

### Test Steps

#### Step 1: Verify Current State (2 workflows)
```bash
# List workflows
ls -l .github/workflows/
# Expected: main-ci.yml, main-ci-deploy.yml, deploy.yml

# Push test commit
echo "# Before fix" >> test.md
git add test.md
git commit -m "Before fix: Test 2 workflows"
git push origin main

# Check Actions: Should see 2 workflow runs
# Check Telegram: Should see 2 messages
```

#### Step 2: Delete Workflow
```bash
# Delete file
git rm .github/workflows/main-ci.yml

# Commit
git commit -m "[QuantAgent-ng1] Remove redundant main-ci.yml workflow"

# Push
git push origin main
```

#### Step 3: Verify New State (1 workflow)
```bash
# List workflows
ls -l .github/workflows/
# Expected: main-ci-deploy.yml, deploy.yml (NO main-ci.yml)

# Push test commit
echo "# After fix" >> test.md
git add test.md
git commit -m "After fix: Test 1 workflow"
git push origin main

# Check Actions: Should see 1 workflow run
# Check Telegram: Should see 1 message
```

#### Step 4: Cleanup
```bash
# Remove test file
git rm test.md
git commit -m "Cleanup: Remove test file"
git push origin main
```

---

## Definition of Done (Testing Checklist)

- [ ] AC-1: `main-ci.yml` file does not exist
- [ ] AC-2: Only 1 workflow run per push to main
- [ ] AC-3: Only 1 Telegram message per push (CI notification)
- [ ] EC-1: Git history preserved (can view old file)
- [ ] EC-2: Other workflows unaffected
- [ ] EC-3: Manual dispatch works
- [ ] NT-1: Deleted workflow does not trigger
- [ ] NT-2: Single failure notification (if CI fails)
- [ ] PV-1: Actions minutes reduced
- [ ] Manual test procedure completed successfully
