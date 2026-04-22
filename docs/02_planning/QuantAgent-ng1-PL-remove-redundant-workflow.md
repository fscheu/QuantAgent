# QuantAgent-ng1 — Planning: Remove Redundant Workflow

**Issue ID:** QuantAgent-ng1  
**Title:** Remove redundant main-ci.yml workflow  
**Type:** Task  
**Priority:** 3

---

## Objective

Delete `.github/workflows/main-ci.yml` to eliminate duplicate CI runs and duplicate Telegram notifications.

---

## Tasks

### Task 1: Delete Workflow File
**Estimate:** 0.05h (3 minutes)

**What:**
- Execute: `git rm .github/workflows/main-ci.yml`
- This removes the file from the repository and stages the deletion

**Why:**
- `main-ci.yml` is redundant (duplicates `main-ci-deploy.yml` CI logic)
- Causes 2 workflow runs per push to main
- Causes 2 Telegram notifications per push

**How to validate:**
```bash
# Verify file is deleted and staged
git status
# Expected: "deleted: .github/workflows/main-ci.yml"

# Verify file doesn't exist
ls .github/workflows/main-ci.yml
# Expected: No such file or directory
```

**Dependencies:** None

---

### Task 2: Commit Deletion
**Estimate:** 0.05h (3 minutes)

**What:**
- Commit the deletion with descriptive message:
  ```bash
  git commit -m "[QuantAgent-ng1] Remove redundant main-ci.yml workflow
  
  - Eliminates duplicate CI runs on push to main
  - Reduces Telegram notifications from 2 to 1 per push
  - main-ci-deploy.yml provides all needed functionality
  
  Closes QuantAgent-ng1"
  ```

**Why:**
- Clear commit message explains the change
- References issue ID for traceability
- Documents the reason (duplicate elimination)

**How to validate:**
```bash
# Verify commit created
git log -1 --oneline
# Expected: Shows commit with "QuantAgent-ng1" in message

# Verify file is in commit
git show --name-only
# Expected: Shows .github/workflows/main-ci.yml as deleted
```

**Dependencies:** Task 1

---

### Task 3: Push to Feature Branch (or Main)
**Estimate:** 0.05h (3 minutes)

**What:**
- Push commit to remote:
  ```bash
  # If using feature branch (recommended):
  git push origin feature/QuantAgent-ng1-remove-redundant-workflow
  
  # Or directly to main (if allowed):
  git push origin main
  ```

**Why:**
- Make change available for testing
- Trigger remaining workflow to verify it still works

**How to validate:**
```bash
# Verify push succeeded
git log origin/main --oneline -1
# (or check feature branch)

# Check remote doesn't have the file
git ls-remote --heads origin | grep ng1
# (if using feature branch, should show branch)
```

**Dependencies:** Task 2

---

### Task 4: Verify Single Workflow Run
**Estimate:** 0.25h (15 minutes)

**What:**
- After pushing, monitor GitHub Actions
- Navigate to: https://github.com/[repo]/actions
- Verify only ONE workflow run appears for the commit
- Workflow name should be: "Main CI + Deploy QA"

**Why:**
- Confirm deletion achieved the goal
- Ensure no workflows are missing

**Expected results:**
- Before fix: 2 runs per push
- After fix: 1 run per push

**How to validate:**
1. Open GitHub Actions tab
2. Find commit with "[QuantAgent-ng1]" in message
3. Count workflow runs
4. **Expected:** 1 run (not 2)
5. **Expected name:** "Main CI + Deploy QA"

**Dependencies:** Task 3

---

### Task 5: Verify Single Telegram Notification
**Estimate:** 0.1h (6 minutes)

**What:**
- Wait for workflow to complete
- Check Telegram channel (ID: -1003835130753, thread 208)
- Count notifications for the commit
- Verify only ONE notification received

**Why:**
- Confirm Telegram duplication is resolved
- User experience improvement verification

**Expected results:**
- Before fix: 2 messages per push
- After fix: 1 message per push

**How to validate:**
1. Open Telegram channel
2. Find messages with commit hash from Task 2
3. Count messages
4. **Expected:** 1 message (✅ CI passed or 🚨 CI Failure)

**Dependencies:** Task 4 (workflow must complete)

---

### Task 6: Update Beads Status
**Estimate:** 0.1h (6 minutes)

**What:**
- Add comment to Beads issue with results
- Update status to `test_done` or `merged`

**Why:**
- Track completion in task system
- Document verification results

**Commands:**
```bash
# Add comment
bd comments add QuantAgent-ng1 -m "Completed: main-ci.yml deleted, verified single workflow run and single Telegram notification"

# Update status (if using Beads workflow)
# (or handle via autodev pipeline)
```

**Dependencies:** Task 5

---

## Total Estimate

**0.6 hours** (36 minutes)

**Breakdown:**
- File deletion and commit: 0.15h (Tasks 1-3)
- Verification: 0.35h (Tasks 4-5)
- Documentation: 0.1h (Task 6)

**Note:** Extremely simple task, most time is waiting for workflows and verification.

---

## Execution Order

1. **Task 1** (delete file) — 3 minutes
2. **Task 2** (commit) — 3 minutes
3. **Task 3** (push) — 3 minutes
4. **Task 4** (verify workflow) — 15 minutes (includes wait time)
5. **Task 5** (verify Telegram) — 6 minutes
6. **Task 6** (update Beads) — 6 minutes

**Total elapsed:** ~36 minutes (mostly waiting)

---

## Risks & Mitigations

### Risk 1: Wrong File Deleted

**Description:** Accidentally delete `main-ci-deploy.yml` instead

**Mitigation:**
- Double-check filename before `git rm`
- Use tab completion: `git rm .github/workflows/main-ci<TAB>`
- Review `git status` before commit
- If wrong file deleted: `git reset HEAD <file>` to unstage

**Probability:** Very Low  
**Impact:** Medium (easy to fix before push)

---

### Risk 2: Remaining Workflow Broken

**Description:** `main-ci-deploy.yml` has issues after deletion

**Mitigation:**
- The two workflows are independent (no shared resources)
- Deletion cannot break the remaining workflow
- Task 4 verifies remaining workflow runs successfully

**Probability:** Zero  
**Impact:** N/A

---

## Testing Strategy

### Pre-Merge Testing
- Task 4: Verify single workflow run
- Task 5: Verify single Telegram notification

### Post-Merge Validation
- Monitor next 2-3 pushes to main
- Confirm consistent behavior (1 run, 1 message)
- Check GitHub Actions usage (should decrease)

---

## Rollback Plan

If issues arise (extremely unlikely):

### Option 1: Revert Commit
```bash
git revert <commit-hash>
git push origin main
# Restores main-ci.yml
```

### Option 2: Restore from Previous Commit
```bash
git checkout <previous-commit> -- .github/workflows/main-ci.yml
git add .github/workflows/main-ci.yml
git commit -m "Restore main-ci.yml (rollback)"
git push origin main
```

---

## Success Criteria

- [ ] `main-ci.yml` deleted and committed
- [ ] Push to remote successful
- [ ] Single workflow run verified
- [ ] Single Telegram notification verified
- [ ] Beads status updated

---

## Next Steps After This Issue

1. **Monitor:** Watch next few pushes to confirm consistent behavior
2. **Cleanup:** Archive or close any related issues about duplicate notifications
3. **Document:** Update internal docs if they reference both workflows

---

## Documentation Files

**Created:**
- `docs/01_requirements/QuantAgent-ng1-RQ-remove-redundant-workflow.md` ✓
- `docs/03_design/QuantAgent-ng1-DS-remove-redundant-workflow.md` ✓
- `docs/05_acceptance_tests/QuantAgent-ng1-AC-remove-redundant-workflow.md` ✓
- `docs/02_planning/QuantAgent-ng1-PL-remove-redundant-workflow.md` (this file) ✓

**To Delete:**
- `.github/workflows/main-ci.yml` (pending implementation)

---

## Final Checklist

Before implementation:
- [ ] Confirm current directory: `/home/azureuser/repos/projects/QuantAgent/.worktrees/feature__QuantAgent-ng1-remove-redundant-main-ci-yml-workflow`
- [ ] Verify two workflows currently exist
- [ ] Have access to GitHub Actions and Telegram for verification

During implementation:
- [ ] Execute Task 1: `git rm .github/workflows/main-ci.yml`
- [ ] Execute Task 2: Commit with message
- [ ] Execute Task 3: Push to remote
- [ ] Execute Task 4: Verify single workflow run
- [ ] Execute Task 5: Verify single Telegram notification
- [ ] Execute Task 6: Update Beads

After implementation:
- [ ] Document results in Beads comment
- [ ] Monitor next 2-3 pushes for consistency
- [ ] Close issue
