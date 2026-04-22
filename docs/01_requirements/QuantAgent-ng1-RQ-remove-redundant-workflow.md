# QuantAgent-ng1 — Requirements: Remove Redundant CI Workflow

**Issue ID:** QuantAgent-ng1  
**Title:** Remove redundant main-ci.yml workflow  
**Type:** Task  
**Priority:** 3  
**Labels:** openclaw:design_approved ci

---

## Objective

Remove the redundant `main-ci.yml` workflow to eliminate duplicate CI runs and duplicate Telegram notifications on every push to main.

---

## Background

Currently, two GitHub Actions workflows trigger on every push to `main`:
1. **`main-ci-deploy.yml`** — Runs CI (lint + tests) + deploys to QA + sends Telegram notifications
2. **`main-ci.yml`** — Runs CI (lint + tests) + sends Telegram notifications

The second workflow is completely redundant — it duplicates the CI portion of the first workflow, causing:
- Two parallel workflow runs per commit
- Two Telegram notification messages per commit (one from each workflow)
- Wasted GitHub Actions minutes

**Root cause:** Both workflows listen to the same trigger (`on: push: branches: [main]`)

---

## Scope

### In Scope
- Delete `.github/workflows/main-ci.yml`

### Out of Scope
- Modifying `.github/workflows/main-ci-deploy.yml` (keep as-is)
- Modifying `.github/workflows/deploy.yml` (GitHub Pages, different trigger)
- Changing Telegram notification logic
- Modifying CI steps or test configuration

---

## Requirements

### FR-1: Remove Redundant Workflow File
**Description:** Delete the `main-ci.yml` workflow file from the repository

**Action:**
```bash
git rm .github/workflows/main-ci.yml
```

**Verification:**
```bash
ls .github/workflows/
# Should NOT show main-ci.yml
```

---

## Acceptance Criteria

### AC-1: Workflow File Removed
**Given** the repository after merge  
**When** checking `.github/workflows/` directory  
**Then** `main-ci.yml` does NOT exist

**Verification:**
```bash
git ls-files .github/workflows/main-ci.yml
# Expected: (no output - file not tracked)
```

### AC-2: Single Workflow Run Per Push
**Given** a commit is pushed to `main` branch  
**When** viewing GitHub Actions tab  
**Then** exactly ONE workflow run is triggered (main-ci-deploy.yml)

**Verification:**
- Navigate to: https://github.com/[repo]/actions
- Push a test commit to main
- Observe: Only "Main CI + Deploy QA" workflow runs (not "Main CI + Notifications")

### AC-3: Single Telegram Notification Per Push
**Given** a commit is pushed to `main` branch  
**When** workflow completes  
**Then** exactly ONE Telegram notification is sent (from main-ci-deploy.yml)

**Before fix:** 2 messages per push  
**After fix:** 1 message per push

**Verification:**
- Push test commit to main
- Check Telegram channel
- Count messages: should be 1 (not 2)

---

## Constraints

- **No other file changes:** Only delete `main-ci.yml`, don't modify anything else
- **Preserve existing functionality:** CI + deploy + notifications continue to work via `main-ci-deploy.yml`
- **No workflow logic changes:** Don't alter the remaining workflow

---

## Definition of Done

- [ ] `main-ci.yml` deleted from repository
- [ ] Test push to main triggers only 1 workflow
- [ ] Test push to main sends only 1 Telegram message
- [ ] Documentation updated (this file)
