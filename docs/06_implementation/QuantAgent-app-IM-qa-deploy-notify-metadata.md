# QuantAgent-app Implementation Notes

**Issue:** QuantAgent-app — Fix QA deploy success notification commit metadata  
**Mode:** Tech Lead correction  
**Branch:** feature/QuantAgent-app-fix-qa-deploy-notify-metadata

## Summary

Fixed the QA deploy workflow so the deploy job has a repository checkout before the success notification step reads commit metadata with `git log`.

## Change made

- Added `actions/checkout@v4` as the first step in `jobs.deploy-qa.steps` of `.github/workflows/main-ci-deploy.yml`.

## Why this fixes the bug

The deploy job previously ran `git log -1 --pretty=%B` without any checkout in that job, so GitHub Actions had no `.git` directory and logged `fatal: not a git repository`. With a checkout present, the step can resolve the commit message for the pushed SHA.

## Scope

In scope:
- QA deploy job checkout for commit metadata access

Out of scope:
- Refactoring Telegram notification formatting
- Changing deploy logic or SSH steps
- Changing CI job notifications

## Verification

- Parsed `.github/workflows/main-ci-deploy.yml` with Python/YAML and asserted:
  - `deploy-qa.steps[0]` is `Checkout code`
  - the checkout uses `actions/checkout@v4`
  - the deploy success notification step still exists
- Ran `git diff --check`

## User manual impact

None. Internal CI/deploy workflow correction only.
