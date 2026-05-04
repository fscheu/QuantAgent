---
run_id: 20260504T024003Z-techlead-correction
phase: tech_lead
skill: tech-lead-autodev
mode: correction
executor: hermes-internal
repo_path: /tmp/autodev-worktrees/QuantAgent/QuantAgent-app/techlead-correction-20260504T024003Z
beads_issue_id: QuantAgent-app
branch: feature/QuantAgent-app-fix-qa-deploy-notify-metadata
capabilities:
  read_repo: true
  write_docs: true
  write_code: true
  write_tests: false
  beads_read: true
  beads_comment_final: true
  git_create_branch: true
  git_commit: true
  git_push: false
  merge_main: false
  deploy: false
forbidden_actions:
  - read or print .env/secrets/tokens
  - force push
  - reset --hard
  - modify unrelated workflows
---

## Objective
Fix the QA deploy success notification so the deploy job can resolve the commit message without logging `fatal: not a git repository`.

## Scope In
- `.github/workflows/main-ci-deploy.yml`
- minimal implementation note and run artifacts

## Scope Out
- deploy infrastructure changes
- secrets/config changes
- user-facing docs
- unrelated CI workflow cleanup

## Source of Truth
- `AGENTS.md`
- `CLAUDE.md`
- Beads issue `QuantAgent-app`
- `.github/workflows/main-ci-deploy.yml`
- Failure taxonomy note

## Acceptance Criteria
- `Notify Telegram on deploy success` no longer depends on a missing checkout
- QA deploy Telegram message can resolve commit metadata from the checked out repo
- Workflow structure remains valid

## Correction Rationale
This qualifies as Tech Lead correction mode because it is a tiny workflow/config fix with an obvious minimal patch and no product/design ambiguity.
