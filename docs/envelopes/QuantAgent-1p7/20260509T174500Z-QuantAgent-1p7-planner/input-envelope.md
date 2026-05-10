---
run_id: "20260509T174500Z-QuantAgent-1p7-planner"
phase: "planner"
executor: "auto"
repo_path: "/tmp/autodev-worktrees/QuantAgent/QuantAgent-1p7/planner-20260509T174006Z"
beads_issue_id: "QuantAgent-1p7"
branch_policy: "worktree-preferred"
base_branch: "main"
current_branch_at_generation: "feature/QuantAgent-1p7-save-stategraph-images-to-disk"
feature_branch: "feature/QuantAgent-1p7-save-stategraph-images-to-disk"
worktree_path: "/tmp/autodev-worktrees/QuantAgent/QuantAgent-1p7/planner-20260509T174006Z"
skill: "autodev-planner"
mode: "write-docs"
capabilities:
  read_repo: true
  write_docs: true
  write_code: false
  write_tests: false
  beads_read: true
  beads_comment_final: true
  beads_update_labels: false
  git_create_branch: false
  git_commit: false
  git_push: false
  merge_main: false
  deploy: false
  send_external_message: false
  touch_secrets: false
forbidden_actions:
  - "git push"
  - "git merge main"
  - "deploy commands"
  - "read or print .env/secrets/tokens"
  - "send Telegram/email/Slack"
  - "stash, discard, reset --hard, or rewrite history"
quality_gates:
  required:
    - "git status --short"
    - "verify issue ID appears in docs paths"
    - "verify acceptance criteria are testable"
  optional:
    - "python -m compileall -q quantagent tests"
artifacts_dir: "/tmp/autodev-worktrees/QuantAgent/QuantAgent-1p7/planner-20260509T174006Z/docs/envelopes/QuantAgent-1p7/20260509T174500Z-QuantAgent-1p7-planner"
generated_at: "2026-05-09T17:45:00Z"
preflight:
  repo_dirty: true
  dirty_files:
    - "docs/01_requirements/README.md"
    - "docs/02_planning/README.md"
    - "docs/03_design/README.md"
    - "docs/05_acceptance_tests/README.md"
    - "docs/01_requirements/QuantAgent-1p7-RQ-stategraph-image-paths.md"
    - "docs/02_planning/QuantAgent-1p7-PL-stategraph-image-paths.md"
    - "docs/03_design/QuantAgent-1p7-DS-stategraph-image-paths.md"
    - "docs/05_acceptance_tests/QuantAgent-1p7-AC-stategraph-image-paths.md"
---

# Autodev Input Envelope — QuantAgent-1p7 — planner

## Objective
Execute the `planner` phase for Beads issue `QuantAgent-1p7`: Save StateGraph images to disk and reference file paths.

## Scope In
- Produce issue-scoped planning docs under `docs/`.
- Keep the solution aligned with the repo-wide `path-only` artifact policy.
- Define the minimal implementation path for disk-backed StateGraph image export.

## Scope Out
- No production code changes.
- No tests changes.
- No merge/push/deploy.
- No changes to secrets, runtime config, or unrelated artifact flows.

## Source of Truth
- Repo instructions: `AGENTS.md`, `CLAUDE.md`
- Beads issue: `QuantAgent-1p7`
- Relevant docs:
  - `docs/01_requirements/ui_streamlit_mvp_requirements.md`
  - `docs/03_design/streamlit_app_architecture.md`
  - `quantagent/graph_setup.py`
  - `quantagent/trading_graph.py`

## Issue Description
Refactor graph visualization to save generated StateGraph images to disk instead of keeping them in memory. Update references to use file paths. This improves debugging and allows sharing graph visualizations outside runtime.

## Acceptance Intent
- Disk-backed export for StateGraph visualization
- Path-only references instead of in-memory image payloads
- Minimal, testable implementation plan for the next phase

## Recent Beads Comments
- No recent comments captured.

## Executor Instructions
- Read this envelope completely before acting.
- Respect the capability block in the YAML header.
- Produce minimal, actionable docs for implementer/tester handoff.
- End with durable artifacts and a final Beads comment.
