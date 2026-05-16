---
run_id: "20260511T073839Z-QuantAgent-375-planner"
phase: "planner"
executor: "auto"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-375"
branch_policy: "worktree-preferred"
base_branch: "main"
current_branch_at_generation: "main"
feature_branch: "feature/QuantAgent-375-scope-replay-signal-lookup-to-selected-s"
worktree_root: "/mnt/actions-runner/autodev-runtime/worktrees/QuantAgent"
worktree_path: "/mnt/actions-runner/autodev-runtime/worktrees/QuantAgent/QuantAgent-375/planner-20260511T073839Z"
shared_venv: "/mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv"
shared_python: "/mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python"
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
  - "change Beads labels/status unless explicitly enabled"
  - "stash, discard, reset --hard, or rewrite history"
quality_gates:
  required:
    - "git status --short"
    - "verify issue ID appears in docs paths"
    - "verify acceptance criteria are testable"
  optional:
    - "python -m compileall -q ."
budget:
  max_turns: 10
  max_minutes: 45
  max_cost_usd: null
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-375/20260511T073839Z-QuantAgent-375-planner"
generated_at: "2026-05-11T07:38:39.693957+00:00"
preflight:
  repo_dirty: true
  dirty_files:
    - " M .beads/issues.jsonl"
    - "?? docs/envelopes/QuantAgent-3o8/"
    - "?? docs/envelopes/QuantAgent-4fm/"
    - "?? docs/envelopes/QuantAgent-l8r/20260508T124108Z-QuantAgent-l8r-planner/"
---

# Autodev Input Envelope — QuantAgent-375 — planner

## Objective

Execute the `planner` phase for Beads issue `QuantAgent-375`: Scope replay signal lookup to selected source run.

## Scope In

- Follow the `autodev-planner` skill for this phase.
- Use the Beads issue, repo instructions, docs, and recent comments as source of truth.
- Preserve migration defaults: no push, no merge, no deploy, no external messages.

## Scope Out

- Do not modify credentials, `.env`, secrets, production service config, or systemd/OpenClaw/Hermes runtime config.
- Do not rename legacy `openclaw:*` labels during this phase.
- Do not work on unrelated issues or opportunistic refactors.
- Do not merge to `main`.

## Source of Truth

- Repo instructions: `/home/azureuser/repos/projects/QuantAgent/AGENTS.md` and `/home/azureuser/repos/projects/QuantAgent/CLAUDE.md` if present.
- Beads issue: `QuantAgent-375`.
- Labels at generation time: `none`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-375/20260511T073839Z-QuantAgent-375-planner/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

Integration review of QuantAgent-3o8 found that `Backtest.run_replay()` loads signals only by symbol/timeframe/date/environment and does not reliably scope them to the selected source BacktestRun.

Evidence:
- Feature branch: `feature/QuantAgent-3o8-implement-replay-execution-mode-reuse-an`
- Replay proof run on 2026-05-10 showed source run 1 consuming signal ID 2 from a different overlapping run because `signal_map[(symbol, generated_at)]` was overwritten.
- Acceptance criteria TC5/TC11 require the replay to use the same source-run signals and keep the provenance chain traceable.

Required change:
1. Add or derive a robust signal-to-source-run linkage (explicit `backtest_run_id` on `Signal` or another deterministic provenance key already persisted).
2. Update replay signal loading to query only the selected source run.
3. Add tests covering overlapping runs with identical symbol/timeframe/timestamp to prevent cross-run signal contamination.
4. Revalidate replay provenance and no-LLM behavior before re-routing integration.

Out of scope:
- New replay UI features
- Parallel replay execution
- Broader backtest refactors

## Acceptance Criteria

No acceptance criteria found in Beads issue.

## Recent Beads Comments

- No recent comments captured.

## Preflight Evidence

- Current branch at generation: `main`.
- Repo dirty at generation: `True`.
- Dirty files:
  - ` M .beads/issues.jsonl`
  - `?? docs/envelopes/QuantAgent-3o8/`
  - `?? docs/envelopes/QuantAgent-4fm/`
  - `?? docs/envelopes/QuantAgent-l8r/20260508T124108Z-QuantAgent-l8r-planner/`

## Executor Instructions

- Read this envelope completely before acting.
- Respect the `capabilities` block in the YAML header.
- If a required capability is false, do not perform that action even if prose could be interpreted otherwise.
- If requirements are ambiguous, stop with `BLOCKED` and explain the minimum missing decision.
- If this is a write phase, use the declared feature branch/worktree policy unless explicitly overridden by Hermes/human.
- For Python repos, use the declared `shared_python` interpreter for tests/tooling when possible. In QuantAgent, worktrees should reuse the shared venv instead of creating a per-worktree `.venv`.
- End with an output envelope/report containing: summary, files changed, commands run, quality gates, artifacts, risks, next step, and Beads update.

## Required Final Beads Comment Template

```md
### Skill: autodev-planner
- **Resultado:** SUCCESS | PARTIAL | BLOCKED | FAIL
- **Run-ID:** 20260511T073839Z-QuantAgent-375-planner
- **Executor:** auto

#### Qué hice
- ...

#### Evidencia
- ...

#### Quality gates
- ...

#### Problemas encontrados
- ...

#### Next step recomendado
- ...
```
