---
run_id: "20260507T174143Z-QuantAgent-uz9-implementer"
phase: "implementer"
executor: "auto"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-uz9"
branch_policy: "worktree-preferred"
base_branch: "main"
current_branch_at_generation: "main"
feature_branch: "feature/QuantAgent-uz9-fix-qa-deploy-build-context-bloat-from-w"
worktree_root: "/mnt/actions-runner/autodev-runtime/worktrees/QuantAgent"
worktree_path: "/mnt/actions-runner/autodev-runtime/worktrees/QuantAgent/QuantAgent-uz9/implementer-20260507T174143Z"
shared_venv: "/mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv"
shared_python: "/mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python"
skill: "autodev-implementer"
mode: "write"
capabilities:
  read_repo: true
  write_docs: true
  write_code: true
  write_tests: false
  beads_read: true
  beads_comment_final: true
  beads_update_labels: false
  git_create_branch: true
  git_commit: true
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
    - "ruff check --fix ."
    - "pytest <relevant subset> -v"
    - "python -m compileall -q ."
  optional:
    - "python -m compileall -q ."
budget:
  max_turns: 10
  max_minutes: 45
  max_cost_usd: null
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-uz9/20260507T174143Z-QuantAgent-uz9-implementer"
generated_at: "2026-05-07T17:41:43.157296+00:00"
preflight:
  repo_dirty: true
  dirty_files:
    - " M .beads/issues.jsonl"
    - "?? docs/envelopes/deploy-verification-20260507T132230Z.md"
---

# Autodev Input Envelope — QuantAgent-uz9 — implementer

## Objective

Execute the `implementer` phase for Beads issue `QuantAgent-uz9`: Fix QA deploy build-context bloat from .worktrees.

## Scope In

- Follow the `autodev-implementer` skill for this phase.
- Use the Beads issue, repo instructions, docs, and recent comments as source of truth.
- Preserve migration defaults: no push, no merge, no deploy, no external messages.

## Scope Out

- Do not modify credentials, `.env`, secrets, production service config, or systemd/OpenClaw/Hermes runtime config.
- Do not rename legacy `openclaw:*` labels during this phase.
- Do not work on unrelated issues or opportunistic refactors.
- Do not merge to `main`.

## Source of Truth

- Repo instructions: `/home/azureuser/repos/projects/QuantAgent/AGENTS.md` and `/home/azureuser/repos/projects/QuantAgent/CLAUDE.md` if present.
- Beads issue: `QuantAgent-uz9`.
- Labels at generation time: `none`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-uz9/20260507T174143Z-QuantAgent-uz9-implementer/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

## Contexto
El deploy QA en el runner self-hosted volvió a fallar con `no space left on device` durante `docker-compose ... build --no-cache`. La verificación post-deploy confirmó dos factores concretos: (1) root disk en 96% y (2) `.worktrees/` del checkout principal entra al contexto Docker porque `.dockerignore` no la excluye.

## Cambio requerido
1. Excluir `.worktrees/` del contexto Docker de QuantAgent
2. Mantener el cambio acotado a higiene de build/deploy (sin tocar lógica de app)
3. Dejar evidencia verificable para que futuros merges a `main` no vuelvan a copiar worktrees locales al build context

## Criterio de aceptación
- [ ] `.dockerignore` ignora `.worktrees/`
- [ ] El diff queda limitado a higiene de build context
- [ ] Se deja artifact/autodev evidence vinculada al ticket
- [ ] `QuantAgent-82t` puede reevaluarse sin este blocker de contexto Docker

## Archivos relevantes
- `.dockerignore`
- `docs/envelopes/deploy-verification-20260507T132230Z.md`
- `.github/workflows/main-ci-deploy.yml` (solo como contexto; fuera de scope del fix)

## Fuera de scope
- pruning manual/destructivo de Docker
- cambios de negocio o app
- refactor del workflow CI más allá de consumir un checkout más limpio


## Acceptance Criteria

No acceptance criteria found in Beads issue.

## Recent Beads Comments

- No recent comments captured.

## Preflight Evidence

- Current branch at generation: `main`.
- Repo dirty at generation: `True`.
- Dirty files:
  - ` M .beads/issues.jsonl`
  - `?? docs/envelopes/deploy-verification-20260507T132230Z.md`

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
### Skill: autodev-implementer
- **Resultado:** SUCCESS | PARTIAL | BLOCKED | FAIL
- **Run-ID:** 20260507T174143Z-QuantAgent-uz9-implementer
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
