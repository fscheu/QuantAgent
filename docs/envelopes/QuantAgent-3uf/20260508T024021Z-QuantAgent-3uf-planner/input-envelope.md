---
run_id: "20260508T024021Z-QuantAgent-3uf-planner"
phase: "planner"
executor: "auto"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-3uf"
branch_policy: "worktree-preferred"
base_branch: "main"
current_branch_at_generation: "main"
feature_branch: "feature/QuantAgent-3uf-fix-positionmonitor-unit-test-regression"
worktree_root: "/mnt/actions-runner/autodev-runtime/worktrees/QuantAgent"
worktree_path: "/mnt/actions-runner/autodev-runtime/worktrees/QuantAgent/QuantAgent-3uf/planner-20260508T024021Z"
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
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-3uf/20260508T024021Z-QuantAgent-3uf-planner"
generated_at: "2026-05-08T02:40:21.186750+00:00"
preflight:
  repo_dirty: true
  dirty_files:
    - "?? docs/envelopes/QuantAgent-uz9/20260507T174143Z-QuantAgent-uz9-implementer/"
    - "?? docs/envelopes/deploy-verification-20260507T132230Z.md"
---

# Autodev Input Envelope — QuantAgent-3uf — planner

## Objective

Execute the `planner` phase for Beads issue `QuantAgent-3uf`: Fix PositionMonitor unit-test regressions exposed by CI gate.

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
- Beads issue: `QuantAgent-3uf`.
- Labels at generation time: `none`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-3uf/20260508T024021Z-QuantAgent-3uf-planner/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

## Contexto
La reevaluación de `QuantAgent-82t` expuso 4 fallas adicionales relacionadas con `PositionMonitor` y constraints de `ActivePosition`.

## Evidencia
Comando exacto del gate:
`DATABASE_URL=postgresql://test:test@localhost:5432/quantagent_test /mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python -m pytest tests/ -v --tb=short --maxfail=10 -m "not integration and not slow"`

Fallas verificables:
- `tests/test_position_monitor.py::test_only_one_active_position_per_symbol`
- `tests/test_position_monitor_constraints.py::test_position_with_all_optional_fields`
- `tests/test_position_monitor_constraints.py::test_get_active_position_returns_most_recent_if_multiple`
- `tests/test_position_monitor_constraints.py::test_closed_position_not_returned_by_get_active`

Se observan dos síntomas principales:
1. múltiples posiciones activas residuales donde el test esperaba una sola
2. FK/semántica de `trade_id` y selección de posición activa no alineadas con el contrato esperado

## Cambio requerido
Alinear tests y/o implementación de `PositionMonitor` con el contrato actual de `ActivePosition`, manteniendo aislamiento entre tests y reglas coherentes para posición activa.

## Criterio de aceptación
- [ ] Los 4 tests fallidos de PositionMonitor pasan
- [ ] El gate exacto de `QuantAgent-82t` deja de frenarse por PositionMonitor

## Archivos relevantes
- `tests/test_position_monitor.py`
- `tests/test_position_monitor_constraints.py`
- `quantagent/trading/position_monitor.py`
- `quantagent/models.py`

## Fuera de scope
- Cambiar workflow CI
- Resolver benchmark fixture faltante o PnL calculation en este ticket


## Acceptance Criteria

No acceptance criteria found in Beads issue.

## Recent Beads Comments

- No recent comments captured.

## Preflight Evidence

- Current branch at generation: `main`.
- Repo dirty at generation: `True`.
- Dirty files:
  - `?? docs/envelopes/QuantAgent-uz9/20260507T174143Z-QuantAgent-uz9-implementer/`
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
### Skill: autodev-planner
- **Resultado:** SUCCESS | PARTIAL | BLOCKED | FAIL
- **Run-ID:** 20260508T024021Z-QuantAgent-3uf-planner
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
