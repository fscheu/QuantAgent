You are running a Hermes autodev phase under a strict executor contract.

        READ THIS FIRST:
        - One run = one Beads issue + one phase.
        - Do not infer extra permissions. The YAML capabilities are authoritative.
        - No git push, no merge to main, no deploy, no external messages.
        - Do not read or print secrets.
        - If blocked, stop and return a structured BLOCKED report.
        - End with a structured output envelope/report.
        - Write canonical artifacts under the declared artifacts dir whenever possible: `result.json`, `run-report.md`, `commands.log`, `quality-gates.log`.
        - If Beads is available and the envelope allows it, add exactly one final Beads comment using the provided template.

        RUN METADATA:
        - Run ID: 20260508T124108Z-QuantAgent-l8r-planner
        - Phase: planner
        - Skill: autodev-planner
        - Issue: QuantAgent-l8r
        - Repo: /home/azureuser/repos/projects/QuantAgent
        - Worktree: /mnt/actions-runner/autodev-runtime/worktrees/QuantAgent/QuantAgent-l8r/planner-20260508T124108Z
        - Artifacts dir: /home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-l8r/20260508T124108Z-QuantAgent-l8r-planner
        - Shared venv: /mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv
        - Shared python: /mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python

        SHARED ENVIRONMENT POLICY:
        - If `shared_python` is declared, prefer it for Python, pytest, and tooling commands.
        - Worktrees should reuse the shared venv declared in the envelope instead of creating a per-worktree `.venv`.
        - The router prepends the shared venv `bin/` directory to PATH for the executor process.

        CAPABILITIES:
        {
  "read_repo": true,
  "write_docs": true,
  "write_code": false,
  "write_tests": false,
  "beads_read": true,
  "beads_comment_final": true,
  "beads_update_labels": false,
  "git_create_branch": false,
  "git_commit": false,
  "git_push": false,
  "merge_main": false,
  "deploy": false,
  "send_external_message": false,
  "touch_secrets": false
}

        FORBIDDEN ACTIONS:
        [
  "git push",
  "git merge main",
  "deploy commands",
  "read or print .env/secrets/tokens",
  "send Telegram/email/Slack",
  "change Beads labels/status unless explicitly enabled",
  "stash, discard, reset --hard, or rewrite history"
]

        QUALITY GATES:
        {
  "required": [
    "git status --short",
    "verify issue ID appears in docs paths",
    "verify acceptance criteria are testable"
  ],
  "optional": [
    "python -m compileall -q ."
  ]
}

        FULL INPUT ENVELOPE:
        ---
        run_id: "20260508T124108Z-QuantAgent-l8r-planner"
phase: "planner"
executor: "claude-code"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-l8r"
branch_policy: "worktree-preferred"
base_branch: "main"
current_branch_at_generation: "main"
feature_branch: "feature/QuantAgent-l8r-fix-trade-p-l-calculation-regressions-ex"
worktree_root: "/mnt/actions-runner/autodev-runtime/worktrees/QuantAgent"
worktree_path: "/mnt/actions-runner/autodev-runtime/worktrees/QuantAgent/QuantAgent-l8r/planner-20260508T124108Z"
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
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-l8r/20260508T124108Z-QuantAgent-l8r-planner"
generated_at: "2026-05-08T12:41:08.189209+00:00"
preflight:
  repo_dirty: true
  dirty_files:
    - " M .beads/issues.jsonl"
        ---

# Autodev Input Envelope — QuantAgent-l8r — planner

## Objective

Execute the `planner` phase for Beads issue `QuantAgent-l8r`: Fix trade P&L calculation regressions exposed by CI gate.

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
- Beads issue: `QuantAgent-l8r`.
- Labels at generation time: `none`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-l8r/20260508T124108Z-QuantAgent-l8r-planner/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

## Contexto
La revalidación de `QuantAgent-82t` mostró que la suite de PnL (`tests/test_r78_trade_pnl_calculation.py`) sigue fallando con resultados numéricos incorrectos, incluso después de haber liberado bloqueadores previos del gate.

## Evidencia
Comando exacto del gate:
`DATABASE_URL=postgresql://test:test@localhost:5432/quantagent_test /mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python -m pytest tests/ -v --tb=short --maxfail=10 -m "not integration and not slow"`

Fallas verificables:
- `TestLongPositionPnL::test_long_position_profit`
- `TestLongPositionPnL::test_long_position_loss`
- `TestShortPositionPnL::test_short_position_profit`
- `TestShortPositionPnL::test_short_position_loss`
- `TestOpeningPositionNoPnL::test_opening_position_has_no_pnl`

Síntoma: el modelo devuelve `Decimal("1000.00000000")` o `Decimal("250.00000000")` donde el contrato de test espera `500`, `-500` o `None`.

## Cambio requerido
Corregir la lógica o los tests del cálculo de P&L para que el contrato documentado vuelva a ser consistente y el gate no falle por este módulo.

## Criterio de aceptación
- [ ] Los 5 tests fallidos de `tests/test_r78_trade_pnl_calculation.py` pasan
- [ ] El gate exacto de `QuantAgent-82t` deja de frenarse por P&L calculation

## Archivos relevantes
- `tests/test_r78_trade_pnl_calculation.py`
- `quantagent/portfolio/manager.py`
- `quantagent/models.py`

## Fuera de scope
- Cambiar workflow CI
- Resolver PositionMonitor o benchmark fixture en este ticket


## Acceptance Criteria

No acceptance criteria found in Beads issue.

## Recent Beads Comments

- No recent comments captured.

## Preflight Evidence

- Current branch at generation: `main`.
- Repo dirty at generation: `True`.
- Dirty files:
  - ` M .beads/issues.jsonl`

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
- **Run-ID:** 20260508T124108Z-QuantAgent-l8r-planner
- **Executor:** hermes-internal

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
