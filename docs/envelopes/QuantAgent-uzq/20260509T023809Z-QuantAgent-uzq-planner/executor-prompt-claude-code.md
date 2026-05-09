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
        - Run ID: 20260509T023809Z-QuantAgent-uzq-planner
        - Phase: planner
        - Skill: autodev-planner
        - Issue: QuantAgent-uzq
        - Repo: /home/azureuser/repos/projects/QuantAgent
        - Worktree: /mnt/actions-runner/autodev-runtime/worktrees/QuantAgent/QuantAgent-uzq/planner-20260509T023809Z
        - Artifacts dir: /home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-uzq/20260509T023809Z-QuantAgent-uzq-planner
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
        run_id: "20260509T023809Z-QuantAgent-uzq-planner"
phase: "planner"
executor: "claude-code"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-uzq"
branch_policy: "worktree-preferred"
base_branch: "main"
current_branch_at_generation: "main"
feature_branch: "feature/QuantAgent-uzq-fix-tradingscheduler-heartbeat-and-sched"
worktree_root: "/mnt/actions-runner/autodev-runtime/worktrees/QuantAgent"
worktree_path: "/mnt/actions-runner/autodev-runtime/worktrees/QuantAgent/QuantAgent-uzq/planner-20260509T023809Z"
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
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-uzq/20260509T023809Z-QuantAgent-uzq-planner"
generated_at: "2026-05-09T02:38:09.438832+00:00"
preflight:
  repo_dirty: true
  dirty_files:
    - "?? docs/envelopes/QuantAgent-l8r/20260508T124108Z-QuantAgent-l8r-planner/"
        ---

# Autodev Input Envelope — QuantAgent-uzq — planner

## Objective

Execute the `planner` phase for Beads issue `QuantAgent-uzq`: Fix TradingScheduler heartbeat and scheduler unit-test regressions exposed by CI gate.

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
- Beads issue: `QuantAgent-uzq`.
- Labels at generation time: `none`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-uzq/20260509T023809Z-QuantAgent-uzq-planner/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

## Contexto
La revalidación de `QuantAgent-82t` sobre `main` con PostgreSQL local mostró 8 fallas pre-existentes concentradas en los tests del scheduler/heartbeat.

## Comando verificado
`DATABASE_URL=postgresql://test:***@localhost:5432/quantagent_test /mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python -m pytest tests/test_vje_scheduler_heartbeat_backend.py tests/trading/test_scheduler.py -v --tb=short --maxfail=10 -m "not integration and not slow"`

## Fallas verificables
### Heartbeat backend (`tests/test_vje_scheduler_heartbeat_backend.py`)
- `test_upsert_heartbeat_start_updates_existing_row_for_environment`
- `test_upsert_heartbeat_complete_sets_last_trade_id`
- `test_analyze_and_trade_continues_when_heartbeat_start_fails`
- `test_analyze_and_trade_full_cycle_writes_completed_heartbeat`
- `test_analyze_and_trade_per_asset_error_completes_heartbeat_with_error_count`

Síntomas observados:
- `TradingScheduler` ya no expone `_upsert_heartbeat_start`
- `analyze_and_trade()` no llama al mock `_upsert_heartbeat_complete`
- no se persiste ninguna fila `SchedulerHeartbeat` en el ciclo real

### Scheduler unit tests (`tests/trading/test_scheduler.py`)
- `test_trading_scheduler_long_signal_executes_order`
- `test_trading_scheduler_short_signal_executes_order`
- `test_trading_scheduler_hold_signal_skips_execution`

Síntoma observado:
- `stats["processed"] == 0` porque `DummySession.query()` devuelve un `Mock`, el scheduler lo interpreta como posición activa y `PositionMonitor.update_candle_tracking()` rompe con `TypeError: unsupported operand type(s) for +=: 'Mock' and 'int'`

## Cambio requerido
Alinear implementación y/o tests del `TradingScheduler` con el contrato actual después de la integración de PositionMonitor y del feature de scheduler heartbeat, para que estos tests vuelvan a pasar de forma reproducible en `main`.

## Criterio de aceptación
- [ ] Los 5 tests fallidos de `tests/test_vje_scheduler_heartbeat_backend.py` pasan
- [ ] Los 3 tests fallidos de `tests/trading/test_scheduler.py` pasan
- [ ] El comando exacto del gate de `QuantAgent-82t` deja de frenarse por estos módulos

## Archivos relevantes
- `quantagent/trading/scheduler.py`
- `quantagent/trading/position_monitor.py`
- `tests/test_vje_scheduler_heartbeat_backend.py`
- `tests/trading/test_scheduler.py`
- `quantagent/models.py`

## Fuera de scope
- Cambiar `.github/workflows/main-ci-deploy.yml`
- Corregir `tests/test_wait_sec_deprecation_removal.py`


## Acceptance Criteria

No acceptance criteria found in Beads issue.

## Recent Beads Comments

- No recent comments captured.

## Preflight Evidence

- Current branch at generation: `main`.
- Repo dirty at generation: `True`.
- Dirty files:
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
- **Run-ID:** 20260509T023809Z-QuantAgent-uzq-planner
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
