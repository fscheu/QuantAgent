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
        - Run ID: 20260503T213823Z-QuantAgent-0b5-tester
        - Phase: tester
        - Skill: autodev-tester
        - Issue: QuantAgent-0b5
        - Repo: /home/azureuser/repos/projects/QuantAgent
        - Artifacts dir: /home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-0b5/20260503T213823Z-QuantAgent-0b5-tester

        CAPABILITIES:
        {
  "read_repo": true,
  "write_docs": true,
  "write_code": false,
  "write_tests": true,
  "beads_read": true,
  "beads_comment_final": true,
  "beads_update_labels": false,
  "git_create_branch": true,
  "git_commit": true,
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
    "confirm branch is not main",
    "pytest <new/changed tests> -v",
    "pytest <relevant subset> -v"
  ],
  "optional": [
    "python -m compileall -q ."
  ]
}

        FULL INPUT ENVELOPE:
        ---
        run_id: "20260503T213823Z-QuantAgent-0b5-tester"
phase: "tester"
executor: "claude-code"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-0b5"
branch_policy: "worktree-preferred"
base_branch: "main"
current_branch_at_generation: "main"
feature_branch: "feature/QuantAgent-0b5-integrate-positionmonitor-into-tradingsc"
worktree_path: "/tmp/autodev-worktrees/QuantAgent/QuantAgent-0b5/tester-20260503T213823Z"
skill: "autodev-tester"
mode: "tests-only"
capabilities:
  read_repo: true
  write_docs: true
  write_code: false
  write_tests: true
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
    - "confirm branch is not main"
    - "pytest <new/changed tests> -v"
    - "pytest <relevant subset> -v"
  optional:
    - "python -m compileall -q ."
budget:
  max_turns: 10
  max_minutes: 45
  max_cost_usd: null
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-0b5/20260503T213823Z-QuantAgent-0b5-tester"
generated_at: "2026-05-03T21:38:23.537937+00:00"
preflight:
  repo_dirty: true
  dirty_files:
    - " M .beads/issues.jsonl"
        ---

# Autodev Input Envelope — QuantAgent-0b5 — tester

## Objective

Execute the `tester` phase for Beads issue `QuantAgent-0b5`: Integrate PositionMonitor into TradingScheduler.

## Scope In

- Follow the `autodev-tester` skill for this phase.
- Use the Beads issue, repo instructions, docs, and recent comments as source of truth.
- Preserve migration defaults: no push, no merge, no deploy, no external messages.

## Scope Out

- Do not modify credentials, `.env`, secrets, production service config, or systemd/OpenClaw/Hermes runtime config.
- Do not rename legacy `openclaw:*` labels during this phase.
- Do not work on unrelated issues or opportunistic refactors.
- Do not merge to `main`.

## Source of Truth

- Repo instructions: `/home/azureuser/repos/projects/QuantAgent/AGENTS.md` and `/home/azureuser/repos/projects/QuantAgent/CLAUDE.md` if present.
- Beads issue: `QuantAgent-0b5`.
- Labels at generation time: `enhancement, openclaw:design_approved, openclaw:impl_done, trading`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-0b5/20260503T213823Z-QuantAgent-0b5-tester/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

PositionMonitor was implemented and tested (QuantAgent-boi, QuantAgent-on4) and integrated into the Backtest path. However, TradingScheduler (apps/paper_trading.py) does not call PositionMonitor at any point. Paper trading exits are handled by fixed-percentage logic in the strategy, bypassing all the position tracking infrastructure.

Current gap:
- TradingScheduler.analyze_and_trade() runs the full LLM graph on every tick without checking PositionMonitor.should_exit()
- Active positions are not passed through PositionMonitor.update_tracking()
- Exit reason tracking (directional accuracy, stop loss, take profit) never gets recorded in live trading

Required changes:
- On each scheduler tick: call PositionMonitor.update_tracking() for all open positions
- Before re-analyzing: check PositionMonitor.should_exit() and execute exit if needed
- Record exit reason in Trade model

Relevant files:
- apps/paper_trading.py — TradingScheduler entry point
- quantagent/trading/scheduler.py — analyze_and_trade() method
- quantagent/trading/position_monitor.py — already implemented

## Acceptance Criteria

- TradingScheduler calls PositionMonitor on each tick
- Exit reasons are persisted to Trade records
- At least 3 new integration tests covering: normal hold, stop-loss exit, take-profit exit
- Existing scheduler unit tests still pass

## Recent Beads Comments

- Comment 71 by Federico Scheu at 2026-04-27T08:14:21Z:

  ### Skill: autodev-implementer
  - **Resultado:** SUCCESS
  - **Run-ID:** 20260426-235500Z

  #### Qué hice
  - **Branch:** `feature/QuantAgent-0b5-integrate-positionmonitor-into-tradingsc`
  - **Commit:** `782a3f9e`
  - **Cambios principales:**

  **Core Implementation (Tasks 1-5 completed):**

  1. **Task 1:** Added PositionMonitor to `TradingScheduler.__init__`
     - Import statement added
     - Instance created with `backtest_run_id=None` for paper trading

  2. **Task 2:** Implemented `_check_exit_conditions()` helper method
     - Checks stop loss (LONG: price <= stop_loss, SHORT: price >= stop_loss)
     - Checks take profit (LONG: price >= take_profit, SHORT: price <= take_profit)
     - Checks max hold (candles_since_entry >= max_hold_candles)
     - Returns tuple: (should_exit: bool, exit_reason: Optional[str])

  3. **Task 3:** Implemented `_execute_position_exit()` helper method
     - Closes position via `PositionMonitor.close_position()`
     - Executes exit order via `OrderManager.execute_decision()` (opposite signal)
     - Updates Trade record with exit_signal and closed_at
     - Proper error handling with rollback on failure

  4. **Task 4:** Integrated position monitoring into `_process_asset()`
     - …
- Comment 85 by Federico Scheu at 2026-05-03T17:50:25Z:

  ### Skill: autodev-implementer
  - **Resultado:** PARTIAL
  - **Run-ID:** 20260503-174957Z

  #### Qué hice
  - **Branch:** `feature/QuantAgent-0b5-integrate-positionmonitor-into-tradingsc-clean`
  - Resolví el conflicto en `quantagent/trading/scheduler.py` preservando tanto la integración de `PositionMonitor` como la lógica de heartbeat.
  - Ajusté `tests/test_vje_scheduler_heartbeat_backend.py` para crear `active_positions` en el fixture SQLite y devolver una orden mock compatible con la nueva apertura de posiciones.
  - Generé commit limpio con el cambio integrado: `c5bfec57`.

  #### Artefactos / evidencia
  - **Commit:** `c5bfec57 feat(QuantAgent-0b5): integrate PositionMonitor into TradingScheduler`
  - **Archivos tocados:**
    - `quantagent/trading/scheduler.py`
    - `tests/trading/test_scheduler.py`
    - `tests/trading/test_scheduler_position_monitor.py`
    - `tests/test_vje_scheduler_heartbeat_backend.py`

  #### Quality gates
  - `python -m ruff check --fix quantagent/trading/scheduler.py tests/trading/test_scheduler.py tests/trading/test_scheduler_position_monitor.py tests/test_vje_scheduler_heartbeat_backend.py` ✅
  - `python -m py_compile quantagent/trading/scheduler.py` ✅
  - `python -m pytest tests…
- Comment 86 by Federico Scheu at 2026-05-03T18:49:16Z:

  ### Tech Lead / Patrol Mode
  - Acción: cleanup de labels BEADS
  - Issue: QuantAgent-0b5

  Se removió el label stale `openclaw:design_pending` porque el ticket ya estaba en etapa posterior (`openclaw:impl_done`).

  Motivo:
  - El orquestador legacy primero saltea cualquier ticket con `openclaw:design_pending`.
  - Con `impl_done` presente, ese label bloqueaba indebidamente que el próximo ciclo lo tome `autodev-tester`.

  Cambio aplicado:
  - removido: `openclaw:design_pending`
  - preservado: `openclaw:impl_done`

  No se tocó código ni docs; sólo estado del ticket para destrabar routing.

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
- End with an output envelope/report containing: summary, files changed, commands run, quality gates, artifacts, risks, next step, and Beads update.

## Required Final Beads Comment Template

```md
### Skill: autodev-tester
- **Resultado:** SUCCESS | PARTIAL | BLOCKED | FAIL
- **Run-ID:** 20260503T213823Z-QuantAgent-0b5-tester
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
