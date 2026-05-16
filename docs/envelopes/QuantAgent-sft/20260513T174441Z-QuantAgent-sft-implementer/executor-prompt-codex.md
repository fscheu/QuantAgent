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
        - Run ID: 20260513T174441Z-QuantAgent-sft-implementer
        - Phase: implementer
        - Skill: autodev-implementer
        - Issue: QuantAgent-sft
        - Repo: /home/azureuser/repos/projects/QuantAgent
        - Worktree: /mnt/actions-runner/autodev-runtime/worktrees/QuantAgent/QuantAgent-sft/implementer-20260513T174441Z
        - Artifacts dir: /home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-sft/20260513T174441Z-QuantAgent-sft-implementer
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
  "write_code": true,
  "write_tests": false,
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
    "ruff check --fix .",
    "pytest <relevant subset> -v",
    "python -m compileall -q ."
  ],
  "optional": [
    "python -m compileall -q ."
  ]
}

        FULL INPUT ENVELOPE:
        ---
        run_id: "20260513T174441Z-QuantAgent-sft-implementer"
phase: "implementer"
executor: "codex"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-sft"
branch_policy: "worktree-preferred"
base_branch: "main"
current_branch_at_generation: "main"
feature_branch: "feature/QuantAgent-sft-endurecer-paper-trading-loop-y-auditor-a"
worktree_root: "/mnt/actions-runner/autodev-runtime/worktrees/QuantAgent"
worktree_path: "/mnt/actions-runner/autodev-runtime/worktrees/QuantAgent/QuantAgent-sft/implementer-20260513T174441Z"
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
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-sft/20260513T174441Z-QuantAgent-sft-implementer"
generated_at: "2026-05-13T17:44:41.132680+00:00"
preflight:
  repo_dirty: true
  dirty_files:
    - "?? docs/01_requirements/QuantAgent-375-RQ-scope-replay-signal-lookup.md"
    - "?? docs/02_planning/QuantAgent-375-PL-scope-replay-signal-lookup.md"
    - "?? docs/03_design/QuantAgent-375-DS-scope-replay-signal-lookup.md"
    - "?? docs/05_acceptance_tests/QuantAgent-375-AC-scope-replay-signal-lookup.md"
    - "?? docs/envelopes/QuantAgent-339/"
    - "?? docs/envelopes/QuantAgent-375/20260511T073839Z-QuantAgent-375-planner/"
    - "?? docs/envelopes/QuantAgent-375/20260511T074338Z-QuantAgent-375-implementer/"
    - "?? docs/envelopes/QuantAgent-3o8/20260510T073737Z-QuantAgent-3o8-implementer/"
    - "?? docs/envelopes/QuantAgent-3o8/20260510T214304Z-QuantAgent-3o8-tech-lead/"
    - "?? docs/envelopes/QuantAgent-4fm/"
    - "?? docs/envelopes/QuantAgent-69d/20260512T171657Z-QuantAgent-69d-tester/"
    - "?? docs/envelopes/QuantAgent-69d/20260512T173947Z-QuantAgent-69d-implementer/"
    - "?? docs/envelopes/QuantAgent-69d/20260512T174841Z-QuantAgent-69d-implementer/"
    - "?? docs/envelopes/QuantAgent-69d/20260512T175807Z-QuantAgent-69d-tester/"
    - "?? docs/envelopes/QuantAgent-l8r/20260508T124108Z-QuantAgent-l8r-planner/"
    - "?? docs/envelopes/QuantAgent-s62/20260513T123928Z-QuantAgent-s62-planner/"
    - "?? docs/envelopes/QuantAgent-s62/20260513T124414Z-QuantAgent-s62-implementer/"
    - "?? docs/envelopes/QuantAgent-s62/20260513T125257Z-QuantAgent-s62-tester/"
    - "?? docs/envelopes/QuantAgent-vje/"
        ---

# Autodev Input Envelope — QuantAgent-sft — implementer

## Objective

Execute the `implementer` phase for Beads issue `QuantAgent-sft`: Endurecer paper trading loop y auditoría operacional.

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
- Beads issue: `QuantAgent-sft`.
- Labels at generation time: `openclaw:design_pending`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-sft/20260513T174441Z-QuantAgent-sft-implementer/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

## Contexto
QuantAgent ya cuenta con TradingScheduler (`QuantAgent-3o4`) y PositionMonitor integrado al loop de paper trading (`QuantAgent-0b5`). Sin embargo, M2 exige demostrar que el runtime paper puede operar de forma repetible y auditable, no solo que la feature existe.

Hoy falta consolidar la capa operacional para que varios ciclos consecutivos no dejen estados inconsistentes, órdenes/trades huérfanos o heartbeats engañosos.

## Cambio requerido
Endurecer el loop de paper trading en environment `paper` para que la operación sea estable y auditable.

Esto implica, como mínimo:
- validar y reforzar invariantes del ciclo scheduler → análisis → ejecución → tracking de posición;
- asegurar que la cadena señal → orden → trade → posición pueda reconstruirse de forma confiable;
- detectar y dejar observable cualquier estado zombie o inconsistente (por ejemplo heartbeat `running` eterno, posición abierta sin continuidad operativa, orden/trade sin linkage útil);
- agregar pruebas enfocadas en multi-cycle paper runtime, no sólo tests unitarios aislados.

## Criterio de aceptación
- [ ] Existe al menos una validación automatizada que cubre múltiples ciclos consecutivos de paper trading sin crash ni corrupción de estado.
- [ ] La cadena señal → orden → trade → posición se puede consultar o reconstruir consistentemente para environment `paper`.
- [ ] Los paths de error fatal y no fatal dejan estado observable y no bloquean el scheduler en un falso `running` permanente.
- [ ] No aparecen registros huérfanos en la operación normal del loop paper (o, si hay degradación aceptada, queda explícitamente detectada y reportada).

## Archivos relevantes
- `quantagent/trading/scheduler.py`
- `quantagent/trading/order_manager.py`
- `quantagent/trading/position_monitor.py`
- `quantagent/models.py`
- `apps/paper_trading.py`
- `tests/trading/`
- `tests/test_vje_scheduler_heartbeat_backend.py`

## Fuera de scope
- Broker real
- Nuevas estrategias M1
- Rediseñar la arquitectura del motor de trading más allá de lo necesario para estabilidad operativa paper

## Notas técnicas
- Reusar como baseline lo ya mergeado en `QuantAgent-3o4`, `QuantAgent-0b5` y `QuantAgent-vje`.
- Si aparecen gaps de provenance hoy cubiertos parcialmente por `thread_id`, `checkpoint_id` u otros vínculos persistidos, cerrarlos sin reabrir scope de replay que ya fue tratado en `QuantAgent-375`.


## Acceptance Criteria

No acceptance criteria found in Beads issue.

## Recent Beads Comments

- Comment 195 by Federico Scheu at 2026-05-13T02:49:29Z:

  ### Skill: autodev-planner
  - **Resultado:** SUCCESS
  - **Run-ID:** 20260513T023641Z

  #### Qué hice
  - Docs creados/editados:
    - `docs/01_requirements/QuantAgent-sft-RQ-paper-runtime-hardening.md`
    - `docs/02_planning/QuantAgent-sft-PL-paper-runtime-hardening.md`
    - `docs/03_design/QuantAgent-sft-DS-paper-runtime-hardening.md`
    - `docs/05_acceptance_tests/QuantAgent-sft-AC-paper-runtime-hardening.md`
    - índices `README.md` de requirements/planning/design/acceptance
  - Artefactos durables:
    - `docs/envelopes/QuantAgent-sft/20260513T023641Z-techlead-planner/run-report.md`
    - `docs/envelopes/QuantAgent-sft/20260513T023641Z-techlead-planner/result.json`
  - Branch publicada: `feature/QuantAgent-sft-paper-runtime-hardening` (`cdc53077`)

  #### Cambios en el ticket
  - Sin cambios de labels/estado en esta corrida.

  #### Problemas encontrados
  - El repo root principal sigue dirty por trabajo previo no atribuible a esta corrida; se aisló todo en worktree.

  #### Dudas / decisiones pendientes
  1) `QuantAgent-s62` y `QuantAgent-339` siguen siendo dependencias del milestone M2.
  2) `QuantAgent-69d` queda como mejora complementaria de telemetry, no bloqueo del runtime base.

  #### Next step recomenda…

## Preflight Evidence

- Current branch at generation: `main`.
- Repo dirty at generation: `True`.
- Dirty files:
  - `?? docs/01_requirements/QuantAgent-375-RQ-scope-replay-signal-lookup.md`
  - `?? docs/02_planning/QuantAgent-375-PL-scope-replay-signal-lookup.md`
  - `?? docs/03_design/QuantAgent-375-DS-scope-replay-signal-lookup.md`
  - `?? docs/05_acceptance_tests/QuantAgent-375-AC-scope-replay-signal-lookup.md`
  - `?? docs/envelopes/QuantAgent-339/`
  - `?? docs/envelopes/QuantAgent-375/20260511T073839Z-QuantAgent-375-planner/`
  - `?? docs/envelopes/QuantAgent-375/20260511T074338Z-QuantAgent-375-implementer/`
  - `?? docs/envelopes/QuantAgent-3o8/20260510T073737Z-QuantAgent-3o8-implementer/`
  - `?? docs/envelopes/QuantAgent-3o8/20260510T214304Z-QuantAgent-3o8-tech-lead/`
  - `?? docs/envelopes/QuantAgent-4fm/`
  - `?? docs/envelopes/QuantAgent-69d/20260512T171657Z-QuantAgent-69d-tester/`
  - `?? docs/envelopes/QuantAgent-69d/20260512T173947Z-QuantAgent-69d-implementer/`
  - `?? docs/envelopes/QuantAgent-69d/20260512T174841Z-QuantAgent-69d-implementer/`
  - `?? docs/envelopes/QuantAgent-69d/20260512T175807Z-QuantAgent-69d-tester/`
  - `?? docs/envelopes/QuantAgent-l8r/20260508T124108Z-QuantAgent-l8r-planner/`
  - `?? docs/envelopes/QuantAgent-s62/20260513T123928Z-QuantAgent-s62-planner/`
  - `?? docs/envelopes/QuantAgent-s62/20260513T124414Z-QuantAgent-s62-implementer/`
  - `?? docs/envelopes/QuantAgent-s62/20260513T125257Z-QuantAgent-s62-tester/`
  - `?? docs/envelopes/QuantAgent-vje/`

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
- **Run-ID:** 20260513T174441Z-QuantAgent-sft-implementer
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
