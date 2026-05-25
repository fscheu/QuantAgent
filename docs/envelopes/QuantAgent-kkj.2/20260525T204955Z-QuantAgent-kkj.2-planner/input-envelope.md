---
run_id: "20260525T204955Z-QuantAgent-kkj.2-planner"
phase: "planner"
executor: "auto"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-kkj.2"
branch_policy: "in-place-publication"
base_branch: "main"
publication_branch: "main"
current_branch_at_generation: "main"
feature_branch: null
worktree_root: null
worktree_path: null
shared_venv: "/home/azureuser/repos/projects/QuantAgent/.venv"
shared_python: "/home/azureuser/repos/projects/QuantAgent/.venv/bin/python"
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
  git_commit: true
  git_push: true
  merge_main: false
  deploy: false
  send_external_message: false
  touch_secrets: false
forbidden_actions:
  - "git push to any branch other than the declared publication branch"
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
    - "confirm repo is clean before canonical planner publication"
    - "confirm current branch matches canonical publication branch"
  optional:
    - "python -m compileall -q ."
budget:
  max_turns: 10
  max_minutes: 45
  max_cost_usd: null
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-kkj.2/20260525T204955Z-QuantAgent-kkj.2-planner"
generated_at: "2026-05-25T20:49:55.790432+00:00"
preflight:
  repo_dirty: false
  dirty_files: []
---

# Autodev Input Envelope — QuantAgent-kkj.2 — planner

## Objective

Execute the `planner` phase for Beads issue `QuantAgent-kkj.2`: Agregar controles de scheduler paper trading en UI (start/stop).

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
- Beads issue: `QuantAgent-kkj.2`.
- Labels at generation time: `none`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-kkj.2/20260525T204955Z-QuantAgent-kkj.2-planner/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

## Contexto

El TradingScheduler de paper trading existe desde QuantAgent-3o4 y fue endurecido en QuantAgent-sft (heartbeat running/stuck/error, cadena signal→orden→trade→posición auditable). Sin embargo, hoy el scheduler solo puede iniciarse y stoppearlos via línea de comandos. La vista Paper Trading en Streamlit (`apps/streamlit/views/paper_trading.py`) solo muestra el estado del scheduler pero no ofrece controles de ciclo de vida.

Observado durante revisión funcional M2 (2026-05-25): el scheduler figura como "stopped", lo cual es correcto, pero el user manual indica que debe correrse via CLI. Operar QuantAgent en paper trading sin acceso a terminal crea una barrera operativa para cualquier operador o evaluador.

Este es un gap práctico para M3: sin controles de UI no se puede arrancar ni detener un ciclo de paper trading de forma autónoma desde la interfaz. El readiness report de QuantAgent-aki (piloto M2) recomienda avanzar a M3, y ese avance requiere poder operar el scheduler desde la UI.

## Cambio requerido

Agregar en la vista Paper Trading controles básicos para el ciclo de vida del scheduler:

- **Start**: iniciar una corrida de scheduler con parámetros configurables (estrategias, universo de activos, environment=paper, modo continuo o N ciclos).
- **Stop**: detener el scheduler activo de forma graceful.
- El estado actual (running/stuck/error/stopped, ya implementado por QuantAgent-sft) se mantiene visible junto a los controles.

## Criterio de aceptación

- [ ] Desde la UI Paper Trading, el operador puede iniciar el scheduler con parámetros básicos (estrategias, universo, environment=paper).
- [ ] Desde la UI Paper Trading, el operador puede detener el scheduler activo.
- [ ] El estado del scheduler (running/stuck/error/stopped) se actualiza en la UI con refresh automático o similar tras iniciar/detener.
- [ ] Si el scheduler ya está running, el botón Start está deshabilitado o muestra advertencia.
- [ ] Si el scheduler está stopped, el botón Stop está deshabilitado.
- [ ] Los estados y la semántica de heartbeat implementados en QuantAgent-sft no se alteran.
- [ ] El user manual se actualiza para reflejar la operación vía UI.

## Archivos relevantes

- `apps/streamlit/views/paper_trading.py` — vista principal a extender.
- `quantagent/trading/scheduler.py` — TradingScheduler a invocar/controlar.
- `quantagent/models.py` — SchedulerHeartbeat para leer estado actual.

## Fuera de scope (no tocar)

- No conectar a broker real.
- No implementar nuevas estrategias.
- No rediseñar el dashboard principal ni otras vistas.
- No implementar modo pause si el scheduler no lo soporta nativamente.

## Notas técnicas

- Considerar si el scheduler se lanza como subprocess (sobrevive reloads de Streamlit) o como thread en proceso. Subprocess es más robusto para Streamlit.
- El estado actual del scheduler puede leerse via heartbeats en DB (ya funciona desde QuantAgent-sft).
- El PID del proceso subprocess debe persistirse (en DB o session_state) para poder enviarlo la señal de stop.


## Acceptance Criteria

No acceptance criteria found in Beads issue.

## Recent Beads Comments

- No recent comments captured.

## Preflight Evidence

- Current branch at generation: `main`.
- Canonical publication branch: `main`.
- Repo dirty at generation: `False`.
- Dirty files:
  - none

## Publication Policy

- Branch policy: `in-place-publication`.
- If `publication_branch` is set and `git_commit` / `git_push` are enabled, publish the canonical planner docs directly on that branch.
- Do not stage or commit unrelated repo changes.
- If the repo is dirty or not on the canonical publication branch, stop with `BLOCKED` instead of improvising around it.

## Executor Instructions

- Read this envelope completely before acting.
- Respect the `capabilities` block in the YAML header.
- If a required capability is false, do not perform that action even if prose could be interpreted otherwise.
- If requirements are ambiguous, stop with `BLOCKED` and explain the minimum missing decision.
- If this is a write phase with declared feature branch/worktree metadata, use that policy unless explicitly overridden by Hermes/human.
- For Python repos, use the declared `shared_python` interpreter for tests/tooling when possible. In QuantAgent, worktrees should reuse the shared venv instead of creating a per-worktree `.venv`.
- End with an output envelope/report containing: summary, files changed, commands run, quality gates, artifacts, risks, next step, and Beads update.

## Required Final Beads Comment Template

```md
### Skill: autodev-planner
- **Resultado:** SUCCESS | PARTIAL | BLOCKED | FAIL
- **Run-ID:** 20260525T204955Z-QuantAgent-kkj.2-planner
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
