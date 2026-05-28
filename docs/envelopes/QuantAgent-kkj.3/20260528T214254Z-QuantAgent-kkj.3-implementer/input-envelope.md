---
run_id: "20260528T214254Z-QuantAgent-kkj.3-implementer"
phase: "implementer"
executor: "auto"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-kkj.3"
branch_policy: "worktree-preferred"
base_branch: "main"
publication_branch: null
current_branch_at_generation: "main"
feature_branch: "feature/QuantAgent-kkj.3-ux-redise-ar-dashboard-para-ser-environm"
worktree_root: "/tmp/autodev-worktrees/QuantAgent"
worktree_path: "/tmp/autodev-worktrees/QuantAgent/QuantAgent-kkj.3/implementer"
shared_venv: "/home/azureuser/repos/projects/QuantAgent/.venv"
shared_python: "/home/azureuser/repos/projects/QuantAgent/.venv/bin/python"
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
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-kkj.3/20260528T214254Z-QuantAgent-kkj.3-implementer"
generated_at: "2026-05-28T21:42:54.108232+00:00"
preflight:
  repo_dirty: false
  dirty_files: []
---

# Autodev Input Envelope — QuantAgent-kkj.3 — implementer

## Objective

Execute the `implementer` phase for Beads issue `QuantAgent-kkj.3`: [UX] Rediseñar Dashboard para ser environment-aware con selector de corridas.

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
- Beads issue: `QuantAgent-kkj.3`.
- Labels at generation time: `none`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-kkj.3/20260528T214254Z-QuantAgent-kkj.3-implementer/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

## Contexto

El Dashboard actual de Streamlit tiene un selector de environment (paper/backtest) en la parte superior, pero las pestañas de Paper Trading y Backtesting siempre están presentes independientemente de la selección. El contenido principal del dashboard no cambia según el environment elegido.

Observado durante revisión funcional M2 (2026-05-25):
- El selector paper/backtest del tope no produce cambios visuales evidentes en el dashboard.
- Las pestañas siempre presentes generan la expectativa de que son independientes del contexto.
- No hay un flujo claro: seleccionar environment → ver corridas de ese tipo → seleccionar corrida → ver estadísticas.
- El indicador de estado del scheduler paper trading es valioso y debe mantenerse visible.

## Cambio requerido

Hacer el Dashboard environment-aware:

- **Modo paper**: mostrar una grilla de corridas de paper trading con un selector para ver estadísticas de una corrida específica. Mantener el indicador de estado del scheduler (running/stuck/error/stopped).
- **Modo backtest**: mostrar la grilla de backtest runs con un selector para ver estadísticas de un run específico. Mostrar indicador si hay un backtest activo.
- Las métricas/estadísticas se muestran al seleccionar una corrida específica, no como pantalla fija completa.

## Criterio de aceptación

- [ ] Al seleccionar environment "paper", el dashboard muestra corridas de paper trading y el estado del scheduler.
- [ ] Al seleccionar environment "backtest", el dashboard muestra backtest runs.
- [ ] Al seleccionar una corrida de la grilla, se muestran sus estadísticas/métricas.
- [ ] El indicador de estado del scheduler paper (running/stuck/error/stopped) es siempre visible en modo paper.
- [ ] El indicador de backtest activo es visible en modo backtest cuando hay un run corriendo.

## Archivos relevantes

- `apps/streamlit/views/dashboard.py` — vista a rediseñar.
- `apps/streamlit/app.py` — routing del environment selector.
- `apps/streamlit/views/paper_trading.py` — referencia del scheduler state widget.

## Fuera de scope (no tocar)

- No rediseñar Configuration ni otras pestañas.
- No implementar métricas que no existen aún en el modelo de datos.
- No cambiar la lógica de paper trading ni backtesting, solo la presentación.
- No eliminar las pestañas dedicadas de Paper Trading y Backtesting; solo mejorar el dashboard principal.


## Acceptance Criteria

No acceptance criteria found in Beads issue.

## Recent Beads Comments

- Comment 243 by Federico Scheu at 2026-05-28T21:41:58Z:

  ### Skill: autodev-planner
  - **Resultado:** SUCCESS
  - **Run-ID:** 20260528T213653Z-QuantAgent-kkj.3-planner
  - **Executor:** auto
  
  #### Qué hice
  - Verifiqué los drafts de RQ y PL existentes contra el código fuente real
  - Confirmé existencia de todos los métodos DB referenciados y campos del modelo BacktestRun
  - Actualicé los README de docs/01_requirements/ y docs/02_planning/ con entradas para kkj.3
  - Creé artifacts de run (result.json, run-report.md) en docs/envelopes/
  
  #### Evidencia
  - `db.get_latest_heartbeat`, `db.get_recent_heartbeats`: ✅ confirmados en db.py:17,59
  - `BacktestRun.total_trades/win_rate/profit_factor/sharpe_ratio/max_drawdown/total_pnl`: ✅ confirmados en models.py:310-315
  - `app.py` routing: ✅ correcto sin cambio necesario
  - `python -m compileall` sobre archivos afectados: ✅ sin errores
  
  #### Quality gates
  - git status limpio antes de publicación: ✅
  - Issue ID en paths de docs: ✅
  - ACs testeables: ✅ (5 criterios concretos y verificables)
  - Branch = main (publication branch): ✅
  - compileall: ✅
  
  #### Problemas encontrados
  - Ninguno bloqueante. Nota menor: `_calculate_status` en dashboard.py tiene lógica simplificada respecto a paper_trading.py (no distingue Running…

## Preflight Evidence

- Current branch at generation: `main`.
- Canonical publication branch: `n/a`.
- Repo dirty at generation: `False`.
- Dirty files:
  - none

## Publication Policy

- Branch policy: `worktree-preferred`.
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
### Skill: autodev-implementer
- **Resultado:** SUCCESS | PARTIAL | BLOCKED | FAIL
- **Run-ID:** 20260528T214254Z-QuantAgent-kkj.3-implementer
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
