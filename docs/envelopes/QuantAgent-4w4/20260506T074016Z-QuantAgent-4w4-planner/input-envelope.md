---
run_id: "20260506T074016Z-QuantAgent-4w4-planner"
phase: "planner"
executor: "auto"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-4w4"
branch_policy: "worktree-preferred"
base_branch: "main"
current_branch_at_generation: "main"
feature_branch: "feature/QuantAgent-4w4-backtest-must-honor-strategy-specific-lo"
worktree_path: "/tmp/autodev-worktrees/QuantAgent/QuantAgent-4w4/planner-20260506T074016Z"
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
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-4w4/20260506T074016Z-QuantAgent-4w4-planner"
generated_at: "2026-05-06T07:40:16.576503+00:00"
preflight:
  repo_dirty: true
  dirty_files:
    - " M .beads/issues.jsonl"
    - "?? .beads/.doctor-test-write"
---

# Autodev Input Envelope — QuantAgent-4w4 — planner

## Objective

Execute the `planner` phase for Beads issue `QuantAgent-4w4`: Backtest must honor strategy-specific lookback windows.

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
- Beads issue: `QuantAgent-4w4`.
- Labels at generation time: `none`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-4w4/20260506T074016Z-QuantAgent-4w4-planner/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

## Contexto
Durante la verificación de integración de `QuantAgent-b8r` se ejecutó el backtest de referencia real para AAPL entre 2022-01-01 y 2023-12-31 en un worktree aislado y contra PostgreSQL local. La estrategia `FiftyTwoWeekHighStrategy` quedó implementada y sus tests pasan, pero el backtest no puede entregarle el historial que necesita para una señal de 52-week high.

## Hallazgo verificable
`quantagent/backtesting/backtest.py` usa un lookback fijo de 30 días en `_analyze_and_trade()`:
- `lookback_days = 30`
- `data_start = current_date - timedelta(days=lookback_days)`

Para una estrategia que requiere ~252 ruedas diarias, el engine solo entrega ~21-22 velas y loguea `Insufficient data` en cada fecha. El backtest completa sin crash y con PnL finito, pero genera `TOTAL_TRADES = 0` por una limitación del engine, no por una validación real de la estrategia.

## Cambio requerido
Hacer que el backtest pida una ventana histórica suficiente para la estrategia activa en vez de usar 30 días hardcodeados.

Opciones válidas de implementación (elegir la mínima):
1. Exponer en `TradingStrategy` / estrategia concreta un requerimiento de historia mínima (`required_history_bars` o equivalente) y usarlo en Backtest.
2. Resolver el lookback desde config/assembler con un valor explícito para estrategias daily de horizonte largo.

## Criterio de aceptación
- [ ] `Backtest._analyze_and_trade()` deja de usar 30 días hardcodeados para todas las estrategias.
- [ ] La estrategia de 52-week high recibe suficiente historia diaria para evaluar el máximo de 52 semanas.
- [ ] El backtest de referencia AAPL 2022-01-01 → 2023-12-31 deja de emitir warnings sistemáticos de `Insufficient data` por ventana insuficiente.
- [ ] Existe evidencia automatizada o reproducible de que el engine entrega >252 barras cuando la estrategia lo requiere.
- [ ] `QuantAgent-b8r` queda desbloqueado para decisión final de integración.

## Archivos relevantes
- `quantagent/backtesting/backtest.py`
- `quantagent/strategy/base.py`
- `quantagent/strategy/fifty_two_week_high_strategy.py`
- `tests/`

## Fuera de scope
- Cambiar la lógica de señal de `FiftyTwoWeekHighStrategy`.
- Optimizar performance del proveedor de datos.
- Introducir datasets nuevos o premium.

## Acceptance Criteria

No acceptance criteria found in Beads issue.

## Recent Beads Comments

- No recent comments captured.

## Preflight Evidence

- Current branch at generation: `main`.
- Repo dirty at generation: `True`.
- Dirty files:
  - ` M .beads/issues.jsonl`
  - `?? .beads/.doctor-test-write`

## Executor Instructions

- Read this envelope completely before acting.
- Respect the `capabilities` block in the YAML header.
- If a required capability is false, do not perform that action even if prose could be interpreted otherwise.
- If requirements are ambiguous, stop with `BLOCKED` and explain the minimum missing decision.
- If this is a write phase, use the declared feature branch/worktree policy unless explicitly overridden by Hermes/human.
- End with an output envelope/report containing: summary, files changed, commands run, quality gates, artifacts, risks, next step, and Beads update.

## Required Final Beads Comment Template

```md
### Skill: autodev-planner
- **Resultado:** SUCCESS | PARTIAL | BLOCKED | FAIL
- **Run-ID:** 20260506T074016Z-QuantAgent-4w4-planner
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
