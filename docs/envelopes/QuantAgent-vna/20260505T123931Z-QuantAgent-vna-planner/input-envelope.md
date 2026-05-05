---
run_id: "20260505T123931Z-QuantAgent-vna-planner"
phase: "planner"
executor: "auto"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-vna"
branch_policy: "worktree-preferred"
base_branch: "main"
current_branch_at_generation: "main"
feature_branch: "feature/QuantAgent-vna-m1-strategy-1-triple-screen-strategy-ale"
worktree_path: "/tmp/autodev-worktrees/QuantAgent/QuantAgent-vna/planner-20260505T123931Z"
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
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-vna/20260505T123931Z-QuantAgent-vna-planner"
generated_at: "2026-05-05T12:39:31.116593+00:00"
preflight:
  repo_dirty: true
  dirty_files:
    - "?? .beads/.doctor-test-write"
---

# Autodev Input Envelope — QuantAgent-vna — planner

## Objective

Execute the `planner` phase for Beads issue `QuantAgent-vna`: M1 Strategy 1 — Triple Screen Strategy (Alexander Elder).

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
- Beads issue: `QuantAgent-vna`.
- Labels at generation time: `backtesting, enhancement, m1, openclaw:design_pending, strategy`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-vna/20260505T123931Z-QuantAgent-vna-planner/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

## Contexto
M1 requiere 3 estrategias de referencia explícitas para declarar backtesting estable. Este ticket pasa a ser la **Estrategia 1 oficial de M1**: Triple Screen de Alexander Elder. Ya existe como idea en backlog, pero necesita un contrato ejecutable para que planner, implementer y tester la puedan tomar con claridad.

## Cambio requerido
Implementar una versión M1 de Triple Screen sobre la abstracción `TradingStrategy`, integrada al flujo de backtesting actual.

La implementación debe preservar la lógica de 3 pantallas:
1. Pantalla 1: filtro de tendencia principal en timeframe superior
2. Pantalla 2: detección de pullback o corrección contra la tendencia en timeframe intermedio
3. Pantalla 3: trigger de entrada en timeframe inferior o señal equivalente soportada por los datos disponibles

El planner puede precisar indicadores concretos, pero debe respetar la semántica Triple Screen y usar datos e infraestructura ya soportados por el repo.

## Criterio de aceptación
- [ ] Existe una estrategia dedicada para Triple Screen integrada en `quantagent/strategy/`
- [ ] La estrategia expone parámetros configurables mínimos (timeframes, indicadores o thresholds necesarios)
- [ ] Hay tests determinísticos para la lógica de tendencia, pullback y trigger
- [ ] Un backtest de referencia completa sin crashes y con PnL calculado
- [ ] La estrategia genera señales reales (no solo HOLD permanente) en un escenario de prueba razonable
- [ ] Queda documentado qué combinación de timeframes e indicadores representa la versión M1

## Archivos relevantes
- `quantagent/strategy/base.py`
- `quantagent/strategy/assembler.py`
- `quantagent/backtesting/backtest.py`
- `quantagent/data/provider.py`
- `tests/`

## Fuera de scope
- Optimización de parámetros por activo
- Variantes avanzadas de Elder no necesarias para M1
- Comparativas extensivas de performance versus todas las estrategias del repo

## Notas técnicas
- Este ticket es uno de los 3 hijos que completan AC3 de `QuantAgent-l0h`.
- Reusar abstractions existentes antes de introducir nuevas capas.
- Mantener el diseño minimalista: versión M1 estable y testeable, no una plataforma genérica de research.


## Acceptance Criteria

No acceptance criteria found in Beads issue.

## Recent Beads Comments

- Comment 57 by Federico Scheu at 2026-04-18T19:05:08Z:

  Blocker QuantAgent-enn está cerrado. Puede entrar al backlog activo, aunque sigue siendo P4 (feature de estrategia, no crítico).

## Preflight Evidence

- Current branch at generation: `main`.
- Repo dirty at generation: `True`.
- Dirty files:
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
- **Run-ID:** 20260505T123931Z-QuantAgent-vna-planner
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
