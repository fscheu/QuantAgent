---
run_id: "20260526T150428Z-QuantAgent-kkj.9-planner"
phase: "planner"
executor: "auto"
repo_path: "/tmp/quantagent-techlead-20260526T150148Z"
beads_issue_id: "QuantAgent-kkj.9"
branch_policy: "in-place-publication"
base_branch: "main"
publication_branch: "main"
current_branch_at_generation: "main"
feature_branch: null
worktree_root: null
worktree_path: null
shared_venv: null
shared_python: "/home/azureuser/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11"
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
artifacts_dir: "/tmp/quantagent-techlead-20260526T150148Z/docs/envelopes/QuantAgent-kkj.9/20260526T150428Z-QuantAgent-kkj.9-planner"
generated_at: "2026-05-26T15:04:28.565092+00:00"
preflight:
  repo_dirty: false
  dirty_files: []
---

# Autodev Input Envelope — QuantAgent-kkj.9 — planner

## Objective

Execute the `planner` phase for Beads issue `QuantAgent-kkj.9`: Agregar selector de estrategia en UI (Backtesting, Paper Trading, Configuration).

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

- Repo instructions: `/tmp/quantagent-techlead-20260526T150148Z/AGENTS.md` and `/tmp/quantagent-techlead-20260526T150148Z/CLAUDE.md` if present.
- Beads issue: `QuantAgent-kkj.9`.
- Labels at generation time: `none`.
- Artifact issue snapshot: `/tmp/quantagent-techlead-20260526T150148Z/docs/envelopes/QuantAgent-kkj.9/20260526T150428Z-QuantAgent-kkj.9-planner/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

## Contexto

QuantAgent tiene 4 estrategias implementadas (`LLMAgentStrategy`, `RSIMeanReversionStrategy`, `FiftyTwoWeekHighStrategy`, `TripleScreenStrategy`) pero ninguna vista de Streamlit permite seleccionar cuál usar. La distinción clave es:

- **Estrategias deterministas** (RSI, 52wHigh, TripleScreen): sin LLM, sin costo de tokens, configurables por parámetros numéricos. Son las estrategias de referencia M1.
- **LLMAgentStrategy**: pipeline multi-agente LangGraph, requiere LLM y tiene costo de tokens. Es la estrategia original del MVP.

Hoy:
- En **Backtesting**: el form tiene selector de "Model preset" pero no de estrategia. Al agregar wiring de ejecución real, sin selector de estrategia no se puede elegir qué lógica corre.
- En **Paper Trading**: el scheduler corre hardcodeado con LLMAgentStrategy (el ticket `kkj.2` agrega controles de start/stop, pero necesita selector de estrategia para poder configurar qué corre).
- En **Configuration**: no existe sección de strategy presets ni defaults de estrategia por environment.

Este ticket depende de `QuantAgent-kkj.8` (strategy registry) que provee el catalog de estrategias disponibles y sus parámetros.

## Cambio requerido

Agregar selectores de estrategia en los tres puntos de configuración:

### 1. Backtesting form (`apps/streamlit/views/backtesting.py`)
- Agregar `st.selectbox("Estrategia", strategy_names)` en el form de creación de run.
- Si la estrategia seleccionada tiene parámetros configurables (del registry), mostrar los controles correspondientes debajo del selector (ej: rsi_period, oversold_threshold).
- Guardar la estrategia seleccionada y sus parámetros en `config_snapshot` del `BacktestRun`.
- Diferenciar visualmente si la estrategia requiere LLM (mostrar advertencia de costo de tokens) o es determinista.

### 2. Paper Trading — Start form (`apps/streamlit/views/paper_trading.py`)
- En el form de inicio del scheduler (kkj.2), agregar selector de estrategia con sus parámetros.
- El label/model preset de LLM solo debe mostrarse si la estrategia seleccionada es tipo `llm`.
- El estado del scheduler debe mostrar qué estrategia está corriendo en la corrida activa.

### 3. Configuration (`apps/streamlit/views/configuration.py`)
- Agregar una sección "Strategy Defaults" (dentro de Portfolio & Universe o como sección separada) donde el operador pueda definir:
  - Estrategia default para paper trading.
  - Estrategia default para backtesting.
- Estos defaults deben pre-seleccionar la estrategia en los forms de Backtesting y Paper Trading.

## Criterio de aceptación

- [ ] En el form de Backtesting, hay un selector de estrategia con las 4 estrategias disponibles.
- [ ] Al seleccionar una estrategia determinista, se ocultan o deshabilitan los campos de LLM preset.
- [ ] Al seleccionar una estrategia determinista con parámetros, se muestran controles de configuración de esos parámetros.
- [ ] Al seleccionar `LLMAgentStrategy`, se muestra una advertencia de "requiere LLM (costo de tokens)".
- [ ] La estrategia seleccionada y sus parámetros se guardan en `config_snapshot` del BacktestRun.
- [ ] En Paper Trading, el form de Start del scheduler (kkj.2) incluye selector de estrategia.
- [ ] En Configuration, el operador puede definir estrategias default por environment.
- [ ] El selector se construye a partir del registry (kkj.8), no con valores hardcodeados en la UI.

## Archivos relevantes

- `apps/streamlit/views/backtesting.py` — agregar selector en el form de creación.
- `apps/streamlit/views/paper_trading.py` — agregar selector en el form de start del scheduler.
- `apps/streamlit/views/configuration.py` — agregar sección Strategy Defaults.
- `quantagent/strategy/registry.py` — fuente de verdad de estrategias y parámetros (depende de kkj.8).

## Fuera de scope (no tocar)

- No implementar el wiring de ejecución de backtesting (ese es un gap separado).
- No crear nuevas estrategias.
- No cambiar la lógica interna de las estrategias.
- No rediseñar el layout general de las vistas más allá de agregar el selector.

## Notas técnicas

- El selector debe construirse dinámicamente desde el registry para que al agregar una nueva estrategia en el futuro, la UI la muestre automáticamente sin cambios manuales.
- Para la generación dinámica de parámetros: usar `st.number_input` para int/float, `st.slider` si hay range conocido, `st.text_input` para strings.
- Los parámetros de estrategia deben guardarse como parte de `config_snapshot` en BacktestRun y como parte del payload de start del scheduler.


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
- **Run-ID:** 20260526T150428Z-QuantAgent-kkj.9-planner
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
