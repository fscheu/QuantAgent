---
run_id: "20260525T205529Z-QuantAgent-kkj.8-planner"
phase: "planner"
executor: "auto"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-kkj.8"
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
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-kkj.8/20260525T205529Z-QuantAgent-kkj.8-planner"
generated_at: "2026-05-25T20:55:29.502311+00:00"
preflight:
  repo_dirty: false
  dirty_files: []
---

# Autodev Input Envelope — QuantAgent-kkj.8 — planner

## Objective

Execute the `planner` phase for Beads issue `QuantAgent-kkj.8`: Crear strategy registry y parametrizar scheduler para selección de estrategia.

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
- Beads issue: `QuantAgent-kkj.8`.
- Labels at generation time: `none`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-kkj.8/20260525T205529Z-QuantAgent-kkj.8-planner/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

## Contexto

QuantAgent tiene 4 estrategias implementadas en `quantagent/strategy/`:
- `LLMAgentStrategy` — wrapper del pipeline multi-agente LangGraph (requiere LLM, tiene costo de tokens).
- `RSIMeanReversionStrategy` — determinista, sin LLM. Params: rsi_period=14, oversold=30, overbought=70, stop_loss_pct, take_profit_pct, trailing_stop_pct.
- `FiftyTwoWeekHighStrategy` — determinista, sin LLM. Params: lookback_days=252, trend_ma_period=50, volume_ma_period=20.
- `TripleScreenStrategy` — determinista, sin LLM. Params: trend_ema_period=13, stoch_k_period=5, stoch_d_period=3.

Sin embargo, `TradingScheduler` está hardcodeado para usar siempre `LLMAgentStrategy`:
```python
# quantagent/trading/scheduler.py:65
self.strategy = strategy or LLMAgentStrategy(self.trading_graph)
```

No existe un registry central que mapee nombres de estrategia a sus clases y parámetros configurables. El script de piloto `scripts/run_paper_pilot.py` instancia RSI y 52wHigh directamente, pero ese es un script ad-hoc, no el scheduler productivo.

Sin este registry, no es posible exponer selección de estrategia en la UI (Backtesting, Paper Trading, Configuration) porque no hay una fuente de verdad de qué estrategias existen y qué parámetros aceptan.

## Cambio requerido

Crear un **strategy registry** como fuente de verdad para la selección de estrategias en todo el sistema, y actualizar el scheduler para aceptar una estrategia configurada externamente.

Esto implica:

1. **`quantagent/strategy/registry.py`** (nuevo): diccionario o función `get_strategy_registry()` que mapea cada estrategia a:
   - nombre de clase y clase Python
   - tipo (`deterministic` | `llm`)
   - parámetros configurables con nombre, tipo, default y descripción breve
   - requerimientos de datos (ej: bars mínimos necesarios)

2. **`TradingScheduler`**: aceptar `strategy: TradingStrategy` como parámetro de construcción (ya existe el slot pero se ignora para LLM). Eliminar la dependencia hardcodeada a `LLMAgentStrategy` cuando se pasa una estrategia explícita.

3. **`quantagent/strategy/base.py`**: agregar opcionalmente un método `@classmethod describe()` en `TradingStrategy` que retorne nombre display, tipo, y descripción corta — para que el registry no sea solo declaración externa sino que las estrategias puedan auto-describirse.

## Criterio de aceptación

- [ ] Existe `quantagent/strategy/registry.py` con las 4 estrategias registradas.
- [ ] El registry expone al menos: nombre (string), clase Python, tipo (deterministic/llm), parámetros configurables (nombre, tipo, default).
- [ ] `TradingScheduler` acepta cualquier `TradingStrategy` (no solo `LLMAgentStrategy`) y la usa correctamente.
- [ ] Tests cubren: instanciar el scheduler con RSIMeanReversionStrategy y verificar que genera señales con esa estrategia.
- [ ] El registry es importable desde `quantagent.strategy` sin side effects.

## Archivos relevantes

- `quantagent/strategy/__init__.py` — punto de export; agregar registry.
- `quantagent/strategy/registry.py` — nuevo archivo.
- `quantagent/strategy/base.py` — agregar describe() si aplica.
- `quantagent/trading/scheduler.py:55-65` — hardcoding a LLMAgentStrategy a eliminar.
- `scripts/run_paper_pilot.py` — referencia de cómo se instancian las estrategias hoy ad-hoc.

## Fuera de scope (no tocar)

- No modificar la lógica interna de ninguna estrategia existente.
- No agregar selectores en la UI (eso va en el ticket de UI correspondiente).
- No crear nuevas estrategias.
- No cambiar el modelo de DB.

## Notas técnicas

El registry puede ser tan simple como un dict:
```python
STRATEGY_REGISTRY = {
    "RSIMeanReversionStrategy": {
        "cls": RSIMeanReversionStrategy,
        "type": "deterministic",
        "params": {
            "rsi_period": {"type": int, "default": 14, "description": "RSI calculation period"},
            "oversold_threshold": {"type": float, "default": 30.0},
            "overbought_threshold": {"type": float, "default": 70.0},
            ...
        }
    },
    ...
}
```

La UI necesitará este registry para generar dinámicamente los controles de configuración de parámetros por estrategia.


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
- **Run-ID:** 20260525T205529Z-QuantAgent-kkj.8-planner
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
