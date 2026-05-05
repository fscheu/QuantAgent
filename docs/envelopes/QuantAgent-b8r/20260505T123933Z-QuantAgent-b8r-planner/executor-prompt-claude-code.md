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
        - Run ID: 20260505T123933Z-QuantAgent-b8r-planner
        - Phase: planner
        - Skill: autodev-planner
        - Issue: QuantAgent-b8r
        - Repo: /home/azureuser/repos/projects/QuantAgent
        - Artifacts dir: /home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-b8r/20260505T123933Z-QuantAgent-b8r-planner

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
        run_id: "20260505T123933Z-QuantAgent-b8r-planner"
phase: "planner"
executor: "claude-code"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-b8r"
branch_policy: "worktree-preferred"
base_branch: "main"
current_branch_at_generation: "main"
feature_branch: "feature/QuantAgent-b8r-m1-strategy-3-52-week-high-momentum-brea"
worktree_path: "/tmp/autodev-worktrees/QuantAgent/QuantAgent-b8r/planner-20260505T123933Z"
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
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-b8r/20260505T123933Z-QuantAgent-b8r-planner"
generated_at: "2026-05-05T12:39:33.604814+00:00"
preflight:
  repo_dirty: true
  dirty_files:
    - "?? .beads/.doctor-test-write"
    - "?? docs/envelopes/QuantAgent-vna/"
        ---

# Autodev Input Envelope — QuantAgent-b8r — planner

## Objective

Execute the `planner` phase for Beads issue `QuantAgent-b8r`: M1 Strategy 3 — 52-week high momentum / breakout para equities US.

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
- Beads issue: `QuantAgent-b8r`.
- Labels at generation time: `backtesting, equities, m1, openclaw:design_pending, strategy`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-b8r/20260505T123933Z-QuantAgent-b8r-planner/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

## Contexto
M1 necesita una tercera estrategia equity-specific con edge real documentado para acciones de US. La elegida es **52-week high momentum / breakout**, basada en la evidencia de George y Hwang (2004), porque agrega diversidad real respecto a Triple Screen y al pipeline multi-agente, y se puede implementar con datos OHLCV estándar.

## Cambio requerido
Implementar una estrategia long-biased de 52-week high momentum / breakout para equities US sobre la abstracción `TradingStrategy`.

Versión M1 esperada:
- usa timeframe diario;
- detecta cercanía o breakout del máximo de 52 semanas;
- aplica filtros simples y explícitos de tendencia y volumen para evitar ruido;
- usa reglas de salida simples (por ejemplo trailing stop y/o pérdida de filtro de tendencia) sin meter datasets adicionales.

## Criterio de aceptación
- [ ] Existe una estrategia dedicada en `quantagent/strategy/` para 52-week high momentum / breakout
- [ ] La estrategia calcula correctamente el máximo rolling de 52 semanas y su señal de proximidad o breakout
- [ ] Incluye filtros mínimos configurables de tendencia y/o volumen apropiados para M1
- [ ] Hay tests determinísticos para cálculo de 52-week high, señales de entrada y reglas de salida
- [ ] Un backtest de referencia sobre equities US completa sin crashes y con PnL calculado
- [ ] La versión M1 queda documentada como long-only salvo que el planner justifique otra cosa sin ampliar scope

## Archivos relevantes
- `quantagent/strategy/base.py`
- `quantagent/strategy/assembler.py`
- `quantagent/data/provider.py`
- `quantagent/backtesting/backtest.py`
- `tests/`

## Fuera de scope
- PEAD o earnings surprise datasets
- Estrategias intradía dependientes de microestructura
- Short-selling sofisticado o borrow constraints
- Optimización cuantitativa avanzada del universo

## Notas técnicas
- Este ticket es uno de los 3 hijos que completan AC3 de `QuantAgent-l0h`.
- Mantenerlo implementable con OHLCV diario y filtros simples; si requiere datos o eventos extra, deja de servir como referencia limpia para M1.
- Referencia conceptual: George y Hwang, "The 52-Week High and Momentum Investing" (Journal of Finance, 2004).


## Acceptance Criteria

No acceptance criteria found in Beads issue.

## Recent Beads Comments

- No recent comments captured.

## Preflight Evidence

- Current branch at generation: `main`.
- Repo dirty at generation: `True`.
- Dirty files:
  - `?? .beads/.doctor-test-write`
  - `?? docs/envelopes/QuantAgent-vna/`

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
- **Run-ID:** 20260505T123933Z-QuantAgent-b8r-planner
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
