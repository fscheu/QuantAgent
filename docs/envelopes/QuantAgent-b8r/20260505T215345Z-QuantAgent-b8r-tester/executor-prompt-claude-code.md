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
        - Run ID: 20260505T215345Z-QuantAgent-b8r-tester
        - Phase: tester
        - Skill: autodev-tester
        - Issue: QuantAgent-b8r
        - Repo: /tmp/autodev-worktrees/QuantAgent/techlead-20260505T213844Z
        - Artifacts dir: /tmp/autodev-worktrees/QuantAgent/techlead-20260505T213844Z/docs/envelopes/QuantAgent-b8r/20260505T215345Z-QuantAgent-b8r-tester

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
        run_id: "20260505T215345Z-QuantAgent-b8r-tester"
phase: "tester"
executor: "claude-code"
repo_path: "/tmp/autodev-worktrees/QuantAgent/techlead-20260505T213844Z"
beads_issue_id: "QuantAgent-b8r"
branch_policy: "worktree-preferred"
base_branch: "main"
current_branch_at_generation: "integration/quantagent-cron-20260505T213844Z"
feature_branch: "feature/QuantAgent-b8r-m1-strategy-3-52-week-high-momentum-brea"
worktree_path: "/tmp/autodev-worktrees/techlead-20260505T213844Z/QuantAgent-b8r/tester-20260505T215345Z"
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
artifacts_dir: "/tmp/autodev-worktrees/QuantAgent/techlead-20260505T213844Z/docs/envelopes/QuantAgent-b8r/20260505T215345Z-QuantAgent-b8r-tester"
generated_at: "2026-05-05T21:53:45.510832+00:00"
preflight:
  repo_dirty: true
  dirty_files:
    - "?? docs/envelopes/QuantAgent-b8r/20260505T213930Z-QuantAgent-b8r-implementer/"
    - "?? docs/envelopes/QuantAgent-vna/20260505T213928Z-QuantAgent-vna-tester/"
        ---

# Autodev Input Envelope — QuantAgent-b8r — tester

## Objective

Execute the `tester` phase for Beads issue `QuantAgent-b8r`: M1 Strategy 3 — 52-week high momentum / breakout para equities US.

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

- Repo instructions: `/tmp/autodev-worktrees/QuantAgent/techlead-20260505T213844Z/AGENTS.md` and `/tmp/autodev-worktrees/QuantAgent/techlead-20260505T213844Z/CLAUDE.md` if present.
- Beads issue: `QuantAgent-b8r`.
- Labels at generation time: `backtesting, equities, m1, openclaw:design_approved, strategy`.
- Artifact issue snapshot: `/tmp/autodev-worktrees/QuantAgent/techlead-20260505T213844Z/docs/envelopes/QuantAgent-b8r/20260505T215345Z-QuantAgent-b8r-tester/issue.json`.
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

- Comment 107 by Federico Scheu at 2026-05-05T12:50:40Z:

  ### Skill: autodev-planner
  - **Resultado:** SUCCESS
  - **Run-ID:** 20260505T123933Z-QuantAgent-b8r-planner
  - **Executor:** auto

  #### Qué hice
  - Leí el issue, AGENTS.md, código de estrategias existentes (RSI, base.py), y los artifacts de QuantAgent-vna como referencia
  - Produje los tres artifacts de planning: RQ, DS, AC para la estrategia 52-week high momentum/breakout
  - Actualicé los README index de docs/01_requirements/, docs/03_design/, docs/05_acceptance_tests/

  #### Evidencia
  - `docs/01_requirements/QuantAgent-b8r-RQ-52week-high-momentum.md` — 12 FRs, incluyendo justificación long-only (FR12)
  - `docs/03_design/QuantAgent-b8r-DS-52week-high-momentum.md` — clase `FiftyTwoWeekHighStrategy`, algoritmo completo, perfil de backtest de referencia
  - `docs/05_acceptance_tests/QuantAgent-b8r-AC-52week-high-momentum.md` — 10 ACs con comandos pytest explícitos

  #### Quality gates
  - git status --short: PASS
  - Issue ID en paths de docs: PASS (QuantAgent-b8r en los 3 paths)
  - Acceptance criteria testables: PASS (todos con pytest commands)

  #### Problemas encontrados
  - Ninguno. Sin ambigüedades que requieran bloqueo.

  #### Next step recomendado
  - **Implementer phase**: implementar `FiftyTwoWee…
- Comment 109 by Federico Scheu at 2026-05-05T12:53:19Z:

  ### Skill: tech-lead-autodev
  - **Resultado:** PARTIAL
  - **Run-ID:** 20260505T123933Z-QuantAgent-b8r-techlead

  #### Qué hice
  - Revisé el output del planner `20260505T123933Z-QuantAgent-b8r-planner` y verifiqué los artifacts en `docs/envelopes/QuantAgent-b8r/20260505T123933Z-QuantAgent-b8r-planner/`.
  - Confirmé que el ticket queda listo para `implementer`.
  - Voy a persistir los docs/artifacts en la branch `feature/quantagent-m1-strategy-planning-20260505` para evitar push innecesario a `main` y no disparar un redeploy por docs-only.

  #### Next step recomendado
  - Implementer sobre la branch `feature/quantagent-m1-strategy-planning-20260505` o cherry-pick del commit de planning a la branch de implementación del ticket.
- Comment 112 by Federico Scheu at 2026-05-05T21:53:11Z:

  ### Skill: autodev-implementer
  - **Resultado:** SUCCESS
  - **Run-ID:** 20260505T213930Z-QuantAgent-b8r-implementer
  - **Executor:** auto

  #### Qué hice
  - Leí el diseño (QuantAgent-b8r-DS-52week-high-momentum.md) y los ACs
  - Implementé `FiftyTwoWeekHighStrategy` en `quantagent/strategy/fifty_two_week_high_strategy.py`
  - Actualicé `quantagent/strategy/__init__.py` con el export de la clase
  - Creé `docs/06_implementation/QuantAgent-b8r-IM-52week-high-momentum.md`
  - Actualicé `docs/06_implementation/README.md` con el link al nuevo artifact

  #### Evidencia
  - Commit `6e0e5ed5`: feat(QuantAgent-b8r): implement FiftyTwoWeekHighStrategy (M1 Strategy 3)
  - Commit `d6de9621`: docs(QuantAgent-b8r): add implementation doc and README index entry
  - Branch: `feature/QuantAgent-b8r-m1-strategy-3-52-week-high-momentum-brea`
  - La clase subclasea `TradingStrategy`, genera señales LONG con breakout de 52w high, filtros SMA-50 y volumen

  #### Quality gates
  - git status --short: PASS
  - ruff check --fix (changed files): PASS
  - python -m compileall -q .: PASS
  - pytest: BLOCKED_PREEXISTING_ENV (talib/langchain_anthropic no instalados en el worktree; misma falla en test_rsi_strategy.py pre-existente)

  #### Prob…

## Preflight Evidence

- Current branch at generation: `integration/quantagent-cron-20260505T213844Z`.
- Repo dirty at generation: `True`.
- Dirty files:
  - `?? docs/envelopes/QuantAgent-b8r/20260505T213930Z-QuantAgent-b8r-implementer/`
  - `?? docs/envelopes/QuantAgent-vna/20260505T213928Z-QuantAgent-vna-tester/`

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
- **Run-ID:** 20260505T215345Z-QuantAgent-b8r-tester
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
