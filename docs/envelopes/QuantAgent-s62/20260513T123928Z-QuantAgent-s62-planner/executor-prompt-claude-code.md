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
        - Run ID: 20260513T123928Z-QuantAgent-s62-planner
        - Phase: planner
        - Skill: autodev-planner
        - Issue: QuantAgent-s62
        - Repo: /home/azureuser/repos/projects/QuantAgent
        - Worktree: /mnt/actions-runner/autodev-runtime/worktrees/QuantAgent/QuantAgent-s62/planner-20260513T123928Z
        - Artifacts dir: /home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-s62/20260513T123928Z-QuantAgent-s62-planner
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
        run_id: "20260513T123928Z-QuantAgent-s62-planner"
phase: "planner"
executor: "claude-code"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-s62"
branch_policy: "worktree-preferred"
base_branch: "main"
current_branch_at_generation: "main"
feature_branch: "feature/QuantAgent-s62-extender-observabilidad-operativa-m-nima"
worktree_root: "/mnt/actions-runner/autodev-runtime/worktrees/QuantAgent"
worktree_path: "/mnt/actions-runner/autodev-runtime/worktrees/QuantAgent/QuantAgent-s62/planner-20260513T123928Z"
shared_venv: "/mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv"
shared_python: "/mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python"
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
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-s62/20260513T123928Z-QuantAgent-s62-planner"
generated_at: "2026-05-13T12:39:28.414095+00:00"
preflight:
  repo_dirty: true
  dirty_files:
    - " M .beads/issues.jsonl"
    - "?? docs/01_requirements/QuantAgent-375-RQ-scope-replay-signal-lookup.md"
    - "?? docs/02_planning/QuantAgent-375-PL-scope-replay-signal-lookup.md"
    - "?? docs/03_design/QuantAgent-375-DS-scope-replay-signal-lookup.md"
    - "?? docs/05_acceptance_tests/QuantAgent-375-AC-scope-replay-signal-lookup.md"
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
    - "?? docs/envelopes/QuantAgent-vje/"
        ---

# Autodev Input Envelope — QuantAgent-s62 — planner

## Objective

Execute the `planner` phase for Beads issue `QuantAgent-s62`: Extender observabilidad operativa mínima en dashboard y logs.

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
- Beads issue: `QuantAgent-s62`.
- Labels at generation time: `none`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-s62/20260513T123928Z-QuantAgent-s62-planner/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

## Contexto
QuantAgent ya tiene piezas parciales de observabilidad: heartbeat del scheduler en Streamlit (`QuantAgent-vje`) y telemetry de tokens/tiempo (`QuantAgent-69d`). Para M2 eso todavía no alcanza: el operador necesita una vista integrada del estado paper sin depender de SSH + grep de logs.

## Cambio requerido
Exponer observabilidad operativa mínima y útil para paper trading, combinando estado del scheduler, actividad de trading y señales básicas de costo/latencia.

Esto implica, como mínimo:
- mostrar en la UI/logs el estado del scheduler, última/próxima corrida, posiciones abiertas, órdenes recientes y PnL / portfolio value;
- reutilizar la telemetry de `QuantAgent-69d` para exponer al menos tiempo/tokens/costo aproximado en la superficie operativa apropiada;
- permitir troubleshooting básico filtrado por environment `paper` sin depender de inspección manual de archivos de log.

## Criterio de aceptación
- [ ] El operador puede inspeccionar desde Streamlit, usando datos reales de DB, el estado paper: scheduler status, última/próxima corrida, órdenes, posiciones y portfolio/PnL.
- [ ] Existe una vista o sección que incorpora la telemetry de tiempo/tokens/costo aproximado relevante para backtest/paper.
- [ ] Los estados sin datos o con telemetry faltante degradan de forma explícita y usable, sin romper la UI.
- [ ] Hay evidencia de validación (tests o verificación manual durable) para filtros por environment y estados vacíos/stale.

## Archivos relevantes
- `apps/streamlit/app.py`
- `apps/streamlit/views/`
- `apps/streamlit/services/db.py`
- `quantagent/llm_telemetry.py`
- `quantagent/models.py`
- `docs/user-manual/monitoring.md`
- `docs/user-manual/paper-trading-automation.md`

## Fuera de scope
- Dashboard en tiempo real por WebSocket
- Analytics avanzadas / warehouse
- Broker real

## Notas técnicas
- Reusar, no duplicar, lo ya resuelto por `QuantAgent-vje` y `QuantAgent-69d`.
- Si alguna métrica hoy existe sólo en logs, llevarla a una superficie operacional mínima en vez de agregar nueva instrumentación innecesaria.


## Acceptance Criteria

No acceptance criteria found in Beads issue.

## Recent Beads Comments

- No recent comments captured.

## Preflight Evidence

- Current branch at generation: `main`.
- Repo dirty at generation: `True`.
- Dirty files:
  - ` M .beads/issues.jsonl`
  - `?? docs/01_requirements/QuantAgent-375-RQ-scope-replay-signal-lookup.md`
  - `?? docs/02_planning/QuantAgent-375-PL-scope-replay-signal-lookup.md`
  - `?? docs/03_design/QuantAgent-375-DS-scope-replay-signal-lookup.md`
  - `?? docs/05_acceptance_tests/QuantAgent-375-AC-scope-replay-signal-lookup.md`
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
### Skill: autodev-planner
- **Resultado:** SUCCESS | PARTIAL | BLOCKED | FAIL
- **Run-ID:** 20260513T123928Z-QuantAgent-s62-planner
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
