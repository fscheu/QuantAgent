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
        - Run ID: 20260514T073839Z-QuantAgent-aki-planner
        - Phase: planner
        - Skill: autodev-planner
        - Issue: QuantAgent-aki
        - Repo: /home/azureuser/repos/projects/QuantAgent
        - Worktree: /mnt/actions-runner/autodev-runtime/worktrees/QuantAgent/QuantAgent-aki/planner-20260514T073839Z
        - Artifacts dir: /home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-aki/20260514T073839Z-QuantAgent-aki-planner
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
        run_id: "20260514T073839Z-QuantAgent-aki-planner"
phase: "planner"
executor: "claude-code"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-aki"
branch_policy: "worktree-preferred"
base_branch: "main"
current_branch_at_generation: "main"
feature_branch: "feature/QuantAgent-aki-ejecutar-piloto-controlado-de-paper-trad"
worktree_root: "/mnt/actions-runner/autodev-runtime/worktrees/QuantAgent"
worktree_path: "/mnt/actions-runner/autodev-runtime/worktrees/QuantAgent/QuantAgent-aki/planner-20260514T073839Z"
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
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-aki/20260514T073839Z-QuantAgent-aki-planner"
generated_at: "2026-05-14T07:38:39.163028+00:00"
preflight:
  repo_dirty: true
  dirty_files:
    - "?? docs/01_requirements/QuantAgent-375-RQ-scope-replay-signal-lookup.md"
    - "?? docs/02_planning/QuantAgent-375-PL-scope-replay-signal-lookup.md"
    - "?? docs/03_design/QuantAgent-375-DS-scope-replay-signal-lookup.md"
    - "?? docs/05_acceptance_tests/QuantAgent-375-AC-scope-replay-signal-lookup.md"
    - "?? docs/envelopes/QuantAgent-339/20260513T142701Z-QuantAgent-339-planner/"
    - "?? docs/envelopes/QuantAgent-339/20260513T142703Z-QuantAgent-339-implementer/"
    - "?? docs/envelopes/QuantAgent-339/20260514T023928Z-QuantAgent-339-tester/"
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
    - "?? docs/envelopes/QuantAgent-s62/20260514T000532Z-QuantAgent-s62-qa-validator/"
    - "?? docs/envelopes/QuantAgent-sft/20260513T174441Z-QuantAgent-sft-implementer/"
    - "?? docs/envelopes/QuantAgent-sft/20260514T000542Z-QuantAgent-sft-qa-validator/"
    - "?? docs/envelopes/QuantAgent-vje/"
        ---

# Autodev Input Envelope — QuantAgent-aki — planner

## Objective

Execute the `planner` phase for Beads issue `QuantAgent-aki`: Ejecutar piloto controlado de paper trading y emitir readiness report.

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
- Beads issue: `QuantAgent-aki`.
- Labels at generation time: `none`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-aki/20260514T073839Z-QuantAgent-aki-planner/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

## Contexto
M2 no se cierra con features mergeadas: requiere evidencia operativa real. Una vez estabilizados runtime, QA y observabilidad, QuantAgent necesita un piloto controlado de paper trading que permita decidir honestamente si el próximo paso es broker real o si todavía faltan bloqueadores.

## Cambio requerido
Diseñar y ejecutar un piloto controlado de paper trading, acotado y repetible.

Esto implica, como mínimo:
- definir runbook del piloto (universo, estrategias, duración o cantidad de ciclos, criterios de éxito/falla);
- limitarlo a 1–3 estrategias M1 ya mergeadas;
- correr el piloto con el runtime actual y capturar evidencia operativa real;
- emitir un readiness report con recomendación explícita sobre el siguiente milestone.

## Criterio de aceptación
- [ ] Existe un runbook/pilot plan versionado en `docs/` con alcance, precondiciones y criterios de salida.
- [ ] El piloto se ejecuta durante un período o cantidad de ciclos definida y deja evidencia real (trades, posiciones, errores, telemetry/costo, timings).
- [ ] Se genera un resumen operativo con trades ejecutados, métricas base, fallas detectadas, costo/latencia aproximados y recomendación clara de go/no-go para broker real.
- [ ] Si el piloto no es apto para avanzar, el reporte lista bloqueadores concretos que puedan volver al backlog como tickets separados.

## Archivos relevantes
- `docs/02_planning/`
- `docs/05_acceptance_tests/`
- `docs/06_implementation/`
- `docs/envelopes/`
- `apps/streamlit/`
- `quantagent/trading/scheduler.py`

## Fuera de scope
- Capital real
- Operación continua sin límite temporal
- Estrategias nuevas fuera del set M1

## Notas técnicas
- Este ticket debe ejecutarse después de que runtime, QA y observabilidad estén en estado suficientemente estable.
- Si se detecta que `QuantAgent-um8` o cualquier otra optimización es blocker real del piloto, documentarlo con evidencia antes de reabsorber scope.


## Acceptance Criteria

No acceptance criteria found in Beads issue.

## Recent Beads Comments

- No recent comments captured.

## Preflight Evidence

- Current branch at generation: `main`.
- Repo dirty at generation: `True`.
- Dirty files:
  - `?? docs/01_requirements/QuantAgent-375-RQ-scope-replay-signal-lookup.md`
  - `?? docs/02_planning/QuantAgent-375-PL-scope-replay-signal-lookup.md`
  - `?? docs/03_design/QuantAgent-375-DS-scope-replay-signal-lookup.md`
  - `?? docs/05_acceptance_tests/QuantAgent-375-AC-scope-replay-signal-lookup.md`
  - `?? docs/envelopes/QuantAgent-339/20260513T142701Z-QuantAgent-339-planner/`
  - `?? docs/envelopes/QuantAgent-339/20260513T142703Z-QuantAgent-339-implementer/`
  - `?? docs/envelopes/QuantAgent-339/20260514T023928Z-QuantAgent-339-tester/`
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
  - `?? docs/envelopes/QuantAgent-s62/20260514T000532Z-QuantAgent-s62-qa-validator/`
  - `?? docs/envelopes/QuantAgent-sft/20260513T174441Z-QuantAgent-sft-implementer/`
  - `?? docs/envelopes/QuantAgent-sft/20260514T000542Z-QuantAgent-sft-qa-validator/`
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
- **Run-ID:** 20260514T073839Z-QuantAgent-aki-planner
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
