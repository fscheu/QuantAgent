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
        - Run ID: 20260504T174002Z-QuantAgent-82t-planner
        - Phase: planner
        - Skill: autodev-planner
        - Issue: QuantAgent-82t
        - Repo: /home/azureuser/repos/projects/QuantAgent
        - Artifacts dir: /home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-82t/20260504T174002Z-QuantAgent-82t-planner

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
        run_id: "20260504T174002Z-QuantAgent-82t-planner"
phase: "planner"
executor: "claude-code"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-82t"
branch_policy: "worktree-preferred"
base_branch: "main"
current_branch_at_generation: "main"
feature_branch: "feature/QuantAgent-82t-re-enable-unit-tests-in-ci-pipeline-main"
worktree_path: "/tmp/autodev-worktrees/QuantAgent/QuantAgent-82t/planner-20260504T174002Z"
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
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-82t/20260504T174002Z-QuantAgent-82t-planner"
generated_at: "2026-05-04T17:40:02.903570+00:00"
preflight:
  repo_dirty: true
  dirty_files:
    - "?? docs/envelopes/QuantAgent-88h/"
    - "?? docs/envelopes/QuantAgent-les/integration-decision-20260504T105500Z.md"
        ---

# Autodev Input Envelope — QuantAgent-82t — planner

## Objective

Execute the `planner` phase for Beads issue `QuantAgent-82t`: Re-enable unit tests in CI pipeline (main-ci-deploy.yml).

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
- Beads issue: `QuantAgent-82t`.
- Labels at generation time: `openclaw:blocked_by:QuantAgent-4ch, openclaw:blocked_by:QuantAgent-sfc, openclaw:design_approved ci`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-82t/20260504T174002Z-QuantAgent-82t-planner/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

El step de pytest en `.github/workflows/main-ci-deploy.yml` está comentado ("Unit tests temporarily disabled"). El CI actual solo corre lint, permitiendo que código roto llegue a QA. Con QuantAgent-sfc y QuantAgent-4ch resueltos, la suite estará limpia.

El workflow ya tiene un servicio PostgreSQL efímero configurado (postgres:16, DB `quantagent_test`, puerto 5432).

**Cambio requerido:**
1. Descomentar el step "Run unit tests" en `main-ci-deploy.yml`
2. Comando: `pytest tests/ -v --tb=short --maxfail=10 -m "not integration and not slow"`
3. Sin `continue-on-error: true` — debe bloquear el deploy si hay fallos
4. Verificar que `DATABASE_URL` esté disponible en el env del step

**Criterio de aceptación:**
- Step "Run unit tests" habilitado y corriendo en cada push a main
- Push con tests fallando bloquea el deploy a QA y notifica Telegram
- Push con tests pasando despliega a QA normalmente

**Archivos relevantes:**
- `.github/workflows/main-ci-deploy.yml` — step comentado ~líneas 47-55
- `pytest.ini` — markers: slow, integration, api, vision

**Fuera de scope:** no modificar los tests, no tocar main-ci.yml (ticket separado QuantAgent-ng1).

**Bloqueado por:** QuantAgent-sfc, QuantAgent-4ch


## Acceptance Criteria

No acceptance criteria found in Beads issue.

## Recent Beads Comments

- Comment 60 by Federico Scheu at 2026-04-21T15:05:42Z:

  ### Skill: autodev-planner
  - **Resultado:** SUCCESS
  - **Run-ID:** 20260421-151500Z

  #### Qué hice
  - Analicé el workflow CI actual (`.github/workflows/main-ci-deploy.yml`)
  - Identifiqué el step comentado de tests (líneas 62-69)
  - Verifiqué configuración de PostgreSQL service (ya configurado correctamente)
  - Creé documentación completa para re-habilitar tests en CI:
    - `docs/01_requirements/QuantAgent-82t-RQ-ci-test-enablement.md` — Requirements (problema, scope, 6 ACs)
    - `docs/03_design/QuantAgent-82t-DS-ci-test-enablement.md` — Design técnico (cambios exactos, decisiones, flujo)
    - `docs/05_acceptance_tests/QuantAgent-82t-AC-ci-test-enablement.md` — Acceptance criteria detallados + edge cases
    - `docs/02_planning/QuantAgent-82t-PL-ci-test-enablement.md` — Planning (7 tareas, 1.35h total)

  #### Cambios en el ticket
  - No apliqué cambios de labels/estado (issue ya tiene `openclaw:design_approved ci`)
  - Issue permanece bloqueado por QuantAgent-sfc y QuantAgent-4ch (correcto)

  #### Problemas encontrados
  - Ninguno — el workflow ya tiene PostgreSQL service configurado correctamente
  - Cambio es simple: descomentar step + ajustar parámetros + remover `continue-on-error`

  #### Dudas / d…

## Preflight Evidence

- Current branch at generation: `main`.
- Repo dirty at generation: `True`.
- Dirty files:
  - `?? docs/envelopes/QuantAgent-88h/`
  - `?? docs/envelopes/QuantAgent-les/integration-decision-20260504T105500Z.md`

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
- **Run-ID:** 20260504T174002Z-QuantAgent-82t-planner
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
