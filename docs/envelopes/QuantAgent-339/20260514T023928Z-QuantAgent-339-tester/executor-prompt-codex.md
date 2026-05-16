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
        - Run ID: 20260514T023928Z-QuantAgent-339-tester
        - Phase: tester
        - Skill: autodev-tester
        - Issue: QuantAgent-339
        - Repo: /home/azureuser/repos/projects/QuantAgent
        - Worktree: /mnt/actions-runner/autodev-runtime/worktrees/QuantAgent/QuantAgent-339/tester-20260514T023928Z
        - Artifacts dir: /home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-339/20260514T023928Z-QuantAgent-339-tester
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
        run_id: "20260514T023928Z-QuantAgent-339-tester"
phase: "tester"
executor: "codex"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-339"
branch_policy: "worktree-preferred"
base_branch: "main"
current_branch_at_generation: "main"
feature_branch: "feature/QuantAgent-339-qa-validator-runtime-real"
worktree_root: "/mnt/actions-runner/autodev-runtime/worktrees/QuantAgent"
worktree_path: "/mnt/actions-runner/autodev-runtime/worktrees/QuantAgent/QuantAgent-339/tester-20260514T023928Z"
shared_venv: "/mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv"
shared_python: "/mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python"
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
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-339/20260514T023928Z-QuantAgent-339-tester"
generated_at: "2026-05-14T02:39:28.831568+00:00"
preflight:
  repo_dirty: true
  dirty_files:
    - " M .beads/.migration-hint-ts"
    - "?? docs/01_requirements/QuantAgent-375-RQ-scope-replay-signal-lookup.md"
    - "?? docs/02_planning/QuantAgent-375-PL-scope-replay-signal-lookup.md"
    - "?? docs/03_design/QuantAgent-375-DS-scope-replay-signal-lookup.md"
    - "?? docs/05_acceptance_tests/QuantAgent-375-AC-scope-replay-signal-lookup.md"
    - "?? docs/envelopes/QuantAgent-339/"
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

# Autodev Input Envelope — QuantAgent-339 — tester

## Objective

Execute the `tester` phase for Beads issue `QuantAgent-339`: Consolidar QA Streamlit y validator post-deploy sobre runtime real.

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

- Repo instructions: `/home/azureuser/repos/projects/QuantAgent/AGENTS.md` and `/home/azureuser/repos/projects/QuantAgent/CLAUDE.md` if present.
- Beads issue: `QuantAgent-339`.
- Labels at generation time: `openclaw:design_pending`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-339/20260514T023928Z-QuantAgent-339-tester/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

## Contexto
QA ya fue movido hacia Streamlit en puerto 8501 y existe documentación/infra de validator local post-deploy. M2 necesita convertir ese avance en una verificación funcional confiable del runtime real, no sólo en healthchecks o evidencia manual parcial.

El objetivo no es rediseñar el deploy, sino consolidar que `Main CI + Deploy QA` y el validator apunten al target correcto, validen las vistas críticas del MVP y produzcan evidencia durable cuando algo falla.

## Cambio requerido
Consolidar la validación post-deploy de QA sobre la UI Streamlit real.

Esto implica, como mínimo:
- alinear workflow, healthcheck y validator al runtime Streamlit servido en 8501;
- verificar funcionalmente las vistas clave del MVP desde el target local soportado por el validator;
- dejar el contrato `deploy_finished` / `qa_verified` consistente con la evidencia del validator;
- capturar artifacts útiles cuando la validación sea PARTIAL/BLOCKED/FAIL.

## Criterio de aceptación
- [ ] El workflow principal de deploy QA verifica el runtime Streamlit correcto y ejecuta el validator contra el target local soportado.
- [ ] El validator cubre al menos Dashboard, Backtesting/Replay, Orders & Positions y Logs/Paper Trading status con evidencia explícita.
- [ ] La salida del validator diferencia claramente `SUCCESS`, `PARTIAL`, `BLOCKED` y `FAIL` usando artifacts durables.
- [ ] Un deploy exitoso de QA deja evidencia verificable de que la UI correcta está sirviendo datos útiles en el runtime real.

## Archivos relevantes
- `.github/workflows/main-ci-deploy.yml`
- `Dockerfile.qa`
- `docker-compose.qa.yml`
- `apps/streamlit/`
- `docs/06_implementation/QuantAgent-app-IM-qa-streamlit-cutover.md`
- `docs/envelopes/`

## Fuera de scope
- Automatizar Cloudflare Access
- Prod deploy
- Rediseño visual del dashboard

## Notas técnicas
- Reusar el cutover ya mergeado a 8501 y el enfoque de validator local como baseline.
- No volver a un target legacy de Flask/8001 salvo evidencia fuerte de regresión real.


## Acceptance Criteria

No acceptance criteria found in Beads issue.

## Recent Beads Comments

- Comment 196 by Federico Scheu at 2026-05-13T02:49:30Z:

  ### Skill: autodev-planner
  - **Resultado:** SUCCESS
  - **Run-ID:** 20260513T023641Z

  #### Qué hice
  - Docs creados/editados:
    - `docs/01_requirements/QuantAgent-339-RQ-qa-validator-real-runtime.md`
    - `docs/02_planning/QuantAgent-339-PL-qa-validator-real-runtime.md`
    - `docs/03_design/QuantAgent-339-DS-qa-validator-real-runtime.md`
    - `docs/05_acceptance_tests/QuantAgent-339-AC-qa-validator-real-runtime.md`
    - índices `README.md` de requirements/planning/design/acceptance
  - Artefactos durables:
    - `docs/envelopes/QuantAgent-339/20260513T023641Z-techlead-planner/run-report.md`
    - `docs/envelopes/QuantAgent-339/20260513T023641Z-techlead-planner/result.json`
  - Branch publicada: `feature/QuantAgent-339-qa-validator-runtime-real` (`8ff6b79d`)

  #### Cambios en el ticket
  - Sin cambios de labels/estado en esta corrida.

  #### Problemas encontrados
  - El repo root principal sigue dirty por trabajo previo no atribuible a esta corrida; se aisló todo en worktree.
  - Parte del comportamiento depende de `qa-validator-poc`, fuera del repo principal.

  #### Dudas / decisiones pendientes
  1) El implementer debe mantener explícita la semántica de `PARTIAL` para empty state sano.
  2) Si el contrato de…
- Comment 197 by Federico Scheu at 2026-05-13T07:48:26Z:

  ### Skill: autodev-implementer
  - **Resultado:** SUCCESS
  - **Run-ID:** 20260513T074107Z-QuantAgent-339-implementer
  - **Executor:** auto

  #### Qué hice
  - Implementé los tres seams del diseño en `.github/workflows/main-ci-deploy.yml` (+69/-2 líneas):
    - **Target seam (AC6):** nuevo step `Verify QA validator target alignment` — detecta drift entre runtime desplegado (8501) y config del validator antes de ejecutar. Falla con evidencia explícita si el config no referencia 8501.
    - **Artifact seam (AC2):** `Read QA validator verdict` ahora lista `result.json`, `run-report.md`, screenshots y logs; nuevo step `Upload QA validator artifacts` persiste todo en GitHub Actions (retención 30 días) en cada run.
    - **Verdict seam (AC4, AC5):** `PARTIAL` es estado propio (`qa_partial` en outputs y payload); `qa_verified` exclusivo de `SUCCESS`; webhook Hermes expone `qa_partial`, `qa_target_alignment_outcome`, `validator_artifacts_count` y `validator_artifacts_list`.
  - Commit: `b438945b` en `feature/QuantAgent-339-qa-validator-runtime-real`

  #### Evidencia
  - `git status --short`: sólo `.github/workflows/main-ci-deploy.yml` modificado
  - `ruff check --fix .`: 5 errores pre-existentes no relacionado…

## Preflight Evidence

- Current branch at generation: `main`.
- Repo dirty at generation: `True`.
- Dirty files:
  - ` M .beads/.migration-hint-ts`
  - `?? docs/01_requirements/QuantAgent-375-RQ-scope-replay-signal-lookup.md`
  - `?? docs/02_planning/QuantAgent-375-PL-scope-replay-signal-lookup.md`
  - `?? docs/03_design/QuantAgent-375-DS-scope-replay-signal-lookup.md`
  - `?? docs/05_acceptance_tests/QuantAgent-375-AC-scope-replay-signal-lookup.md`
  - `?? docs/envelopes/QuantAgent-339/`
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
### Skill: autodev-tester
- **Resultado:** SUCCESS | PARTIAL | BLOCKED | FAIL
- **Run-ID:** 20260514T023928Z-QuantAgent-339-tester
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
