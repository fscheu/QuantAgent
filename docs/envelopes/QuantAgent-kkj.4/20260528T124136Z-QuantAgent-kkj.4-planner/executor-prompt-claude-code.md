You are running a Hermes autodev phase under a strict executor contract.

        READ THIS FIRST:
        - One run = one Beads issue + one phase.
        - Do not infer extra permissions. The YAML capabilities are authoritative.
        - Git push is forbidden unless the envelope explicitly enables planner canonical publication.
        - No merge to main, no deploy, no external messages.
        - Do not read or print secrets.
        - If blocked, stop and return a structured BLOCKED report.
        - End with a structured output envelope/report.
        - Write canonical artifacts under the declared artifacts dir whenever possible: `result.json`, `run-report.md`, `commands.log`, `quality-gates.log`.
        - If Beads is available and the envelope allows it, add exactly one final Beads comment using the provided template.
        - If adding that comment updates `.beads/issues.jsonl`, stage it and include it in the final planner publication commit/push so the repo does not end dirty.

        RUN METADATA:
        - Run ID: 20260528T124136Z-QuantAgent-kkj.4-planner
        - Phase: planner
        - Skill: autodev-planner
        - Issue: QuantAgent-kkj.4
        - Repo: /home/azureuser/repos/projects/QuantAgent
        - Execution workspace: /home/azureuser/repos/projects/QuantAgent
        - Working branch hint: main
        - Canonical publication branch: main
        - Artifacts dir: /home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-kkj.4/20260528T124136Z-QuantAgent-kkj.4-planner
        - Allowed run-owned dirty paths: ["docs/envelopes/QuantAgent-kkj.4/20260528T124136Z-QuantAgent-kkj.4-planner"]
        - Shared venv: /home/azureuser/repos/projects/QuantAgent/.venv
        - Shared python: /home/azureuser/repos/projects/QuantAgent/.venv/bin/python
        - Beads CLI: /home/azureuser/.local/bin/bd
        - Beads safe wrapper: /home/azureuser/repos/agents/openclaw/scripts/bd_safe.sh

        PUBLICATION SAFETY:
        - If `publication_branch` is set and `capabilities.git_commit` / `capabilities.git_push` are true, publish the canonical planner docs directly on that branch.
        - Do the Beads final comment before the last planner commit/push whenever possible so any `.beads/issues.jsonl` sync lands in the same publication commit.
        - If `.beads/issues.jsonl` changes only after the comment step, amend or add one follow-up sync commit before finishing, then push, so the repo ends clean.
        - Do not stage or commit unrelated repo changes.
        - If this is an in-place publication on `publication_branch`, stop with `BLOCKED` when the repo is dirty or on the wrong branch.
        - For write phases running in an isolated execution workspace, validate cleanliness in that execution workspace/worktree. Untracked files under `allowed_dirty_paths` are run-owned operational artifacts and do not count as unrelated drift by themselves.
        - The execution workspace above is authoritative for this run, even if an earlier generation-time worktree hint differed.

        SHARED ENVIRONMENT POLICY:
        - If `shared_python` is declared, prefer it for Python, pytest, and tooling commands.
        - Worktrees should reuse the shared venv declared in the envelope instead of creating a per-worktree `.venv`.
        - The router prepends the shared venv `bin/` directory to PATH for the executor process.
        - If Beads access is needed, prefer `AUTODEV_BEADS_BIN` / `AUTODEV_BEADS_SAFE_WRAPPER` (or the absolute paths above) instead of assuming bare `bd` resolves correctly in every executor environment.

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
  "git_commit": true,
  "git_push": true,
  "merge_main": false,
  "deploy": false,
  "send_external_message": false,
  "touch_secrets": false
}

        FORBIDDEN ACTIONS:
        [
  "git push to any branch other than the declared publication branch",
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
    "verify acceptance criteria are testable",
    "confirm repo is clean before canonical planner publication",
    "confirm current branch matches canonical publication branch"
  ],
  "optional": [
    "python -m compileall -q ."
  ]
}

        FULL INPUT ENVELOPE:
        ---
        run_id: "20260528T124136Z-QuantAgent-kkj.4-planner"
phase: "planner"
executor: "claude-code"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-kkj.4"
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
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-kkj.4/20260528T124136Z-QuantAgent-kkj.4-planner"
generated_at: "2026-05-28T12:41:36.292844+00:00"
preflight:
  repo_dirty: false
  dirty_files: "[]"
execution_workspace: "/home/azureuser/repos/projects/QuantAgent"
current_branch_at_execution: "main"
allowed_dirty_paths:
  - "docs/envelopes/QuantAgent-kkj.4/20260528T124136Z-QuantAgent-kkj.4-planner"
beads_bin: "/home/azureuser/.local/bin/bd"
beads_safe_wrapper: "/home/azureuser/repos/agents/openclaw/scripts/bd_safe.sh"
        ---

# Autodev Input Envelope — QuantAgent-kkj.4 — planner

## Objective

Execute the `planner` phase for Beads issue `QuantAgent-kkj.4`: [UX] Separar pestaña Configuration en LLM Settings y Portfolio & Universe.

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
- Beads issue: `QuantAgent-kkj.4`.
- Labels at generation time: `none`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-kkj.4/20260528T124136Z-QuantAgent-kkj.4-planner/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

## Contexto

La pestaña Configuration de Streamlit mezcla dos categorías conceptualmente independientes en la misma vista:

1. **LLM profiles/presets**: configuración de providers (OpenAI, Anthropic, etc.), modelos, temperatura y parámetros de inferencia.
2. **Portfolio/Universe config**: portfolios (universo de activos), perfiles de riesgo, presets de estrategias, y defaults paper/backtest.

Además incluye:
- Combos "Paper default portfolio" y "Backtest default portfolio" que aparecen vacíos sin explicación de cómo generarlos ni para qué sirven.
- Una sección de carga de Profile en JSON mezclada con un selector de universe etiquetado "(for portfolio profiles only)" que no queda claro en qué contexto aplica.

Observado durante revisión funcional M2 (2026-05-25): un operador con conocimiento del sistema tuvo que consultar el user manual para entender la UI. La mezcla de LLM config con portfolio config genera confusión sobre qué afecta a qué.

## Cambio requerido

Separar la pestaña Configuration en dos secciones claramente delimitadas (pueden ser sub-tabs dentro de Configuration o una reorganización visual explícita con headers):

- **LLM Settings**: providers, modelos, presets, temperatura, parámetros de inferencia.
- **Portfolio & Universe**: portfolio profiles, risk profiles, universos de activos, defaults paper/backtest, carga de profiles JSON.

Dentro de Portfolio & Universe, aclarar con texto de ayuda cómo se genera un portfolio profile y qué hacen los selectores de default paper/backtest.

## Criterio de aceptación

- [ ] La vista Configuration tiene dos secciones claramente separadas y etiquetadas: LLM Settings y Portfolio & Universe.
- [ ] Los combos "Paper default portfolio" y "Backtest default portfolio" tienen caption o texto de ayuda que explica cómo crear/cargar un portfolio profile.
- [ ] La sección de carga de Profile JSON está dentro de Portfolio & Universe, no mezclada con LLM settings.
- [ ] El selector de universe está en Portfolio & Universe y su label o caption explica cuándo aplica.
- [ ] La reorganización no elimina funcionalidad existente.

## Archivos relevantes

- `apps/streamlit/views/configuration.py` — vista a reorganizar.

## Fuera de scope (no tocar)

- No cambiar la lógica backend de configuración ni los modelos de datos.
- No rediseñar otras pestañas.
- No implementar nuevos tipos de profiles ni nuevas integraciones de LLM.


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
- **Run-ID:** 20260528T124136Z-QuantAgent-kkj.4-planner
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
