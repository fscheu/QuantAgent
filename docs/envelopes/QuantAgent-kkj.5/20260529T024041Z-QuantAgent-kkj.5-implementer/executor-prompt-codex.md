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
        - Run ID: 20260529T024041Z-QuantAgent-kkj.5-implementer
        - Phase: implementer
        - Skill: autodev-implementer
        - Issue: QuantAgent-kkj.5
        - Repo: /home/azureuser/repos/projects/QuantAgent
        - Execution workspace: /tmp/autodev-worktrees/QuantAgent/QuantAgent-kkj.5/implementer
        - Working branch hint: feature/QuantAgent-kkj.5-ux-agregar-help-contextual-en-configurat
        - Canonical publication branch: n/a
        - Artifacts dir: /home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-kkj.5/20260529T024041Z-QuantAgent-kkj.5-implementer
        - Allowed run-owned dirty paths: ["docs/envelopes/QuantAgent-kkj.5/20260529T024041Z-QuantAgent-kkj.5-implementer"]
        - Shared venv: /home/azureuser/repos/projects/QuantAgent/.venv
        - Shared python: /home/azureuser/repos/projects/QuantAgent/.venv/bin/python
        - Beads CLI: /home/azureuser/snap/codex/common/bin/bd
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
  "write_code": true,
  "write_tests": false,
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
    "ruff check --fix .",
    "pytest <relevant subset> -v",
    "python -m compileall -q ."
  ],
  "optional": [
    "python -m compileall -q ."
  ]
}

        FULL INPUT ENVELOPE:
        ---
        run_id: "20260529T024041Z-QuantAgent-kkj.5-implementer"
phase: "implementer"
executor: "codex"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-kkj.5"
branch_policy: "worktree-preferred"
base_branch: "main"
publication_branch: null
current_branch_at_generation: "main"
feature_branch: "feature/QuantAgent-kkj.5-ux-agregar-help-contextual-en-configurat"
worktree_root: "/tmp/autodev-worktrees/QuantAgent"
worktree_path: "/tmp/autodev-worktrees/QuantAgent/QuantAgent-kkj.5/implementer"
shared_venv: "/home/azureuser/repos/projects/QuantAgent/.venv"
shared_python: "/home/azureuser/repos/projects/QuantAgent/.venv/bin/python"
skill: "autodev-implementer"
mode: "write"
capabilities:
  read_repo: true
  write_docs: true
  write_code: true
  write_tests: false
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
    - "ruff check --fix ."
    - "pytest <relevant subset> -v"
    - "python -m compileall -q ."
  optional:
    - "python -m compileall -q ."
budget:
  max_turns: 10
  max_minutes: 45
  max_cost_usd: null
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-kkj.5/20260529T024041Z-QuantAgent-kkj.5-implementer"
generated_at: "2026-05-29T02:40:41.864130+00:00"
preflight:
  repo_dirty: false
  dirty_files: "[]"
execution_workspace: "/tmp/autodev-worktrees/QuantAgent/QuantAgent-kkj.5/implementer"
current_branch_at_execution: "feature/QuantAgent-kkj.5-ux-agregar-help-contextual-en-configurat"
allowed_dirty_paths:
  - "docs/envelopes/QuantAgent-kkj.5/20260529T024041Z-QuantAgent-kkj.5-implementer"
beads_bin: "/home/azureuser/snap/codex/common/bin/bd"
beads_safe_wrapper: "/home/azureuser/repos/agents/openclaw/scripts/bd_safe.sh"
        ---

# Autodev Input Envelope — QuantAgent-kkj.5 — implementer

## Objective

Execute the `implementer` phase for Beads issue `QuantAgent-kkj.5`: [UX] Agregar help contextual en Configuration (tooltips y texto descriptivo).

## Scope In

- Follow the `autodev-implementer` skill for this phase.
- Use the Beads issue, repo instructions, docs, and recent comments as source of truth.
- Preserve migration defaults: no push, no merge, no deploy, no external messages.

## Scope Out

- Do not modify credentials, `.env`, secrets, production service config, or systemd/OpenClaw/Hermes runtime config.
- Do not rename legacy `openclaw:*` labels during this phase.
- Do not work on unrelated issues or opportunistic refactors.
- Do not merge to `main`.

## Source of Truth

- Repo instructions: `/home/azureuser/repos/projects/QuantAgent/AGENTS.md` and `/home/azureuser/repos/projects/QuantAgent/CLAUDE.md` if present.
- Beads issue: `QuantAgent-kkj.5`.
- Labels at generation time: `none`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-kkj.5/20260529T024041Z-QuantAgent-kkj.5-implementer/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

## Contexto

La pestaña Configuration de Streamlit carece de texto descriptivo y ayuda contextual. El operador no puede entender qué hace cada campo, combo o selector sin consultar el user manual externo.

Puntos específicos con mayor necesidad de ayuda contextual, identificados durante revisión funcional M2 (2026-05-25):
- Selector de provider/modelo LLM: no es claro qué afecta (solo a LLM strategies, no a deterministic strategies como RSI o 52-week high).
- Combos "Paper default portfolio" y "Backtest default portfolio": aparecen vacíos sin explicación de cómo generarlos.
- Selector "Universe (for portfolio profiles only)": la aclaración está ahí pero no queda claro en qué operación concreta aplica.
- Sección de carga de Profile JSON: no está claro si reemplaza o complementa los campos del form.

## Cambio requerido

Agregar help contextual en la vista Configuration:
- `st.caption` o texto descriptivo gris bajo cada sección principal con una línea de propósito y cuándo usarlo.
- `help=` parameter en `st.selectbox`, `st.text_input`, etc., para los campos más críticos (provider, model, universe, portfolio defaults).
- En combos vacíos (portfolios), texto que explica el flujo de creación (ej: "Cargá un portfolio profile desde JSON para habilitar este selector").

## Criterio de aceptación

- [ ] Cada sección principal de Configuration tiene un caption o subtítulo que describe su propósito en una línea.
- [ ] Los combos de portfolio (paper default / backtest default) muestran un texto de ayuda cuando están vacíos.
- [ ] El selector de universe tiene un tooltip (`help=`) o caption que explica cuándo aplica y a qué afecta.
- [ ] El selector de provider/modelo LLM tiene un caption que aclara que solo afecta a LLM-based strategies.
- [ ] La sección de carga de Profile JSON tiene un caption que explica si reemplaza o complementa el form.

## Archivos relevantes

- `apps/streamlit/views/configuration.py` — vista a enriquecer con help contextual.

## Fuera de scope (no tocar)

- No cambiar lógica backend ni modelos.
- No rediseñar el layout de la pestaña (eso va en el ticket de split LLM/Portfolio).
- No implementar onboarding wizard ni tours guiados.
- No agregar validaciones que hoy no existen.


## Acceptance Criteria

No acceptance criteria found in Beads issue.

## Recent Beads Comments

- Comment 245 by Federico Scheu at 2026-05-29T02:39:24Z:

  ### Skill: autodev-planner
  - **Resultado:** SUCCESS
  - **Run-ID:** 20260529T023634Z-QuantAgent-kkj.5-planner
  - **Executor:** auto

  #### Qué hice
  - Analicé el código actual de `apps/streamlit/views/configuration.py` (post kkj.4: ya tiene dos tabs LLM Settings / Portfolio & Universe)
  - Produje tres artifacts canónicos: RQ, AC (8 criterios Given/When/Then), PL con 6 cambios aditivos detallados
  - Actualicé los README index de `docs/01_requirements/` y `docs/05_acceptance_tests/`

  #### Evidencia
  - `docs/01_requirements/QuantAgent-kkj.5-RQ-configuration-contextual-help.md`
  - `docs/05_acceptance_tests/QuantAgent-kkj.5-AC-configuration-contextual-help.md`
  - `docs/02_planning/QuantAgent-kkj.5-PL-configuration-contextual-help.md`

  #### Quality gates
  - git status: solo archivos planner en dirty
  - Issue ID en rutas de docs: PASS
  - ACs testeables: PASS (8 criterios con comandos de verificación)
  - Branch == main (publication branch): PASS
  - compileall configuration.py: SYNTAX_OK

  #### Problemas encontrados
  - Ninguno. El código post kkj.4 ya tiene la estructura de tabs que este ticket enriquece.

  #### Next step recomendado
  - Fase implementer: aplicar los 6 cambios de `QuantAgent-kkj.5-PL-configura…

## Preflight Evidence

- Current branch at generation: `main`.
- Canonical publication branch: `n/a`.
- Repo dirty at generation: `False`.
- Dirty files:
  - none

## Publication Policy

- Branch policy: `worktree-preferred`.
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
### Skill: autodev-implementer
- **Resultado:** SUCCESS | PARTIAL | BLOCKED | FAIL
- **Run-ID:** 20260529T024041Z-QuantAgent-kkj.5-implementer
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
