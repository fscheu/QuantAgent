---
run_id: "20260529T023634Z-QuantAgent-kkj.5-planner"
phase: "planner"
executor: "auto"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-kkj.5"
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
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-kkj.5/20260529T023634Z-QuantAgent-kkj.5-planner"
generated_at: "2026-05-29T02:36:34.644778+00:00"
preflight:
  repo_dirty: false
  dirty_files: []
---

# Autodev Input Envelope — QuantAgent-kkj.5 — planner

## Objective

Execute the `planner` phase for Beads issue `QuantAgent-kkj.5`: [UX] Agregar help contextual en Configuration (tooltips y texto descriptivo).

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
- Beads issue: `QuantAgent-kkj.5`.
- Labels at generation time: `none`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-kkj.5/20260529T023634Z-QuantAgent-kkj.5-planner/issue.json`.
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
- **Run-ID:** 20260529T023634Z-QuantAgent-kkj.5-planner
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
