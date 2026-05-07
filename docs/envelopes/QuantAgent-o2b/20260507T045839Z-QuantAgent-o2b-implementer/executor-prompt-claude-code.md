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
        - Run ID: 20260507T045839Z-QuantAgent-o2b-implementer
        - Phase: implementer
        - Skill: autodev-implementer
        - Issue: QuantAgent-o2b
        - Repo: /tmp/autodev-worktrees/QuantAgent/QuantAgent-o2b/implementer-20260507T045511Z
        - Worktree: /mnt/actions-runner/autodev-runtime/worktrees/implementer-20260507T045511Z/QuantAgent-o2b/implementer-20260507T045839Z
        - Artifacts dir: /tmp/autodev-worktrees/QuantAgent/QuantAgent-o2b/implementer-20260507T045511Z/docs/envelopes/QuantAgent-o2b/20260507T045839Z-QuantAgent-o2b-implementer
        - Shared venv: /mnt/actions-runner/autodev-runtime/venvs/implementer-20260507T045511Z/.venv
        - Shared python: /mnt/actions-runner/autodev-runtime/venvs/implementer-20260507T045511Z/.venv/bin/python

        SHARED ENVIRONMENT POLICY:
        - If `shared_python` is declared, prefer it for Python, pytest, and tooling commands.
        - Worktrees should reuse the shared venv declared in the envelope instead of creating a per-worktree `.venv`.
        - The router prepends the shared venv `bin/` directory to PATH for the executor process.

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
        run_id: "20260507T045839Z-QuantAgent-o2b-implementer"
phase: "implementer"
executor: "claude-code"
repo_path: "/tmp/autodev-worktrees/QuantAgent/QuantAgent-o2b/implementer-20260507T045511Z"
beads_issue_id: "QuantAgent-o2b"
branch_policy: "worktree-preferred"
base_branch: "main"
current_branch_at_generation: "feature/QuantAgent-o2b-fix-azure-provider-gate-failures"
feature_branch: "feature/QuantAgent-o2b-fix-azure-provider-gate-failures-after-c"
worktree_root: "/mnt/actions-runner/autodev-runtime/worktrees/implementer-20260507T045511Z"
worktree_path: "/mnt/actions-runner/autodev-runtime/worktrees/implementer-20260507T045511Z/QuantAgent-o2b/implementer-20260507T045839Z"
shared_venv: "/mnt/actions-runner/autodev-runtime/venvs/implementer-20260507T045511Z/.venv"
shared_python: "/mnt/actions-runner/autodev-runtime/venvs/implementer-20260507T045511Z/.venv/bin/python"
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
artifacts_dir: "/tmp/autodev-worktrees/QuantAgent/QuantAgent-o2b/implementer-20260507T045511Z/docs/envelopes/QuantAgent-o2b/20260507T045839Z-QuantAgent-o2b-implementer"
generated_at: "2026-05-07T04:58:39.938833+00:00"
preflight:
  repo_dirty: true
  dirty_files:
    - " M tests/test_azure_openai_provider.py"
        ---

# Autodev Input Envelope — QuantAgent-o2b — implementer

## Objective

Execute the `implementer` phase for Beads issue `QuantAgent-o2b`: Fix Azure provider gate failures after collection blockers.

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

- Repo instructions: `/tmp/autodev-worktrees/QuantAgent/QuantAgent-o2b/implementer-20260507T045511Z/AGENTS.md` and `/tmp/autodev-worktrees/QuantAgent/QuantAgent-o2b/implementer-20260507T045511Z/CLAUDE.md` if present.
- Beads issue: `QuantAgent-o2b`.
- Labels at generation time: `none`.
- Artifact issue snapshot: `/tmp/autodev-worktrees/QuantAgent/QuantAgent-o2b/implementer-20260507T045511Z/docs/envelopes/QuantAgent-o2b/20260507T045839Z-QuantAgent-o2b-implementer/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

## Contexto
Después de corregir `QuantAgent-8yr`, el comando exacto del gate de CI dejó de fallar en collection y expuso un nuevo bloqueador real en `tests/test_azure_openai_provider.py`.

## Comando verificado
`DATABASE_URL=postgresql://test:***@localhost:5432/quantagent_test /mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python -m pytest tests/ -v --tb=short --maxfail=10 -m "not integration and not slow"`

## Fallas verificables
- `TestAzureConfiguration.test_azure_missing_endpoint_raises_error` → no levanta `ValueError`
- `TestAzureConfiguration.test_azure_missing_deployment_raises_error` → no levanta `ValueError`
- `TestAzureConfiguration.test_azure_api_version_default` → `AzureChatOpenAI` mock no es invocado
- `TestAzureLLMInstantiation.test_azure_llm_instantiation_with_correct_params` → `AzureChatOpenAI` mock no es invocado

## Cambio requerido
Corregir el contrato o los tests del provider Azure para que la suite vuelva a reflejar el comportamiento real esperado con `AzureChatOpenAI`.

## Criterio de aceptación
- [ ] `tests/test_azure_openai_provider.py -v` pasa en el entorno de test soportado.
- [ ] El comando exacto del gate deja de fallar por los 4 tests Azure.
- [ ] La corrección no rompe los providers OpenAI/Anthropic/Qwen ya cubiertos por la misma suite.

## Archivos relevantes
- `tests/test_azure_openai_provider.py`
- `quantagent/trading_graph.py`
- `tests/conftest.py` si la suite requiere bootstrap de provider opcional

## Fuera de scope
- Cambiar el workflow CI.
- Resolver fallas de otros módulos no Azure expuestas por el mismo gate.

## Acceptance Criteria

No acceptance criteria found in Beads issue.

## Recent Beads Comments

- No recent comments captured.

## Preflight Evidence

- Current branch at generation: `feature/QuantAgent-o2b-fix-azure-provider-gate-failures`.
- Repo dirty at generation: `True`.
- Dirty files:
  - ` M tests/test_azure_openai_provider.py`

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
### Skill: autodev-implementer
- **Resultado:** SUCCESS | PARTIAL | BLOCKED | FAIL
- **Run-ID:** 20260507T045839Z-QuantAgent-o2b-implementer
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
