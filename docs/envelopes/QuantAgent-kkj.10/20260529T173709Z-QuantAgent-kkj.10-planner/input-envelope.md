---
run_id: "20260529T173709Z-QuantAgent-kkj.10-planner"
phase: "planner"
executor: "auto"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-kkj.10"
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
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-kkj.10/20260529T173709Z-QuantAgent-kkj.10-planner"
generated_at: "2026-05-29T17:37:09.480078+00:00"
preflight:
  repo_dirty: false
  dirty_files: []
---

# Autodev Input Envelope — QuantAgent-kkj.10 — planner

## Objective

Execute the `planner` phase for Beads issue `QuantAgent-kkj.10`: Persistir y seedear catálogo base de configuración para QA/DEV limpio.

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
- Beads issue: `QuantAgent-kkj.10`.
- Labels at generation time: `configuration, dx, qa`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-kkj.10/20260529T173709Z-QuantAgent-kkj.10-planner/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

## Contexto

Durante la revisión funcional de `QuantAgent-339` quedó expuesto un gap operativo concreto: QuantAgent ya tiene scripts para seedear datos de DEV (`scripts/seed_dev.py`) y un bootstrap determinístico mínimo para QA (`scripts/bootstrap_qa_minimal.py`), pero no tiene una forma confiable de inicializar un environment limpio con un catálogo base de configuración utilizable desde la UI.

Hallazgos verificados en código actual:
- `StrategyConfig` persiste perfiles `portfolio | risk | combined` en DB (`quantagent/models.py`), así que hay dónde guardar portfolios y profiles base.
- La UI de Configuration muestra defaults paper/backtest leyendo portfolios desde DB o `st.session_state`, pero hoy esos combos pueden quedar vacíos en una DB limpia (`apps/streamlit/views/configuration.py`).
- Los `model_presets` hoy viven sólo en `st.session_state` y arrancan con un único preset `default`; no existe persistencia durable para presets LLM (`apps/streamlit/app.py`, `apps/streamlit/views/configuration.py`).
- La UI sólo expone providers `openai`, `anthropic`, `qwen`, mientras que el backend de `TradingGraph` también soporta `azure`; no existe soporte real hoy para `litellm`, `github-copilot` ni `azure-foundry` como providers de QuantAgent.
- `DEFAULT_SCHEDULER_ASSETS = ["BTC", "SPX"]` es demasiado chico como base operativa y no resuelve presets listos para backtest/paper trading (`quantagent/settings.py`).

Conclusión técnica:
- Sí es viable implementar un bootstrap “vivo” de configuración.
- No alcanza con un YAML lindo: para seedear presets LLM de forma útil primero hay que darles persistencia real y alinear la lista de providers soportados entre UI y backend.
- Meter `github-copilot`/`litellm`/`azure-foundry` en el seed ahora sería humo: quedarían como presets muertos porque QuantAgent todavía no los sabe ejecutar end-to-end. Primero hay que soportarlos de verdad o dejarlos fuera de este ticket.

## Cambio requerido

Implementar un bootstrap versionado de configuración base para environments limpios, cubriendo perfiles operativos y presets LLM realmente utilizables hoy.

El ticket debe incluir, como mínimo:

1. Crear una fuente de verdad versionada para configuración base (`config/seed/base_catalog.yaml` o similar) con:
   - 2-3 portfolio profiles listos para uso operativo;
   - 1-2 risk/combined profiles base;
   - un universo de assets más amplio que el default actual, con perfiles separados al menos para:
     - paper trading base;
     - backtesting equity base;
     - backtesting crypto/mixed base.

2. Implementar `scripts/seed_config_catalog.py` (o nombre equivalente) que:
   - lea ese catálogo versionado;
   - haga upsert idempotente de los profiles en DB;
   - soporte `--reset` para reinicialización limpia;
   - falle con mensaje claro si la DB no está accesible o si el schema no es compatible con el catálogo.

3. Dar persistencia durable a los model presets LLM para que el seed no dependa de `session_state`. Se acepta cualquiera de estas dos opciones mínimas:
   - extender `StrategyConfig`/su uso para soportar `kind=model_preset`; o
   - crear una entidad persistente dedicada para model presets.

4. Unificar la fuente de verdad de providers soportados entre UI y backend para evitar el desfasaje actual (`openai/anthropic/qwen` en UI vs `openai/anthropic/qwen/azure` en backend).

5. Seedear presets LLM base sólo para providers realmente soportados por QuantAgent después del punto anterior. Mínimo esperado:
   - un preset OpenAI;
   - un preset Anthropic;
   - un preset Azure si queda soportado end-to-end en Configuration/UI;
   - `qwen` opcional si sigue siendo provider soportado y testeable.

6. Documentar el flujo operativo de bootstrap limpio para QA/DEV y dejar al menos un smoke test automatizado que detecte drift básico entre catálogo, schema y loaders.

## Acceptance Criteria

- [ ] Existe un catálogo versionado de configuración base en el repo (`config/seed/` o equivalente).
- [ ] Existe un script ejecutable que carga ese catálogo en una DB limpia y es idempotente.
- [ ] Tras correr el script en una DB limpia, Configuration muestra portfolios base utilizables para `Paper default portfolio` y `Backtest default portfolio`.
- [ ] El catálogo incluye al menos 3 profiles operativos con universos más amplios que el default actual y propósito explícito (paper / backtest equity / backtest crypto o mixed).
- [ ] Los model presets dejan de vivir sólo en `st.session_state` y pasan a tener persistencia durable.
- [ ] La UI y el backend comparten la misma lista efectiva de providers soportados.
- [ ] Los presets seeded sólo incluyen providers realmente soportados por QuantAgent; no se cargan presets “fantasma” para `litellm`, `github-copilot` o `azure-foundry` sin soporte real en runtime.
- [ ] Hay al menos un smoke test que falla si el catálogo queda incompatible con el schema o con los loaders.
- [ ] README o user manual documenta cómo reinicializar QA/DEV con este bootstrap.

## Archivos relevantes

- `quantagent/models.py`
- `quantagent/settings.py`
- `quantagent/trading_graph.py`
- `apps/streamlit/app.py`
- `apps/streamlit/views/configuration.py`
- `scripts/seed_dev.py`
- `scripts/bootstrap_qa_minimal.py`
- `config/seed/` (nuevo)
- `tests/` (smoke test nuevo)

## Fuera de scope (no tocar)

- No poblar trades, orders, backtest runs o heartbeats sintéticos en este ticket; eso va separado en fixtures QA runtime.
- No automatizar la ejecución del seed en deploy/CI por defecto.
- No agregar soporte runtime nuevo para `litellm`, `github-copilot` o `azure-foundry` sólo para decorar presets.
- No rediseñar la UX completa de Configuration más allá de lo necesario para consumir la persistencia nueva.

## Notas técnicas

- Recomendación de scope: mantener este ticket chico pero real. Catálogo versionado + persistencia de model presets + provider registry compartido + smoke test. Nada de framework genérico mágico porque después nadie lo mantiene.
- El catálogo debe ser explícito y legible, no inferido desde defaults dispersos en código. Si mañana cambia el schema, el test tiene que romper antes de que QA quede a medio cocinar.
- Los profiles seeded deberían usar nombres de negocio claros, por ejemplo: `paper_us_base`, `backtest_equities_swing`, `backtest_crypto_intraday`. Nada de nombres crípticos para que después haya que adivinar qué demonios carga cada uno.
- Ticket hermano/follow-up natural: fixtures runtime de QA para poblar runs, señales, órdenes y trades sin ejecutar backtesting real.


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
- **Run-ID:** 20260529T173709Z-QuantAgent-kkj.10-planner
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
