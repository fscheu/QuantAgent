---
run_id: "20260526T080715Z-QuantAgent-kkj.11-planner"
phase: "planner"
executor: "auto"
repo_path: "/tmp/quantagent-main-clean-20260526T080519Z"
beads_issue_id: "QuantAgent-kkj.11"
branch_policy: "in-place-publication"
base_branch: "main"
publication_branch: "main"
current_branch_at_generation: "main"
feature_branch: null
worktree_root: null
worktree_path: null
shared_venv: null
shared_python: "/home/azureuser/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11"
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
artifacts_dir: "/tmp/quantagent-main-clean-20260526T080519Z/docs/envelopes/QuantAgent-kkj.11/20260526T080715Z-QuantAgent-kkj.11-planner"
generated_at: "2026-05-26T08:07:15.662730+00:00"
preflight:
  repo_dirty: false
  dirty_files: []
---

# Autodev Input Envelope — QuantAgent-kkj.11 — planner

## Objective

Execute the `planner` phase for Beads issue `QuantAgent-kkj.11`: Configurar routing multi-provider por rol para estrategias costo-eficientes.

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

- Repo instructions: `/tmp/quantagent-main-clean-20260526T080519Z/AGENTS.md` and `/tmp/quantagent-main-clean-20260526T080519Z/CLAUDE.md` if present.
- Beads issue: `QuantAgent-kkj.11`.
- Labels at generation time: `configuration, cost, llm`.
- Artifact issue snapshot: `/tmp/quantagent-main-clean-20260526T080519Z/docs/envelopes/QuantAgent-kkj.11/20260526T080715Z-QuantAgent-kkj.11-planner/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

## Contexto

QuantAgent hoy tiene una configuración de providers demasiado rígida para el caso de uso real que estamos empujando.

Hallazgos verificados en código actual:
- `quantagent/settings.py` sólo define dos slots globales de LLM: `AGENT_LLM_PROVIDER` y `GRAPH_LLM_PROVIDER`, con sus modelos y temperaturas asociados.
- `TradingGraph` inicializa exactamente dos LLMs (`agent_llm` y `graph_llm`) a partir de esos settings globales (`quantagent/trading_graph.py`).
- Backtesting y scheduler persisten metadata de un solo provider/modelo principal por corrida o señal (`quantagent/backtesting/backtest.py`, `quantagent/trading/scheduler.py`).
- La UI/validación todavía está acoplada a providers concretos y además está desfasada respecto al backend: en Configuration aparecen `openai`, `anthropic`, `qwen`, mientras el backend también soporta `azure`.
- No existe hoy una abstracción de "rol de provider" orientada a costo/performance, por ejemplo:
  - provider de razonamiento profundo;
  - provider barato/rápido para tareas livianas;
  - provider para tareas multimodales o de imagen.
- Tampoco existe configuración por estrategia para decidir qué rol usa cada etapa. Hoy el sistema opera más como "un provider global para casi todo". Elegante como un martillo para arreglar relojes.

Impacto:
- Se dificulta optimizar costos porque no se puede enrutar tareas simples a un modelo barato y reservar el caro para reasoning pesado.
- No hay forma limpia de expresar que una estrategia use distintos providers según etapa o capacidad requerida.
- Los seeds/presets de configuración quedan limitados porque no hay un modelo durable de routing multi-provider que valga la pena persistir.

## Cambio requerido

Diseñar e implementar una configuración multi-provider por rol/capacidad, persistible y consumible por estrategias, backtesting y paper trading.

El ticket debe cubrir como mínimo:

1. Introducir una abstracción de roles de provider, con al menos estos roles configurables:
   - `deep_reasoning_provider`
   - `lite_provider`
   - `image_provider`

2. Definir una fuente de verdad versionada/persistible para esos roles y sus modelos asociados, incluyendo como mínimo:
   - provider
   - model_name
   - temperature
   - timeout/retries opcionales
   - capability tags opcionales (`reasoning`, `cheap`, `vision`, `image`, etc.)

3. Implementar un resolver/registry central para providers soportados y sus capacidades efectivas, evitando listas hardcodeadas divergentes entre UI, settings y runtime.

4. Permitir que estrategias y/o corridas expresen qué rol consumen por etapa. Mínimo aceptable:
   - una estrategia puede declarar o recibir qué rol usar para reasoning principal;
   - una estrategia o pipeline multimodal puede declarar qué rol usar para imagen/visión;
   - existe fallback explícito cuando un rol no está configurado.

5. Mantener compatibilidad hacia atrás con la configuración actual durante la transición:
   - si sólo existen `AGENT_LLM_PROVIDER` y `GRAPH_LLM_PROVIDER`, el sistema sigue arrancando;
   - el nuevo sistema puede mapear esos defaults legacy a roles razonables.

6. Exponer esta configuración en la capa de Configuration/presets de forma persistible, para que después pueda ser seed-eada y reutilizada por environment.

7. Registrar en metadata de señales/runs qué provider/modelo/rol se usó realmente, para trazabilidad de costo y debugging.

## Acceptance Criteria

- [ ] Existe una abstracción central de roles de provider con al menos `deep_reasoning_provider`, `lite_provider` e `image_provider`.
- [ ] Existe una única fuente de verdad para providers soportados y la UI no diverge del runtime.
- [ ] El sistema mantiene backward compatibility con la config legacy basada en `AGENT_LLM_PROVIDER` / `GRAPH_LLM_PROVIDER`.
- [ ] Al menos una estrategia o pipeline puede resolverse usando roles distintos para tareas distintas, sin hardcodear el mismo provider para todo.
- [ ] Existe fallback claro cuando un rol requerido no está configurado o el provider no soporta la capacidad pedida.
- [ ] La configuración multi-provider es persistible y no vive sólo en variables de entorno efímeras o `session_state`.
- [ ] Backtests/señales/logs dejan trazabilidad suficiente para saber qué rol/provider/modelo se usó en runtime.
- [ ] Hay tests unitarios de resolución de roles/providers/fallbacks y al menos un test de integración liviano del wiring.
- [ ] La documentación operativa explica cómo configurar una estrategia costo-eficiente usando provider deep reasoning + lite + image.

## Archivos relevantes

- `quantagent/settings.py`
- `quantagent/trading_graph.py`
- `quantagent/backtesting/backtest.py`
- `quantagent/trading/scheduler.py`
- `apps/streamlit/views/configuration.py`
- `apps/streamlit/app.py`
- `apps/flask/web_interface.py`
- `quantagent/strategy/` 
- `tests/`

## Fuera de scope (no tocar)

- No integrar en este ticket providers nuevos que todavía no tengan soporte runtime real en QuantAgent si eso requiere SDKs/adapters grandes.
- No migrar credenciales sensibles a otro sistema secreto/config manager.
- No rediseñar todas las estrategias existentes para explotar al máximo el routing nuevo; alcanza con dejar el contrato y una adopción mínima verificable.
- No prometer optimización automática de costo basada en benchmark real si no existe medición todavía.

## Notas técnicas

- Importante separar dos cosas que suelen mezclarse y terminar en barro:
  1. catálogo de providers soportados;
  2. política de routing por rol.
  No son lo mismo.
- `image_provider` debe interpretarse como rol para tareas que requieran capacidad multimodal/imagen en el pipeline. Si hoy algunas imágenes se generan localmente con utilidades de charting y no con IA, eso no invalida el rol: simplemente hay que dejar claro cuándo aplica y cuándo no.
- Si aparece soporte futuro para LiteLLM, GitHub Copilot, Azure Foundry u otros gateways, debe entrar a través del registry/resolver central y no como otro `if provider == ...` desparramado por el repo.
- Este ticket conversa directamente con `QuantAgent-kkj.10`: primero conviene tener persistencia/routing real; después seedear presets más ricos deja de ser maquillaje.


## Acceptance Criteria

No acceptance criteria found in Beads issue.

## Recent Beads Comments

- Comment 149 by Federico Scheu at 2026-05-26T07:59:38Z:

  ### Skill: tech-lead-autodev
  - **Resultado:** BLOCKED
  - **Run-ID:** 20260526T075727Z-QuantAgent-kkj.11-planner
  
  #### Qué hice
  - Generé envelope de planner para `QuantAgent-kkj.11`.
  - Intenté correr el planner portable con `--executor auto` desde un checkout aislado y limpio.
  
  #### Bloqueo
  - `run_executor.py` rechazó la ejecución porque la publicación canónica del planner exige generarse desde la branch de publicación `main`.
  - Este cron estaba operando desde `integration/QuantAgent-kkj.8-20260526T074618Z` para no mezclar el root checkout dirty con la integración de `QuantAgent-kkj.8`.
  - El root checkout en `main` sigue sucio por artifacts no trackeados de un run previo, así que no era seguro rerutear ahí dentro de esta misma corrida.
  
  #### Evidencia
  - Artifact bloqueado: `docs/envelopes/QuantAgent-kkj.11/20260526T075727Z-QuantAgent-kkj.11-planner/`
  - Error del router: `Canonical planner publication requires generation from the publication branch. Generated on 'integration/QuantAgent-kkj.8-20260526T074618Z', expected 'main'.`
  
  #### Next step recomendado
  - Limpiar/normalizar un checkout `main` dedicado para planner publication y rerun de `QuantAgent-kkj.11`.

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
- **Run-ID:** 20260526T080715Z-QuantAgent-kkj.11-planner
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
