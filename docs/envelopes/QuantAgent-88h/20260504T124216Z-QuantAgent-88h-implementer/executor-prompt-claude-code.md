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
        - Run ID: 20260504T124216Z-QuantAgent-88h-implementer
        - Phase: implementer
        - Skill: autodev-implementer
        - Issue: QuantAgent-88h
        - Repo: /home/azureuser/repos/projects/QuantAgent
        - Artifacts dir: /home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-88h/20260504T124216Z-QuantAgent-88h-implementer

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
        run_id: "20260504T124216Z-QuantAgent-88h-implementer"
phase: "implementer"
executor: "claude-code"
repo_path: "/home/azureuser/repos/projects/QuantAgent"
beads_issue_id: "QuantAgent-88h"
branch_policy: "worktree-preferred"
base_branch: "main"
current_branch_at_generation: "main"
feature_branch: "feature/QuantAgent-88h-create-seed-data-script-for-dev-and-qa-d"
worktree_path: "/tmp/autodev-worktrees/QuantAgent/QuantAgent-88h/implementer-20260504T124216Z"
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
artifacts_dir: "/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-88h/20260504T124216Z-QuantAgent-88h-implementer"
generated_at: "2026-05-04T12:42:16.352439+00:00"
preflight:
  repo_dirty: true
  dirty_files:
    - " M .beads/issues.jsonl"
    - "?? docs/envelopes/QuantAgent-les/integration-decision-20260504T105500Z.md"
        ---

# Autodev Input Envelope — QuantAgent-88h — implementer

## Objective

Execute the `implementer` phase for Beads issue `QuantAgent-88h`: Create seed data script for DEV and QA databases.

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
- Beads issue: `QuantAgent-88h`.
- Labels at generation time: `openclaw:design_approved testing dx`.
- Artifact issue snapshot: `/home/azureuser/repos/projects/QuantAgent/docs/envelopes/QuantAgent-88h/20260504T124216Z-QuantAgent-88h-implementer/issue.json`.
- Existing docs under `docs/` for this issue, if present.

## Issue Description

El proyecto no tiene datos de prueba reproducibles. Cada desarrollador/autodev arranca con una base vacía, lo que dificulta testear casuísticas reales y reproduce bugs que solo aparecen con datos. Se necesita un script que genere un estado de base de datos completo y realista, ejecutable en DEV y QA cuantas veces sea necesario.

---

**Análisis de tablas por categoría:**

**MAESTROS** (configuración estática, igual en todos los entornos):
- `strategy_configs` — configuraciones de estrategias (RSI, MACD, Triple Screen). Seed: 3-4 registros con JSON de config real.

**DATOS BASE** (datos de mercado históricos, reales):
- `market_data` — OHLCV histórico. Descargar via yfinance al momento de ejecutar el seed.
  - BTC, timeframe 4h, últimos 180 días
  - AAPL, timeframe 1d, últimos 180 días
  - SPY, timeframe 1d, últimos 180 días
  Estos 3 assets/timeframes cubren el 90% de los tests y fixtures existentes.

**TRANSACCIONALES** (casuísticas de la app, sintéticos pero realistas):
Las tablas están encadenadas: `signals → orders → fills/trades → active_positions → backtest_runs`.

Casuísticas a generar:
1. Trade ganador completo: signal LONG → order FILLED → fill → trade cerrado con PnL positivo → active_position closed
2. Trade perdedor completo: misma cadena, PnL negativo
3. Trade abierto (en curso): signal + order FILLED + active_position sin cerrar (is_active=True) — testea lógica de monitoreo
4. Señal sin orden ejecutada: signal NEUTRAL, sin order asociado
5. Orden cancelada: order con status=CANCELLED, sin fill ni trade
6. Backtest run completo: 1 run con 10+ trades en active_positions, métricas calculadas (win_rate, sharpe_ratio, max_drawdown, profit_factor, total_pnl)
7. Backtest run en progreso: run creado sin métricas (total_trades=None), simula ejecución en curso

---

**Cambio requerido:**

Crear `scripts/seed_dev.py` que:
1. Acepta `--db-url` (default: lee de DATABASE_URL env var)
2. Acepta `--reset` flag para truncar las tablas antes de insertar (orden inverso de FK)
3. Crea los maestros (strategy_configs)
4. Descarga y carga market_data via yfinance para BTC/4h, AAPL/1d, SPY/1d
5. Genera los 7 escenarios transaccionales descriptos arriba
6. Imprime resumen al final: N registros por tabla

El script debe ser idempotente con `--reset`: correrlo dos veces con `--reset` produce el mismo estado.

**Criterio de aceptación:**
- [ ] `python scripts/seed_dev.py --reset` corre sin errores contra la DEV DB
- [ ] `python scripts/seed_dev.py --reset --db-url postgresql://qa_user:qa_pass@localhost:5433/quantagent_qa` corre sin errores contra QA DB
- [ ] Después del seed, `SELECT COUNT(*) FROM market_data` retorna > 500 rows
- [ ] Después del seed, todas las tablas transaccionales tienen al menos 1 registro
- [ ] Los 7 escenarios están presentes y son consultables (active_positions con is_active=True, trade cerrado con pnl>0, etc.)
- [ ] El script imprime resumen de registros por tabla al finalizar

**Archivos relevantes:**
- `scripts/seed_dev.py` — a crear
- `quantagent/models.py` — modelos SQLAlchemy con todas las tablas
- `quantagent/database.py` — get_session(), Base, engine setup
- `tests/conftest.py` — fixtures de referencia para estructura de datos esperada
- `.env` — DATABASE_URL default

**Fuera de scope (no tocar):**
- No modificar modelos existentes
- No agregar dependencias nuevas (yfinance ya está en pyproject.toml)
- No generar datos para `logs` (tabla de logging, no es datos de negocio)
- No generar datos para `fills` independientemente — siempre asociados a un order

**Notas técnicas:**
- Usar `PGPASSWORD=password psql ... -c "TRUNCATE ... CASCADE"` o SQLAlchemy `session.execute(text("TRUNCATE ... CASCADE"))` para el `--reset`
- El orden de truncado debe respetar FKs: active_positions → trades → fills → orders → signals → backtest_runs → strategy_configs → market_data
- Para market_data: yfinance `yf.download(ticker, period="6mo", interval="4h")` devuelve DataFrame listo para insertar
- Los campos `environment` en orders/signals/trades/active_positions deben ser "dev"


## Acceptance Criteria

No acceptance criteria found in Beads issue.

## Recent Beads Comments

- Comment 61 by Federico Scheu at 2026-04-21T19:07:34Z:

  ### Skill: autodev-planner
  - **Resultado:** SUCCESS
  - **Run-ID:** 20260421-190000Z

  #### Qué hice
  - Analicé los modelos SQLAlchemy (quantagent/models.py) para entender el schema completo
  - Identifiqué las tablas y sus relaciones FK: signals → orders → fills/trades → active_positions → backtest_runs
  - Diseñé estructura completa para `scripts/seed_dev.py` con 7 escenarios transaccionales
  - Creé documentación exhaustiva:
    - `docs/01_requirements/QuantAgent-88h-RQ-seed-data-script.md` — Requirements (6 ACs, scope completo)
    - `docs/03_design/QuantAgent-88h-DS-seed-data-script.md` — Design técnico (arquitectura, decisiones, ejemplos de código)
    - `docs/05_acceptance_tests/QuantAgent-88h-AC-seed-data-script.md` — Acceptance criteria detallados + queries SQL de validación
    - `docs/02_planning/QuantAgent-88h-PL-seed-data-script.md` — Planning (10 tareas, 9h total)

  #### Cambios en el ticket
  - No apliqué cambios de labels/estado (issue ya tiene `openclaw:design_approved testing dx`)

  #### Problemas encontrados
  - Ninguno — schema está bien definido, yfinance ya está disponible
  - Identificado orden correcto de truncado FK: active_positions → trades → fills → orders → signals → backtest_ru…

## Preflight Evidence

- Current branch at generation: `main`.
- Repo dirty at generation: `True`.
- Dirty files:
  - ` M .beads/issues.jsonl`
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
### Skill: autodev-implementer
- **Resultado:** SUCCESS | PARTIAL | BLOCKED | FAIL
- **Run-ID:** 20260504T124216Z-QuantAgent-88h-implementer
- **Executor:** claude-code

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
