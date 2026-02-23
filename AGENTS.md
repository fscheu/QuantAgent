# AGENTS.md

This file provides guidance to coding agents when working in this repository.

---

## Project Overview

**QuantAgent** is a multi-agent trading analysis system that uses vision-capable LLMs (Claude, GPT-4, Qwen) to analyze financial markets in high-frequency trading (HFT) contexts. It combines technical indicators, candlestick pattern recognition, and trend analysis through a LangGraph-orchestrated agent pipeline.

**Reference Paper:** arXiv:2509.09995 - "Price-Driven Multi-Agent LLMs for High-Frequency Trading"

---

## Repository Structure

This repository is organized around three main axes:
1. Core trading / backtesting engine (`quantagent/`)
2. Applications and interfaces (`apps/`)
3. Documentation, planning, and operational context (`docs/`, `.beads/`)

---

### quantagent/ — Core Engine & Domain Logic

Main Python package containing all trading, backtesting, and agent logic.

#### Agent & Graph Core
- `agent_state.py` — LangGraph shared state schema
- `agent_models.py` — Agent-related domain models
- `agent_utils.py` — Shared utilities for agents
- `graph_setup.py` — LangGraph StateGraph definition
- `trading_graph.py` — Main orchestrator (LLM providers, graph execution)
- `default_config.py` / `settings.py` — Configuration defaults

#### Market Analysis Agents
- `indicator_agent.py` — Technical indicators (RSI, MACD, etc.)
- `pattern_agent.py` — Candlestick pattern detection (vision LLM)
- `trend_agent.py` — Trend & support/resistance analysis
- `decision_agent.py` — Final LONG / SHORT / HOLD decision

#### Backtesting
- `backtesting/`
  - `backtest.py` — Backtest execution engine

#### Strategy & Assembly
- `strategy/`
  - `assembler.py` — StrategyAssembler (factory/builder for trading components)

#### Trading Execution
- `trading/`
  - `order_manager.py`
  - `paper_broker.py`
  - `position_sizer.py`
  - `risk_manager.py`

#### Portfolio
- `portfolio/`
  - `manager.py` — Portfolio state and PnL tracking

#### Data Layer
- `data/`
  - `provider.py` — Market data providers (yfinance, etc.)

#### Persistence
- `database.py` — SQLAlchemy engine & session
- `models.py` — ORM models (Order, Trade, Signal, etc.)

---

### apps/ — User Interfaces

End-user applications built on top of the core engine.

#### Streamlit App
- `apps/streamlit/`
  - `app.py` — Main entry point
  - `views/` — UI pages (dashboard, backtesting, configuration, replay, etc.)
  - `services/` — DB and backend services
  - `utils/` — UI helpers

#### Flask App (legacy / demo)
- `apps/flask/`
  - `web_interface.py` — Flask demo app
  - `templates/`, `static/`, `assets/` — UI assets

---

### docs/ — Project Documentation (Source of Truth)

Structured documentation, divided by intent. Read "Documentation as a Single System" in this document down

---

### tests/ — Test Suite

- Unit, integration, and regression tests
- Testing conventions and setup described in:
  - `tests/README.md`
  - `docs/03_technical/TESTING_PATTERNS.md`

---

### examples/ — Usage Examples
- `run_backtest.py` — Example script for executing backtests

---

### alembic/ — Database Migrations
- `versions/` — Schema migration scripts
- `env.py` — Alembic environment configuration

---

### .beads/ — Operational Memory (Beads)

- `beads.db` — Local Beads database
- `config.yaml` — Beads configuration
- Used as the **task DAG and long-term memory** for work in progress

---

### Root Configuration & Tooling

- `AGENTS.md` — Agent operating rules (this file)
- `CLAUDE.md` — Claude-specific agent rules (mirrors AGENTS.md)
- `docker-compose.yml`, `Dockerfile` — Local infrastructure
- `requirements*.txt`, `pyproject.toml` — Python dependencies
- `.env`, `.env.example` — Environment configuration
- `.vscode/`, `.github/` — IDE and CI configuration

---

## ULTRAIMPORTANT

Think carefully and only action the specific task I have given you with the most concise and elegant solution that changes as little code as possible.
Avoid over-engineering. Only make changes that are directly requested or clearly necessary. Keep solutions simple and focused.

Don't add features, refactor code, or make "improvements" beyond what was asked. A bug fix doesn't need surrounding code cleaned up. A simple feature doesn't need extra configurability.

Don't add error handling, fallbacks, or validation for scenarios that can't happen. Trust internal code and framework guarantees. Only validate at system boundaries (user input, external APIs). Don't use backwards-compatibility shims when you can just change the code.

Don't create helpers, utilities, or abstractions for one-time operations. Don't design for hypothetical future requirements. The right amount of complexity is the minimum needed for the current task. Reuse existing abstractions where possible and follow the DRY principle.

Do not write tests with excessive mockups that do not test any solution code at all.

---

## Working System: Roles, Artifacts, and Handoffs (MVP)

This repo uses a **multi-agent workflow** based on **explicit artifacts** and **clear handoffs**. Agents should not rely on chat history as the source of truth; they should read and write the artifacts below.

There is **a single documentation tree** under `docs/`.  
No parallel documentation systems are used.

---

## Documentation as a Single System (`docs/`)

The documentation lives in `docs/` folder.

It serves two complementary purposes:
1. **Durable documentation** (architecture, requirements, guides)
2. **Operational, per-change artifacts** produced by agents and tracked via Beads

When generating documentation always divide between analysis or functional documentation and technical specification. Mainly, do not mix both in the same file, unless when a technical detail helps in the functional explanation.

Fallback to update existing documents. Try to not generate new documents all the time. We should have a main document for each one of the types described next and detail documents for specific requirements or technical decisions/implementations. The main type of documents should be:
- Requirements: Functional and UI specifications.
- Planning or task management: details of which are the plans and current status
- Technical specification: Architecture, Design, Decisions, Implementation, technical configurations.

Both live in the same tree and follow the same rules.

Standards:
1. Prefer updating existing docs over creating new ones
2. Link between related documents; **do not repeat** the same content in multiple places
3. Code comments only when logic isn’t self-evident

---

### Documentation Structure

```
docs/
├─ 01_requirements/
│ ├─ README.md
│ ├─ trading_system_requirements.md
│ ├─ QuantAgent-a3f2dd-RQ-<short-slug>.md
│ └─ QuantAgent-b91c02-RQ-<short-slug>.md
│
├─ 02_planning/
│ ├─ README.md
│ ├─ phase1_roadmap.md
│ └─ QuantAgent-a3f2dd-PL-<short-slug>.md
│
├─ 03_design/
│ ├─ README.md
│ ├─ architecture.md
│ ├─ TESTING_PATTERNS.md
│ └─ QuantAgent-a3f2dd-DS-<short-slug>.md
│
├─ 04_decisions/
│ ├─ README.md
│ └─ QuantAgent-a3f2dd-DC-<short-slug>.md
│
├─ 05_acceptance_tests/
│ ├─ README.md
│ └─ QuantAgent-a3f2dd-AC-<short-slug>.md
│
├─ 06_implementation/
│ ├─ README.md
│ └─ QuantAgent-a3f2dd-IM-<short-slug>.md
```

---

- `docs/01_requirements/README.md`  
  Current functional requirements index + links to per-change requirement files.
- `docs/02_planning/README.md`  
  Current roadmaps and phases.
- `docs/03_design/README.md`  
  Current technical design index + links to per-change design deltas.
- `docs/04_decisions/README.md`  
  Decision log index + per-change ADR-like entries.
- `docs/05_acceptance_tests/README.md`  
  Acceptance criteria / oracles index + per-change entries (Given/When/Then).
- `docs/06_implementation/README.md`  
  Implementation notes index + per-change deltas.

---

### Folder Semantics

Each documentation folder follows the same pattern:

- `README.md`  
  The **entry point and current truth** for that topic.
  Agents must read this file first.

- `QuantAgent-<issue-id>-<doc-type-key>-<short-slug>.md`  
  A **per-change artifact**, linked to a Beads issue.
  These files are **append-only** and describe deltas introduced by that issue.

Folders represent **types of knowledge**, not workflow steps.

---

### Versioning & Precedence Rules

- Per-change files must be prefixed with the related Beads issue ID.
- Agents must never retroactively rewrite historical per-issue files.
- In case of conflict:
  1. Folder `README.md`
  2. Most recent per-issue file (by Beads ID / timestamp)
  3. Older documentation

If ambiguity remains, the agent must stop and ask.

---

## Beads (bd) Usage

Beads (`bd`) is the **memory + task DAG** for ongoing work.

### Core rules
- Every non-trivial change should map to a **Beads issue**.
- Each issue may produce one or more documentation files under `docs/`,
  prefixed with the same issue ID.
- Agents should query Beads to understand status and dependencies,
  and consult `docs/` for context and decisions.

### Typical commands
- `bd status` — current state
- `bd ready` — work that is unblocked and ready
- `bd doctor` — setup health checks

> Note: This repo uses the **`bd`** CLI (not `beads`).

---

## Commit Policy (Human-in-the-loop)

Default policy: **agents do not commit to main**.

### Allowed workflow
- Agent may implement code and prepare a **commit plan** (grouping + messages + how to test).
- Agent may create a **feature branch** (optional, if you explicitly want it):
  - Branch naming: `feature/<issue-id>-<short-slug>` (e.g., `feature/QuantAgent-a3f2dd-config-ui`)
  - Agent commits only on that feature branch.
- Human reviews, runs tests, and integrates (merge/rebase) to main.

### What a commit/PR must include
- What changed (1–3 bullets)
- Why (linked to requirement/design/issue)
- How to test (exact commands)
- Any deviations from `design/` or `decisions/`

---

## Session Completion (Landing the Plane) — adjusted to Human Gate

When ending a work session:

1. **Update Beads state**
   - Create issues for remaining work
   - Close or update in-progress issues

2. **Update operational context (if changed)**
   - Add per-issue artifact deltas if needed (requirements/design/implementation)

3. **Quality gates**
   - Run relevant tests/lint/build for the changes you made (exact commands documented in your commit/PR notes)

4. **Integration**
   - Agent: prepare a commit plan and/or feature branch commits (if allowed)
   - Human: review, run tests, and push

CRITICAL RULE:
- Work is not “done” until there is a clear handoff: issue status + changes integrated (by human gate).
