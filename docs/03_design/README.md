# Design & Technical Specifications

This folder contains architecture, design patterns, technical decisions, and implementation specifications for the QuantAgent system.

## Reading Order

Agents working in this repository should read design documents in this order:

### Core Architecture (start here)
- [streamlit_app_architecture.md](./streamlit_app_architecture.md) - Streamlit application architecture overview
- [backtesting_engine.md](./backtesting_engine.md) - Backtesting engine design and flow
- [strategy_assembler_architecture.md](./strategy_assembler_architecture.md) - Strategy assembly pattern

### Component-Specific Design
- [MESSAGE_STATE_MANAGEMENT.md](./MESSAGE_STATE_MANAGEMENT.md) - LangGraph state management patterns
- [POSITION_MANAGEMENT_STRATEGIES.md](./POSITION_MANAGEMENT_STRATEGIES.md) - Position sizing and management
- [data_caching_architecture.md](./data_caching_architecture.md) - Market data caching strategy
- [langgraph_improvements.md](./langgraph_improvements.md) - LangGraph optimizations and patterns

### Technical Configuration & Patterns
- [CONFIGURATION.md](./CONFIGURATION.md) - Configuration management approach
- [LOGGING_STRATEGY.md](./LOGGING_STRATEGY.md) - Logging standards and patterns
- [TESTING_PATTERNS.md](./TESTING_PATTERNS.md) - Testing conventions and best practices

### Infrastructure & Tooling
- [docker_deployment.md](./docker_deployment.md) - Docker deployment setup
- [MIGRATIONS.md](./MIGRATIONS.md) - Database migration guide
- [dev-tools-setup.md](./dev-tools-setup.md) - Development tooling setup
- [workflow-integration.md](./workflow-integration.md) - Development workflow integration

## Current Truth

These documents represent the current technical design of the system. When implementing features:

- Follow patterns defined in relevant design documents
- Consult architecture docs before making structural changes
- Reference TESTING_PATTERNS.md for test conventions
- Check CONFIGURATION.md and LOGGING_STRATEGY.md for cross-cutting concerns

## Active Per-Change Design

- [QuantAgent-r78-DS-trade-pnl-calculation.md](./QuantAgent-r78-DS-trade-pnl-calculation.md) - Trade P&L calculation bug fix (P1)
- [QuantAgent-nu7-DS-active-position-monitoring.md](./QuantAgent-nu7-DS-active-position-monitoring.md) - Active Position Monitoring System (Epic: QuantAgent-nu7)
- [QuantAgent-7bn-DS-azure-openai-support.md](./QuantAgent-7bn-DS-azure-openai-support.md) - Azure OpenAI LLM provider support

## Per-Change Design Documents

Per-change design documents should follow this naming convention:

```
QuantAgent-<issue-id>-DS-<short-slug>.md
```

Example: `QuantAgent-a3f2dd-DS-agent-parallelization.md`

These files describe design deltas, architectural decisions, and technical specifications for specific changes linked to Beads issues (when applicable for system implementations).
