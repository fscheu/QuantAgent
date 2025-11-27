# QuantAgent Trading System Documentation

Welcome! This folder contains all documentation for transforming QuantAgent into a production trading system.

---

## 📚 Quick Navigation

### For First Time Reading
1. **Start here**: [Phase 1 Roadmap](02_planning/phase1_roadmap.md) - Overview of what we're building and timeline
2. **Then read**: [Trading System Requirements](01_requirements/trading_system_requirements.md) - Detailed specs
3. **Database setup**: [MIGRATIONS.md](MIGRATIONS.md) - DB setup and Alembic migrations
4. **Understand decisions**: [Technical Decisions](03_technical/) - How we're building it

### By Role

**Project Manager / Product Owner**:
- [Phase 1 Roadmap](02_planning/phase1_roadmap.md) - Timeline, milestones, deliverables
- [Trading System Requirements](01_requirements/trading_system_requirements.md) - What's being built

**Backend Engineer**:
- [Phase 1 Roadmap](02_planning/phase1_roadmap.md) - Week-by-week tasks
- [Data Caching Architecture](03_technical/data_caching_architecture.md) - DB design
- [Docker Deployment](03_technical/docker_deployment.md) - Dev environment setup
- [MIGRATIONS](MIGRATIONS.md) - Alembic workflow and DB models

**QA / Tester**:
- [Trading System Requirements](01_requirements/trading_system_requirements.md) - Acceptance criteria
- [Phase 1 Roadmap](02_planning/phase1_roadmap.md) - Testing milestones

**DevOps / Infrastructure**:
- [Docker Deployment](03_technical/docker_deployment.md) - Container setup
- [Phase 2 Roadmap](02_planning/phase2_roadmap.md) - Future deployment (FastAPI+Angular)

---

## 📁 Documentation Structure

### 01_requirements/
**Purpose**: Functional specifications and requirements

- **[trading_system_requirements.md](01_requirements/trading_system_requirements.md)**
  - What we're building (detailed specs)
  - Acceptance criteria for each feature
  - API contracts and data models
  - Success criteria for MVP

### 02_planning/
**Purpose**: Roadmaps, timelines, and task management

- **[phase1_roadmap.md](02_planning/phase1_roadmap.md)**
  - Timeline: Week 1-10
  - Week-by-week tasks
  - Key decisions made
  - Go/no-go criteria

- **[phase2_roadmap.md](02_planning/phase2_roadmap.md)**
  - Future features (real broker, FastAPI+Angular, macro agents)
  - Timeline: Weeks 11-20+
  - Advanced features
  - Expected improvements

### 03_technical/
**Purpose**: Technical architecture and implementation decisions

- **[docker_deployment.md](03_technical/docker_deployment.md)**
  - Lightweight Docker setup (PostgreSQL in container)
  - Development workflow
  - Production deployment approach
  - Troubleshooting

- **[data_caching_architecture.md](03_technical/data_caching_architecture.md)**
  - How we cache market data locally
  - Database schema for OHLC data
  - Implementation details
  - Performance characteristics

- **[ui_framework_decision.md](03_technical/ui_framework_decision.md)**
  - Why Streamlit for MVP
  - Why FastAPI+Angular for Phase 2
  - Hybrid migration strategy
  - Code examples

Note on UI: Current UI in this repo is Flask (`apps/flask/web_interface.py`). Streamlit dashboard is proposed for MVP and is marked as Planned in the UI document.

Language: Documentation is bilingual (English/Spanish). Dev tools and workflow guides are in Spanish; most architecture and planning docs are in English.

---

## 🎯 Key Decisions Summary

### 5 Major Decisions Made

| Decision | Choice | Why |
|----------|--------|-----|
| **Data Caching** | Local DB cache | Backtesting 10x faster |
| **MVP Scope** | Paper trading + Backtesting | Validate before execution |
| **Future Agents** | Macro agents in Phase 2 | Don't overcomplicate MVP |
| **Docker** | Lightweight setup | Portability + corporate friendly |
| **UI Framework** | Streamlit → FastAPI+Angular | Fast MVP, professional Phase 2 |

---

## 📊 Timeline at a Glance

```
Phase 1: MVP Paper Trading (8-10 weeks)
├─ Week 1-2:  Database + Infrastructure
├─ Week 3-4:  Portfolio + Risk Management
├─ Week 5-6:  Paper Broker + Order Execution
├─ Week 7-8:  Backtesting + Data Caching
└─ Week 9-10: Streamlit Dashboard + Integration

Phase 2: Production Ready (8-10 weeks, after MVP validation)
├─ Week 1-2:  Real Broker Integration
├─ Week 3-4:  Real-Time Data Pipeline
├─ Week 5-6:  Advanced Risk Management
├─ Week 7-8:  FastAPI + Angular Dashboard
└─ Week 9-20: Macro Analysis Agents
```

---

## ✅ Success Criteria

### Phase 1 MVP Done When:

**Automated Paper Trading**:
- ✅ Analysis runs every 1 hour
- ✅ Trades execute automatically
- ✅ Portfolio tracks correctly
- ✅ Risk limits enforced

**Strategy Validated**:
- ✅ Backtest: Win rate ≥ 40%
- ✅ Backtest: Sharpe ratio ≥ 1.0
- ✅ Backtest: Max drawdown ≤ 15%

**Operations Ready**:
- ✅ Uptime ≥ 99%
- ✅ Complete audit trail
- ✅ Dashboard shows metrics
- ✅ Full documentation

---

## 🔗 Related Files

**Main Repository**:
- [CLAUDE.md](../CLAUDE.md) - Claude Code guidance for this project
- [README.md](../README.md) - Project README

---

## 📞 Questions?

Each document has its own focus:
- **"What are we building?"** → [trading_system_requirements.md](01_requirements/trading_system_requirements.md)
- **"When will it be done?"** → [phase1_roadmap.md](02_planning/phase1_roadmap.md)
- **"How are we building it?"** → Documents in [03_technical/](03_technical/)
- **"What's the timeline?"** → [phase1_roadmap.md](02_planning/phase1_roadmap.md)

---

## 📈 Next Steps

1. **Read** [Phase 1 Roadmap](02_planning/phase1_roadmap.md) for full timeline
2. **Understand** [Trading System Requirements](01_requirements/trading_system_requirements.md) for specs
3. **Start coding** Week 1 tasks (Database + Infrastructure)
4. **Follow** week-by-week tasks in roadmap

---

**Last Updated**: November 2024
**Status**: Phase 1 planning complete, ready for implementation

