# Decisions

This folder contains architectural decision records (ADR-like entries) documenting significant technical choices and their rationale.

## Reading Order

Agents working in this repository should review decision documents chronologically to understand the evolution of technical choices:

### Current Decisions
- [ui_framework_decision.md](./ui_framework_decision.md) - UI framework selection (Streamlit vs alternatives)

## Current Truth

Decision documents capture the context, alternatives considered, and rationale for significant technical choices. When working on related features:

- Review relevant decision docs to understand constraints and trade-offs
- Respect decisions unless there's a compelling reason to revisit them
- Document new significant decisions in this folder

## Decision Document Format

Decision documents should include:

1. **Context** - What problem are we solving?
2. **Options Considered** - What alternatives were evaluated?
3. **Decision** - What was chosen and why?
4. **Consequences** - What are the implications?
5. **Status** - Accepted, superseded, or deprecated

## Active Per-Change Decisions

- [QuantAgent-69d-DC-reuse-logs-for-llm-telemetry.md](./QuantAgent-69d-DC-reuse-logs-for-llm-telemetry.md) - Choose existing `logs` over new metrics tables for this P3 feature

## Per-Change Decision Documents

Per-change decision documents should follow this naming convention:

```
QuantAgent-<issue-id>-DC-<short-slug>.md
```

Example: `QuantAgent-a3f2dd-DC-caching-strategy.md`

These files describe decision deltas and architectural choices for specific changes linked to Beads issues (when applicable for system implementations).
