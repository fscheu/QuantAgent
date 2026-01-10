# Requirements

This folder contains functional requirements and UI specifications for the QuantAgent system.

## Reading Order

Agents working in this repository should read requirements documents in this order:

1. **Current System Requirements** (foundational)
   - [trading_system_requirements.md](./trading_system_requirements.md) - Core trading system functional requirements
   - [ui_streamlit_mvp_requirements.md](./ui_streamlit_mvp_requirements.md) - Streamlit UI MVP specifications

2. **Future Requirements** (reference)
   - [ideas_post_mvp.md](./ideas_post_mvp.md) - Post-MVP feature ideas and enhancements

## Current Truth

The files listed above represent the current functional requirements for the system. When implementing features:

- Verify requirements against these documents
- Ensure UI changes align with `ui_streamlit_mvp_requirements.md`
- Consider post-MVP ideas only when explicitly requested

## Active Per-Change Requirements

- [QuantAgent-r78-RQ-trade-pnl-calculation.md](./QuantAgent-r78-RQ-trade-pnl-calculation.md) - Trade P&L calculation bug fix (P1)
- [QuantAgent-nu7-RQ-active-position-monitoring.md](./QuantAgent-nu7-RQ-active-position-monitoring.md) - Active Position Monitoring System (Epic: QuantAgent-nu7)
- [QuantAgent-7bn-RQ-azure-openai-support.md](./QuantAgent-7bn-RQ-azure-openai-support.md) - Azure OpenAI LLM provider support

## Per-Change Requirements

Per-change requirement documents should follow this naming convention:

```
QuantAgent-<issue-id>-RQ-<short-slug>.md
```

Example: `QuantAgent-a3f2dd-RQ-fee-tracking.md`

These files describe requirement deltas introduced by specific changes and are linked to Beads issues (when applicable for system implementations).
