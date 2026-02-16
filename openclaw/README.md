# OpenClaw Autodev (QuantAgent)

This folder contains the first iteration of an autonomous dev pipeline that integrates:
- **bd** (Beads) as the backlog / gate mechanism
- OpenClaw skills (LLM) for analysis + design generation
- A Python orchestrator script for robust execution

## Gate mechanism (v1)
- Orchestrator picks items from `bd ready`.
- When a design is generated, the orchestrator sets the issue to **blocked** and adds label `openclaw:design_pending`.
- To approve a design, set the issue back to **open** and add label `openclaw:design_approved` (or remove the pending label).

## Run
```bash
python3 openclaw/scripts/dev_orchestrator.py --limit 3
```

## Notes
- Design docs are committed to the `epic/openclaw-dev` branch (local by default).
- Pushing to origin is intentionally manual for now.
