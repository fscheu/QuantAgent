# QuantAgent-bdm — Requirements — Fix tests expecting shared state updates across multi-agent calls

## Objective
Align the test suite with the intended LangGraph state contract:
- Analysis agents (Indicator/Pattern/Trend) **return structured reports** and **do not update** shared `messages` state.
- Only the Decision Agent updates `messages`.

## Context
The repository design explicitly separates:
- **Inter-agent communication** via structured reports (`IndicatorReport`, `PatternReport`, `TrendReport`)
- **Conversation history** (`messages`) owned/updated by the Decision Agent only

See: `docs/03_design/MESSAGE_STATE_MANAGEMENT.md`.

## In Scope
- Update tests that:
  - call multiple agents (sequentially or via compiled graph) and expect `state["messages"]` to be updated by analysis agents
  - assert that all agents “produce messages”
- Tests should instead verify:
  - analysis agents return their respective structured report keys
  - `messages` is unchanged/absent until the Decision Agent runs
  - the compiled graph output includes `messages` **because** the Decision Agent ran

## Out of Scope
- Changes to production agent behavior/state schema (unless a true mismatch is discovered)
- Rewriting unrelated tests for style/coverage
- Changing model/provider behavior

## Constraints
- Keep changes minimal and localized to failing/incorrect assertions.
- Do not introduce test patterns that validate mocks instead of behavior (see `docs/03_design/TESTING_PATTERNS.md`).

## Definition of Done
- Tests no longer assume shared `messages` updates across analysis-agent calls.
- Tests assert structured report outputs for analysis agents.
- Final graph invocation tests assert presence of `messages` only at/after Decision Agent.
