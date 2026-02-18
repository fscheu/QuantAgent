# QuantAgent-6t4 — Acceptance Criteria: Structured output for pattern/trend agents

Links:
- Requirements: `docs/01_requirements/QuantAgent-6t4-RQ-structured-output-agents.md`

## Pattern agent
- Given `pattern_agent_node(...)` runs with a valid `pattern_image` in state
  When the vision LLM is invoked
  Then the node obtains a `PatternReport` via `with_structured_output(PatternReport)` (no manual JSON parsing of `.content`).

- Given the vision LLM response cannot be produced (error/timeout)
  When `pattern_agent_node(...)` completes
  Then `state["pattern_report"]` is still present and is a valid `PatternReport` instance (fallback allowed).

- Given the first attempt with `[SystemMessage, HumanMessage]` fails due to system-message incompatibility
  When the agent retries
  Then it retries the structured output invocation with only `[HumanMessage]`.

## Trend agent
- Given `trend_agent_node(...)` runs with a valid `trend_image` in state
  When the vision LLM is invoked
  Then the node obtains a `TrendReport` via `with_structured_output(TrendReport)` (no manual JSON parsing of `.content`).

- Given the vision LLM response cannot be produced (error/timeout)
  When `trend_agent_node(...)` completes
  Then `state["trend_report"]` is still present and is a valid `TrendReport` instance (fallback allowed).

- Given the first attempt with `[SystemMessage, HumanMessage]` fails due to system-message incompatibility
  When the agent retries
  Then it retries the structured output invocation with only `[HumanMessage]`.

## Regression / Invariants
- Given existing downstream code expects the same state keys
  When the agents complete
  Then the returned dict keys remain unchanged (only the internal parsing mechanism changes).
