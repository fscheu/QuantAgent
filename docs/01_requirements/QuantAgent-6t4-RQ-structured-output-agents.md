# QuantAgent-6t4 — Requirements: Structured output for pattern_agent and trend_agent

## Objective
Replace manual JSON parsing in `pattern_agent.py` and `trend_agent.py` with LangChain / LangGraph LLM `with_structured_output(...)`, returning typed `PatternReport` / `TrendReport` objects (matching the existing approach in `indicator_agent.py`).

## Scope
- `quantagent/pattern_agent.py`
  - Use `graph_llm.with_structured_output(PatternReport)` for the vision LLM call.
  - Remove manual `json.loads(...)` parsing of `final_response.content`.
- `quantagent/trend_agent.py`
  - Use `graph_llm.with_structured_output(TrendReport)` for the vision LLM call.
  - Remove manual `json.loads(...)` parsing of `final_response.content`.

## Non-scope
- Changes to:
  - `PatternReport` / `TrendReport` schemas in `quantagent/agent_models.py`
  - Prompt wording (beyond minimal adjustments needed to support structured output)
  - Tooling behavior (image generation, retries) other than wiring the structured output call
  - Any downstream agents / graph wiring unless required for type compatibility

## Constraints
- Must keep changes minimal and localized to the two agents.
- Must preserve existing retry behavior and the Anthropic compatibility fallback (retry without system message).
- Returned values must remain valid instances of the corresponding Pydantic models even on failure (fallback report is acceptable).

## Definition of Done
- Both agents obtain `PatternReport` / `TrendReport` via `with_structured_output(...)` (no manual JSON parsing).
- Public behavior is preserved:
  - The node returns the same state keys (e.g. `pattern_report`, `trend_report`, plus existing trend image fields).
  - Fallback behavior still produces a valid report.
