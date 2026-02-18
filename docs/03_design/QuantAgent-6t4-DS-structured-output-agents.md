# QuantAgent-6t4 — Design: Structured output for vision agents

Links:
- Requirements: `docs/01_requirements/QuantAgent-6t4-RQ-structured-output-agents.md`

## Context
`indicator_agent.py` already uses `llm.bind_tools(...).with_structured_output(IndicatorReport)` to return typed output and avoid manual JSON parsing.

`pattern_agent.py` and `trend_agent.py` currently:
- call `graph_llm.invoke(...)`
- parse `final_response.content` as JSON (including markdown fence stripping)

## Proposed Change
### pattern_agent
- Create a structured wrapper for the LLM used for vision:
  - `structured_llm = graph_llm.with_structured_output(PatternReport)`
- Replace the `graph_llm.invoke(...)` call with `structured_llm.invoke(...)`.
- Expect the call to return a `PatternReport` instance directly.
- Keep the existing retry logic and the "retry without system message" fallback.

### trend_agent
- Mirror the same pattern:
  - `structured_llm = graph_llm.with_structured_output(TrendReport)`
  - Call `structured_llm.invoke(...)` (with the same fallback behavior).

## Validation / Fallback Rules
- If the LLM call fails or returns a non-model object, create a minimal valid `PatternReport` / `TrendReport` (consistent with `indicator_agent.py`’s validation style).
- Remove JSON extraction logic (markdown fence stripping + `json.loads`).

## Notes / Compatibility
- Prompts may still describe an "output shape" but should not require explicit JSON parsing in the client code.
- Anthropic compatibility fallback must remain: if the LLM errors on system messages, retry with only a `HumanMessage`.

### Example (minimal)
Illustrative shape only (not full code):

```python
structured_llm = graph_llm.with_structured_output(PatternReport)
pattern_report = invoke_with_retry(structured_llm.invoke, agent_messages, retries=3, wait_sec=8)
```
