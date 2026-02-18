# QuantAgent-6t4 — Plan: Use with_structured_output in pattern/trend agents

Links:
- Requirements: `docs/01_requirements/QuantAgent-6t4-RQ-structured-output-agents.md`
- Design: `docs/03_design/QuantAgent-6t4-DS-structured-output-agents.md`
- Acceptance: `docs/05_acceptance_tests/QuantAgent-6t4-AC-structured-output-agents.md`

## Tasks
1. Inspect current `indicator_agent.py` structured output pattern (baseline).
2. Update `quantagent/pattern_agent.py`
   - Replace `graph_llm.invoke(...)` + JSON parsing with `graph_llm.with_structured_output(PatternReport).invoke(...)`.
   - Keep retry + Anthropic fallback behavior.
   - Remove now-unused `json` import and parsing helpers.
3. Update `quantagent/trend_agent.py`
   - Replace `graph_llm.invoke(...)` + JSON parsing with `graph_llm.with_structured_output(TrendReport).invoke(...)`.
   - Keep retry + Anthropic fallback behavior.
   - Remove now-unused `json` import and parsing helpers.
4. Sanity-check typing and runtime behavior
   - Ensure the node return payload keys are unchanged.
   - Ensure fallback reports are still produced on failures.
5. Validation
   - Run formatting/lint (repo standard).
   - Run tests/smoke checks relevant to agents.

## How to validate (commands)
- Unit/integration suite (if available):
  - `pytest`
- Basic import check:
  - `python -c "from quantagent.pattern_agent import create_pattern_agent; from quantagent.trend_agent import create_trend_agent"`

## Risks / Dependencies
- Dependency: `graph_llm` implementation must support `.with_structured_output(...)` for vision calls.
- If structured output is not supported by the selected provider/model for multimodal messages, the change may require a minimal prompt adjustment or a model/provider constraint (escalate; do not silently change behavior).
