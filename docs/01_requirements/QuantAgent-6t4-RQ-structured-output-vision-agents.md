# Requirements: Use `with_structured_output` in `pattern_agent` and `trend_agent`

**Issue ID:** QuantAgent-6t4  
**Type:** Refactor  
**Priority:** P3

---

## Context

`indicator_agent.py` already uses LangChain's structured-output path:
1. bind tools when needed,
2. call `with_structured_output(...)`,
3. receive a typed Pydantic model directly.

`pattern_agent.py` and `trend_agent.py` still ask the vision LLM for JSON text, then strip markdown fences and parse the response manually with `json.loads(...)`.

That manual parsing adds avoidable failure modes and makes these two agents inconsistent with the rest of the agent pipeline.

---

## Objective

Refactor `pattern_agent.py` and `trend_agent.py` so both agents obtain `PatternReport` and `TrendReport` through `with_structured_output(...)` instead of manually parsing JSON strings from the vision LLM response.

---

## Scope

### In Scope
- Replace manual JSON parsing in `quantagent/pattern_agent.py`
- Replace manual JSON parsing in `quantagent/trend_agent.py`
- Preserve current return contract (`pattern_report`, `trend_report`, existing trend image fields)
- Preserve existing fallback behavior: agents must still return valid report objects when the vision/tool path fails
- Add or adjust tests that prove the new structured-output integration is actually used

### Out of Scope
- Refactoring `indicator_agent.py`
- Changing prompt semantics beyond what is required to support structured output
- Reworking toolkit image generation
- Changing graph orchestration or shared state shape
- Broad cleanup of existing agent tests unrelated to this issue

---

## Functional Requirements

1. `pattern_agent` must request a typed `PatternReport` from the vision-capable LLM instead of parsing JSON text.
2. `trend_agent` must request a typed `TrendReport` from the vision-capable LLM instead of parsing JSON text.
3. If the structured-output call fails, each agent must still return a valid fallback report with safe default values.
4. Existing non-parsing behavior must remain intact:
   - image generation still happens when the precomputed image is missing;
   - precomputed images are still reused when present;
   - `trend_agent` still returns the trend image metadata fields.

---

## Constraints

- Keep the change minimal and local to the two agent modules plus targeted tests.
- Reuse the existing `PatternReport` and `TrendReport` models; do not introduce new schemas.
- Do not add new abstraction layers or helper utilities for this one refactor.
- Match the existing `indicator_agent` pattern where that produces the smallest correct diff.

---

## Edge Cases

- Vision LLM raises before returning a structured object
- Vision LLM returns an invalid object or unexpected type
- Tool-based image generation fails and no image is available
- Anthropic compatibility fallback without `SystemMessage` still needs to work with structured output

---

## Definition of Done

- [ ] `pattern_agent.py` no longer imports or relies on `json.loads(...)` for LLM output parsing
- [ ] `trend_agent.py` no longer imports or relies on `json.loads(...)` for LLM output parsing
- [ ] Both agents call `with_structured_output(...)` with the correct schema
- [ ] Existing report contracts remain valid
- [ ] Targeted tests pass and fail meaningfully if structured output is not used
