# Planning: Use `with_structured_output` in `pattern_agent` and `trend_agent`

**Issue ID:** QuantAgent-6t4  
**Related:**
- [Requirements](../01_requirements/QuantAgent-6t4-RQ-structured-output-vision-agents.md)
- [Acceptance Criteria](../05_acceptance_tests/QuantAgent-6t4-AC-structured-output-vision-agents.md)

---

## Overview

This is a small consistency refactor. The work should stay local to the two vision-agent modules and a narrow set of tests.

**Estimated effort:** 30-45 minutes  
**Risk level:** Low  
**Complexity:** Small

---

## Tasks

### Task 1: Refactor `pattern_agent`
**Effort:** 10-15 minutes

**Actions:**
1. Replace the raw `graph_llm.invoke(...)` parsing path with `graph_llm.with_structured_output(PatternReport)`.
2. Preserve the existing Anthropic-compatible retry branch, but make both branches return a typed `PatternReport` directly.
3. Keep the existing fallback report creation for failure cases.

**Files:**
- `quantagent/pattern_agent.py`

### Task 2: Refactor `trend_agent`
**Effort:** 10-15 minutes

**Actions:**
1. Replace the raw `graph_llm.invoke(...)` parsing path with `graph_llm.with_structured_output(TrendReport)`.
2. Preserve the existing Anthropic-compatible retry branch, but make both branches return a typed `TrendReport` directly.
3. Keep the existing output shape and fallback report creation.

**Files:**
- `quantagent/trend_agent.py`

### Task 3: Add targeted regression coverage
**Effort:** 10-15 minutes

**Actions:**
1. Add tests that assert `with_structured_output(...)` is called with `PatternReport` and `TrendReport`.
2. Add tests that prove the nodes still return valid reports and preserve output shape.
3. Keep the tests narrowly scoped; do not rewrite the broader agent test suite.

**Files:**
- `tests/test_pattern_agent_refactor.py`
- `tests/test_trend_agent_refactor.py`

---

## Dependencies

- Existing `PatternReport` and `TrendReport` schemas in `quantagent/agent_models.py`
- Existing structured-output test fixtures in `tests/conftest.py`

---

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---:|---|
| Anthropic retry branch breaks while removing text parsing | Medium | Low | Keep current retry structure and only swap the LLM object being invoked |
| Tests become tautological | Medium | Medium | Assert schema usage and result contract instead of only mocked field echoes |
| Output shape drifts in `trend_agent` | Low | Low | Keep return dict unchanged and cover it with a focused test |

---

## Testing Strategy

### Implementer gates
1. `ruff check --fix .`
2. `python -m pytest tests/test_pattern_agent_refactor.py tests/test_trend_agent_refactor.py -v`
3. `python -m compileall -q quantagent tests`

### Optional confidence check
- Run a focused integration subset that instantiates the agent nodes from the graph-level tests.

---

## Rollout Plan

1. Write requirements + acceptance docs
2. Implement the minimal code change
3. Run targeted tests and compile checks
4. If green, hand off to tester or complete tester validation in the same branch
5. Integrate only after evidence shows the diff stayed in scope

---

## Validation Checklist

- [ ] Both agent modules use `with_structured_output(...)`
- [ ] Manual JSON parsing removed from both main analysis paths
- [ ] Fallback reports still validate
- [ ] Targeted tests pass
- [ ] Diff stays limited to docs, the two agent modules, and narrow regression tests
