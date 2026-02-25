# Requirements: Remove wait_sec Deprecation

**Issue ID:** QuantAgent-lmn  
**Type:** Technical debt / deprecation cleanup  
**Priority:** Low  
**Status:** Planning

---

## Objective

Remove the deprecated `wait_sec` parameter from `invoke_with_retry()` function in `quantagent/agent_utils.py` and update all usages to use `base_wait` instead.

---

## Context

The codebase currently generates deprecation warnings:
```
2026-01-06 21:58:17,452 - quantagent.agent_utils - WARNING - wait_sec parameter is deprecated, use base_wait instead
```

The parameter was marked as deprecated in favor of `base_wait`, but the old parameter and its handling logic remain in the code.

---

## Scope

### In Scope
- Remove `wait_sec` parameter from function signature
- Remove deprecation warning logic (lines 133-140 in agent_utils.py)
- Update test cases using `wait_sec` to use `base_wait`
- Remove deprecation warning test case

### Out of Scope
- Any other refactoring or changes to retry logic
- Performance optimization
- Additional parameter changes

---

## Affected Files

1. `quantagent/agent_utils.py` — function signature and deprecation logic
2. `tests/test_agent_utils_retry.py` — 2 test cases + 1 deprecation test

---

## Definition of Done

- No more `wait_sec` parameter in code or tests
- All tests pass
- No deprecation warnings in logs
