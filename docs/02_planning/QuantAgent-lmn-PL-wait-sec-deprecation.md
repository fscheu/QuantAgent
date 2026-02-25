# Planning: Remove wait_sec Deprecation

**Issue ID:** QuantAgent-lmn  
**Related:** [QuantAgent-lmn-RQ-wait-sec-deprecation.md](../01_requirements/QuantAgent-lmn-RQ-wait-sec-deprecation.md)

---

## Tasks

### 1. Update Function Signature (30 min)

**File:** `quantagent/agent_utils.py`

- Remove `wait_sec: float | None = None` from line 100
- Update docstring to remove `wait_sec` parameter documentation (line 116)

### 2. Remove Deprecation Logic (15 min)

**File:** `quantagent/agent_utils.py`

- Delete lines 133-140 (deprecation warning and fallback logic)

### 3. Update Test Cases (20 min)

**File:** `tests/test_agent_utils_retry.py`

- Line 389: Change `wait_sec=4` to `base_wait=4`
- Line 400: Change `wait_sec=4` to `base_wait=4`
- Delete `test_deprecation_warning_for_wait_sec` function (lines ~393-401)

### 4. Validation (15 min)

- Run `pytest tests/test_agent_utils_retry.py -v`
- Verify no deprecation warnings
- Search codebase for remaining `wait_sec` references

---

## Dependencies

None — all changes are isolated to agent_utils.py and its tests.

---

## Risks

**Low risk:**
- Simple parameter removal
- Only 2 test usages to update
- No external callers (all use `base_wait` already)

---

## Testing Strategy

1. Run existing test suite for `agent_utils`
2. Verify no warnings in test output
3. Check that retry behavior remains unchanged

---

## Rollout

Single commit with all changes together (atomic change).

---

## Estimated Effort

**Total:** ~1.5 hours (including testing and validation)
