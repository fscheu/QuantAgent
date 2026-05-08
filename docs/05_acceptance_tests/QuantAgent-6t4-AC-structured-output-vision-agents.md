# Acceptance Criteria: Use `with_structured_output` in `pattern_agent` and `trend_agent`

**Issue ID:** QuantAgent-6t4  
**Related:** [QuantAgent-6t4-RQ-structured-output-vision-agents.md](../01_requirements/QuantAgent-6t4-RQ-structured-output-vision-agents.md)

---

## Success Criteria

### 1. Pattern agent uses structured output
**Given** a pattern-agent node created with a vision-capable LLM  
**When** the node analyzes a state with an available pattern image  
**Then** it requests `PatternReport` through `with_structured_output(...)`  
**And** the returned `pattern_report` is a valid `PatternReport` instance.

### 2. Trend agent uses structured output
**Given** a trend-agent node created with a vision-capable LLM  
**When** the node analyzes a state with an available trend image  
**Then** it requests `TrendReport` through `with_structured_output(...)`  
**And** the returned `trend_report` is a valid `TrendReport` instance.

### 3. Manual JSON parsing is removed
**Given** the production source files for both agents  
**When** `pattern_agent.py` and `trend_agent.py` are inspected  
**Then** neither file relies on manual JSON parsing of LLM text responses for the main analysis path.

### 4. Existing fallback behavior is preserved
**Given** image generation fails or the structured-output LLM call raises  
**When** either agent runs  
**Then** it still returns a valid fallback report object with safe default values  
**And** the node does not crash.

### 5. Trend agent keeps its current output shape
**Given** the trend agent completes successfully  
**When** the node result is inspected  
**Then** it still includes:
- `trend_report`
- `trend_image`
- `trend_image_filename`
- `trend_image_description`

---

## Negative Cases

### 1. Wrong schema requested
**When** the pattern or trend agent requests the wrong schema  
**Then** the targeted tests fail.

### 2. Structured output not used
**When** the agent bypasses `with_structured_output(...)` and falls back to raw text parsing  
**Then** the targeted tests fail.

### 3. Fallback invalid
**When** the LLM path raises and the fallback report violates Pydantic constraints  
**Then** the targeted tests fail.

---

## Verification Commands

```bash
cd /tmp/autodev-worktrees/QuantAgent/QuantAgent-6t4/implementer-<timestamp>

ruff check --fix .
python -m pytest tests/test_pattern_agent_refactor.py tests/test_trend_agent_refactor.py -v
python -m pytest tests/test_integration_full_graph.py -k "Pattern Agent returns PatternReport or Trend Agent returns TrendReport" -v
python -m compileall -q quantagent tests
```

---

## Boundary Conditions

1. Precomputed image present: structured-output call should still work without regenerating the image.
2. Precomputed image absent: tool path should still generate the image before analysis.
3. Anthropic-style retry without `SystemMessage`: the structured-output path must keep working in that branch too.
