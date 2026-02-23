# Testing Patterns & Guidelines

This document defines testing standards for QuantAgent to ensure tests provide meaningful validation rather than false coverage.

## Core Principle

**Tests should validate behavior and structure, not mock outputs.**

If a test only validates that `mock_output == verified_output`, it's a tautology. It proves nothing about the code under test.

---

## Anti-Patterns (What NOT to Do)

### 1. Tautological Tests
❌ **BAD:**
```python
def test_bullish_scenario(self, mock_llm):
    # Mock the LLM to return bullish values
    mock_llm.invoke = Mock(return_value=Mock(content=json.dumps({
        "trend_direction": "bullish",
        "confidence": 0.9
    })))
    result = agent(state)
    # Verify it returns what we mocked
    assert result["report"].trend_direction == "bullish"
    assert result["report"].confidence == 0.9
```

**Why it's bad:**
- Only validates the mock, not the agent logic
- Doesn't test real LLM behavior
- A bug in the agent won't be caught (test still passes)
- Creates false sense of coverage

### 2. Excessive Mocking
❌ **BAD:**
```python
# Mock within mock within mock - loses sight of what's being tested
mock_llm = Mock()
mock_llm.bind_tools = Mock(return_value=Mock())
mock_llm.with_structured_output = Mock(return_value=Mock(
    invoke=Mock(return_value=Mock(...))
))
```

**Why it's bad:**
- Mocking the entire call chain
- Can't distinguish real failures from mock issues
- Difficult to maintain
- Reduces confidence in actual code

### 3. No Validation of Constraints
❌ **BAD:**
```python
def test_indicator_agent(self, mock_llm, mock_toolkit, sample_state):
    result = agent_node(sample_state)
    # Only checks it exists
    assert "indicator_report" in result
```

**Why it's bad:**
- Doesn't validate Pydantic constraints
- Doesn't catch out-of-range values
- Doesn't test fallback logic

### 4. Tests That Can't Fail
❌ **BAD:**
```python
def test_agent_handles_data(self, mock_llm):
    # This test passes regardless of what the code does
    try:
        result = agent(state)
        assert result is not None  # Always true
    except:
        assert True  # Also always true
```

---

## Valid Patterns (What TO Do)

### 1. Structure & Type Validation
✅ **GOOD:**
```python
def test_output_is_pydantic_model(self, mock_llm, mock_toolkit, sample_state):
    """Verify output is IndicatorReport instance with all required fields."""
    result = create_indicator_agent(mock_llm, mock_toolkit)(sample_state)
    report = result["indicator_report"]

    # Validate type
    assert isinstance(report, IndicatorReport)

    # Validate all fields exist
    required_fields = ["macd", "rsi", "trend_direction", "confidence"]
    for field in required_fields:
        assert hasattr(report, field)
        assert getattr(report, field) is not None
```

**Why it works:**
- Tests the actual contract (what type/structure is returned)
- Will fail if agent returns wrong type
- Independent of mock values

### 2. Constraint Validation
✅ **GOOD:**
```python
def test_rsi_within_valid_range(self, mock_llm, mock_toolkit, sample_state):
    """Verify Pydantic validation constrains RSI to 0-100."""
    result = create_indicator_agent(mock_llm, mock_toolkit)(sample_state)
    report = result["indicator_report"]

    assert 0 <= report.rsi <= 100
    assert 0.0 <= report.confidence <= 1.0
```

**Why it works:**
- Tests Pydantic field validators
- Independent of mock behavior
- Validates constraints are enforced

### 3. Error Handling & Fallback
✅ **GOOD:**
```python
def test_fallback_on_llm_failure(self, mock_toolkit, sample_state):
    """Verify agent returns valid fallback when LLM fails."""
    from unittest.mock import Mock

    # Mock LLM to raise error
    mock_llm = Mock()
    mock_llm.with_structured_output = Mock(side_effect=ValueError("LLM error"))

    agent_node = create_indicator_agent(mock_llm, mock_toolkit)
    result = agent_node(sample_state)
    report = result["indicator_report"]

    # Verify fallback is valid
    assert isinstance(report, IndicatorReport)
    assert report.confidence == 0.0  # Fallback confidence
    assert "failed" in report.reasoning.lower()
```

**Why it works:**
- Tests error path, not happy path
- Validates fallback mechanism works
- Independent of normal flow

### 4. State & Messages Preservation
✅ **GOOD:**
```python
def test_messages_preserved_in_state(self, mock_llm, mock_toolkit, sample_state):
    """Verify message history is preserved and system message included."""
    result = create_indicator_agent(mock_llm, mock_toolkit)(sample_state)

    messages = result["messages"]
    assert isinstance(messages, list)
    assert len(messages) > 0

    # Verify system message is included
    assert any(isinstance(msg, SystemMessage) for msg in messages)
    # Verify human message is included
    assert any(isinstance(msg, HumanMessage) for msg in messages)
```

**Why it works:**
- Tests state flow, not LLM output
- Validates message structure
- Independent of mock values

### 5. Edge Cases
✅ **GOOD:**
```python
def test_empty_kline_data(self, mock_llm, mock_toolkit):
    """Verify agent handles empty OHLCV data gracefully."""
    state = {
        "kline_data": {"timestamps": [], "opens": [], "highs": [], "lows": [], "closes": [], "volumes": []},
        "time_frame": "1hour",
        "stock_name": "BTC",
        "messages": []
    }

    result = create_indicator_agent(mock_llm, mock_toolkit)(state)
    report = result["indicator_report"]

    # Should still return valid report (possibly fallback)
    assert isinstance(report, IndicatorReport)
```

**Why it works:**
- Tests boundary conditions
- Validates robustness
- Independent of normal operation

### 6. Data Transformation
✅ **GOOD:**
```python
def test_timeframe_included_in_system_message(self, mock_llm, mock_toolkit, sample_state):
    """Verify timeframe is properly included in system message."""
    result = create_indicator_agent(mock_llm, mock_toolkit)(sample_state)

    messages = result["messages"]
    system_msg = next((m for m in messages if isinstance(m, SystemMessage)), None)

    assert system_msg is not None
    assert sample_state["time_frame"] in system_msg.content
```

**Why it works:**
- Tests data transformation
- Validates message construction
- Independent of mock behavior

---

## Testing Agents: Specific Guidelines

### For All Agents (indicator, pattern, trend, decision)

**DO test:**
- ✅ Output is correct Pydantic model (IndicatorReport, PatternReport, etc.)
- ✅ All required fields are present and not None
- ✅ Field values respect constraints (0-1 ranges, enum values, etc.)
- ✅ Fallback report is returned on LLM error
- ✅ Messages are properly constructed and preserved
- ✅ State fields are correctly extracted and used
- ✅ Edge cases (empty data, missing fields, extreme values)

**DON'T test:**
- ❌ Mock what the LLM should return for specific scenarios
- ❌ Validate specific analysis results (bullish vs bearish)
- ❌ Verify the LLM "understood" the analysis
- ❌ Test the mock instead of the agent

### For Agents with Tools (indicator_agent)

**DO test:**
- ✅ Tools list is properly defined
- ✅ Tools are bound to LLM (`bind_tools` called)
- ✅ System message includes tool instructions
- ✅ System message is properly formatted

**DON'T test:**
- ❌ That tools were actually called
- ❌ Specific indicator values from tools
- ❌ Tool computation logic (that's tool's test)

### For Agents with Vision (pattern_agent, trend_agent)

**DO test:**
- ✅ Image is included in messages
- ✅ Image is optional (uses precomputed or generates)
- ✅ Fallback works when vision fails

**DON'T test:**
- ❌ Chart generation (that's chart's test)
- ❌ Vision model accuracy
- ❌ Pattern/trend recognition results

---

## Test Suite Checklist

When creating tests for a new agent, include:

- [ ] **Structure Tests**
  - [ ] Output is correct Pydantic model
  - [ ] All required fields present
  - [ ] Can instantiate without mock

- [ ] **Constraint Tests**
  - [ ] Field ranges respected (0-1, 0-100, enums)
  - [ ] Invalid values properly validated
  - [ ] Fallback has valid constraints

- [ ] **Error Handling**
  - [ ] LLM failure → valid fallback
  - [ ] Malformed response → valid fallback
  - [ ] Missing fields → valid fallback

- [ ] **State Management**
  - [ ] Messages initialized correctly
  - [ ] Messages preserved in output
  - [ ] State fields properly used
  - [ ] System message includes key info

- [ ] **Edge Cases**
  - [ ] Empty/minimal data
  - [ ] Extreme values
  - [ ] Missing optional fields
  - [ ] Different parameter combinations

- [ ] **Message Structure** (if applicable)
  - [ ] SystemMessage properly formatted
  - [ ] HumanMessage includes data
  - [ ] Tools mentioned in system (if applicable)

---

## Quick Reference: Test Types

| Test Type | Validates | Mock Level | Value |
|-----------|-----------|-----------|-------|
| **Structure** | Type, fields exist | Full mock OK | High |
| **Constraints** | Pydantic validation | Full mock OK | High |
| **Error Handling** | Fallback mechanism | Error mock required | High |
| **State Flow** | Message/state preservation | Full mock OK | Medium |
| **Integration** | Real LLM behavior | Minimal mock | High |
| **Scenario** | Specific outcomes | Full mock | ❌ Low/useless |

**Most valuable:** Structure, Constraints, Error Handling, State Flow
**Least valuable:** Scenario tests with full mocking

---

## Example: Refactoring Bad Tests

### Before (Tautological)
```python
def test_bullish_scenario(self, mock_llm):
    mock_llm.invoke = Mock(return_value=Mock(content=json.dumps({
        "trend_direction": "bullish",
        "confidence": 0.9,
        ...
    })))
    result = create_indicator_agent(mock_llm, toolkit)(state)
    assert result["report"].trend_direction == "bullish"  # Only validates mock
```

### After (Meaningful)
```python
def test_output_structure(self, mock_llm, mock_toolkit, sample_state):
    """Verify agent returns valid IndicatorReport with all required fields."""
    result = create_indicator_agent(mock_llm, mock_toolkit)(sample_state)
    report = result["indicator_report"]

    assert isinstance(report, IndicatorReport)
    assert hasattr(report, "trend_direction")
    assert report.trend_direction in ["bullish", "bearish", "neutral"]

def test_confidence_constraint(self, mock_llm, mock_toolkit, sample_state):
    """Verify confidence is always in valid 0.0-1.0 range."""
    result = create_indicator_agent(mock_llm, mock_toolkit)(sample_state)
    report = result["indicator_report"]

    assert 0.0 <= report.confidence <= 1.0

def test_fallback_on_error(self, mock_toolkit, sample_state):
    """Verify fallback report is returned when LLM fails."""
    mock_llm = Mock()
    mock_llm.with_structured_output = Mock(side_effect=ValueError("LLM error"))

    result = create_indicator_agent(mock_llm, mock_toolkit)(sample_state)
    report = result["indicator_report"]

    assert isinstance(report, IndicatorReport)
    assert report.confidence == 0.0
```

Each test is **independent, meaningful, and can fail if code is broken.**

