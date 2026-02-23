# QuantAgent Test Suite

This directory contains the test suite for QuantAgent, a multi-agent trading analysis system using vision-capable LLMs.

## Quick Start

### Installation

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Or install from pyproject.toml extras
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_example.py

# Run specific test class
pytest tests/test_example.py::TestSampleOHLCVData

# Run specific test function
pytest tests/test_example.py::TestSampleOHLCVData::test_sample_ohlcv_data_structure
```

## Available Fixtures

All fixtures are defined in `conftest.py` and are automatically available to all test modules.

### Data Fixtures

- **`sample_ohlcv_data`**: Dict with 30 OHLCV candlesticks (open, high, low, close, volume)
- **`sample_ohlcv_dataframe`**: Pandas DataFrame with datetime index and OHLCV data
- **`sample_state`**: Valid `IndicatorAgentState` object with all required fields

### Configuration Fixtures

- **`mock_config`**: Mock LLM configuration with test API keys
- **`mock_env_vars`**: Sets mock environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, DASHSCOPE_API_KEY)

### LLM Fixtures

- **`mock_openai_llm`**: MagicMock for OpenAI LLM
- **`mock_anthropic_llm`**: MagicMock for Anthropic LLM
- **`mock_vision_llm`**: MagicMock for vision-capable LLM

### Directory Fixtures

- **`temp_output_dir`**: Temporary directory for test outputs (auto-cleanup)
- **`temp_chart_dir`**: Temporary directory for chart outputs (auto-cleanup)

### Mocking Fixtures

- **`patch_yfinance`**: Patches yfinance.download to return mock data
- **`patch_talib`**: Patches TA-Lib functions with mock implementations

### Utility Fixtures

- **`project_root`**: Path object to project root directory (session-scoped)
- **`benchmark_data_dir`**: Path to `benchmark/` directory (session-scoped)

## Test Markers

Use markers to categorize and filter tests:

```bash
# Run only slow tests
pytest -m slow

# Run all tests except slow tests
pytest -m "not slow"

# Run integration tests
pytest -m integration

# Run tests requiring API calls
pytest -m api

# Run tests using vision LLMs
pytest -m vision

# Combine markers
pytest -m "api and not slow"
```

## Example Usage

### Test with Sample Data

```python
def test_indicator_calculation(sample_ohlcv_data):
    """Test indicator computation with sample data."""
    from graph_util import TechnicalTools

    tools = TechnicalTools()
    rsi = tools.compute_rsi(sample_ohlcv_data["close"])
    assert len(rsi) == 30
    assert all(0 <= val <= 100 for val in rsi)
```

### Test with Mock State

```python
def test_agent_processing(sample_state):
    """Test agent processing with sample state."""
    from trading_graph import TradingGraph

    graph = TradingGraph()
    result = graph.graph.invoke(sample_state)
    assert "final_trade_decision" in result
```

### Test with Mock LLM

```python
def test_llm_response(mock_openai_llm):
    """Test LLM invocation with mock."""
    response = mock_openai_llm.invoke("test prompt")
    assert response == "Mock LLM response"
    mock_openai_llm.invoke.assert_called_once()
```

### Test with Temporary Directory

```python
def test_chart_generation(temp_chart_dir, sample_ohlcv_data):
    """Test chart generation in temporary directory."""
    from graph_util import TechnicalTools

    tools = TechnicalTools()
    chart_path = temp_chart_dir / "chart.png"

    # Generate chart
    tools.generate_kline_chart(
        sample_ohlcv_data,
        str(chart_path)
    )

    assert chart_path.exists()
```

### Test with Mocked yfinance

```python
@pytest.mark.api
def test_data_fetching(patch_yfinance):
    """Test data fetching with mocked yfinance."""
    import yfinance

    df = yfinance.download("BTC-USD", period="30h")
    assert len(df) == 30
    assert "Open" in df.columns
```

## Coverage

Run tests with coverage reporting:

```bash
# Generate coverage report
pytest --cov=quantagent --cov-report=html

# View coverage in terminal
pytest --cov=quantagent --cov-report=term-missing

# Exclude specific markers from coverage
pytest --cov=quantagent -m "not integration"
```

## Writing New Tests

### File Naming Convention

- Test files: `test_<module>.py` or `<module>_test.py`
- Test classes: `Test<ModuleName>`
- Test functions: `test_<feature>`

### Best Practices

1. **Use appropriate fixtures**: Don't repeat fixture-like setup code
2. **Mark integration/API tests**: Use `@pytest.mark.integration` or `@pytest.mark.api`
3. **Use assertions clearly**: Make test intent obvious
4. **Minimal test scope**: Each test should test one thing
5. **Avoid real API calls**: Use `patch_yfinance` and other mocks

### Example Test Structure

```python
import pytest

class TestNewModule:
    """Tests for new_module.py"""

    def test_feature_basic(self, sample_ohlcv_data):
        """Test basic feature functionality."""
        # Arrange
        expected = 30

        # Act
        result = len(sample_ohlcv_data["close"])

        # Assert
        assert result == expected

    @pytest.mark.integration
    def test_feature_with_dependencies(self, sample_state, temp_output_dir):
        """Test feature integration with other components."""
        # Arrange
        output_file = temp_output_dir / "output.txt"

        # Act
        # ... test code ...

        # Assert
        assert output_file.exists()
```

## Troubleshooting

### TA-Lib Not Found
If TA-Lib installation fails, tests with `patch_talib` fixture will handle it gracefully. For full functionality:

```bash
conda install -c conda-forge ta-lib
```

### Import Errors
Ensure the quantagent package is installed in development mode:

```bash
pip install -e .
```

### Fixture Not Found
Run pytest with verbose output to see available fixtures:

```bash
pytest --fixtures
```

## CI/CD Integration

For GitHub Actions or other CI systems:

```yaml
- name: Run tests
  run: |
    pip install -r requirements-dev.txt
    pytest --cov=quantagent --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Pytest Markers](https://docs.pytest.org/en/stable/how-to-use-pytest.html#mark)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
