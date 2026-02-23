# pytest Framework Setup - Complete

## Summary

The pytest testing framework has been successfully configured for QuantAgent with a comprehensive conftest.py and supporting documentation.

## Files Created

### Core Testing Files

1. **`tests/conftest.py`** (11 KB)
   - 15+ reusable fixtures for testing
   - Data fixtures: `sample_ohlcv_data`, `sample_ohlcv_dataframe`, `sample_state`
   - Configuration fixtures: `mock_config`, `mock_env_vars`
   - LLM mocking: `mock_openai_llm`, `mock_anthropic_llm`, `mock_vision_llm`
   - Temporary directories: `temp_output_dir`, `temp_chart_dir`
   - API mocking: `patch_yfinance`, `patch_talib`
   - Utility fixtures: `project_root`, `benchmark_data_dir`

2. **`tests/__init__.py`**
   - Package initialization file

3. **`tests/test_example.py`** (7 KB)
   - Example tests demonstrating fixture usage
   - 7 test classes with 20+ test cases
   - Covers data fixtures, configuration, LLMs, directories, and markers

4. **`tests/README.md`** (6 KB)
   - Complete testing guide
   - Fixture documentation
   - Usage examples
   - Coverage instructions
   - Troubleshooting guide

### Configuration Files

5. **`pytest.ini`**
   - Test discovery configuration
   - Custom markers (slow, integration, api, vision)
   - Coverage settings
   - Output formatting options

6. **`pyproject.toml`** (updated)
   - Added pytest configuration section
   - Added pytest-mock to dev dependencies
   - Coverage configuration
   - Filter warnings settings

### Documentation Files

7. **`docs/dev-tools-setup.md`** (updated)
   - New section: "Testing Framework"
   - Detailed explanation of pytest
   - conftest.py structure and fixtures
   - How fixtures work
   - Available fixtures in QuantAgent

8. **`docs/workflow-integration.md`** (updated)
   - New section: "2.5. Testing Workflow (Local)"
   - How to run tests locally
   - Testing workflow diagram
   - Fixtures usage examples
   - Test markers explanation
   - Updated CI/CD section with pytest
   - Updated complete workflow
   - Updated Best Practices

## Quick Start

### 1. Install Testing Dependencies

```bash
pip install -r requirements-dev.txt
```

This installs:
- pytest>=7.4
- pytest-cov>=4.1
- pytest-mock>=3.11

### 2. Run Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_example.py

# Run only fast tests (exclude 'slow')
pytest -m "not slow"

# Run with coverage report
pytest --cov=quantagent --cov-report=html
```

### 3. Available Fixtures

```python
# Data fixtures
def test_with_data(sample_ohlcv_data):
    """Use sample OHLCV data"""
    assert len(sample_ohlcv_data["close"]) == 30

# Configuration fixtures
def test_with_config(mock_config, mock_env_vars):
    """Use mock config and env vars"""
    assert mock_config["agent_llm_provider"] in ["openai", "anthropic", "qwen"]

# Temporary directories
def test_with_temp_dir(temp_output_dir):
    """Use temporary directory for outputs"""
    output_file = temp_output_dir / "output.txt"
    output_file.write_text("test")
    assert output_file.exists()

# Mocked APIs
def test_with_mocked_yfinance(patch_yfinance):
    """Use mocked yfinance (no real API calls)"""
    import yfinance
    df = yfinance.download("BTC-USD", period="30h")
    assert len(df) == 30
```

## Test Markers

Use markers to categorize and filter tests:

```bash
pytest -m slow              # Only slow tests
pytest -m "not slow"        # Exclude slow tests
pytest -m integration       # Only integration tests
pytest -m api               # Only API-dependent tests
pytest -m vision            # Only vision LLM tests
pytest -m "api and not slow"  # Combine markers
```

**Marker Types:**
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.integration` - Multi-component tests
- `@pytest.mark.api` - Tests requiring external APIs
- `@pytest.mark.vision` - Tests using vision-capable LLMs

## Workflow Integration

### Local Development

```
1. Write code + tests
2. VSCode auto-formats (Black/isort)
3. Run: pytest -v
4. Pre-commit hooks verify
5. git commit
```

### Remote (GitHub Actions)

```
1. git push
2. GitHub Actions runs:
   - Black check
   - isort check
   - Flake8
   - Pylint
   - mypy
   - pytest (with coverage)
3. Coverage report to Codecov
4. PR status: ✅ or ❌
```

## Configuration Details

### `conftest.py` Structure

```
conftest.py
├── Data Fixtures (OHLCV, DataFrames, State)
├── Configuration Fixtures (Config, Environment)
├── LLM Mock Fixtures (OpenAI, Anthropic, Vision)
├── Temporary Directory Fixtures
├── Mocking Utilities (yfinance, TA-Lib)
├── Session-scoped Fixtures (project_root, benchmark_dir)
└── Pytest Configuration Hooks (markers, collection)
```

### `pytest.ini` Configuration

```ini
[pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = -v --strict-markers --tb=short
markers = [slow, integration, api, vision]
```

### `pyproject.toml` Pytest Section

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
addopts = "-v --strict-markers --tb=short"
markers = ["slow", "integration", "api", "vision"]

[tool.coverage.run]
source = ["quantagent"]
omit = ["*/tests/*", "*/__init__.py"]
```

## Coverage Reports

Generate and view coverage:

```bash
# Generate HTML report
pytest --cov=quantagent --cov-report=html

# View in terminal
pytest --cov=quantagent --cov-report=term-missing

# Exclude specific markers
pytest --cov=quantagent -m "not integration"
```

Reports are generated in `htmlcov/` directory.

## Documentation References

- **`tests/README.md`** - Complete testing guide with examples
- **`docs/dev-tools-setup.md`** - pytest explanation and conftest details
- **`docs/workflow-integration.md`** - Testing in development workflow

## Next Steps

### For Developers

1. Read `tests/README.md` for detailed usage
2. Look at `test_example.py` for fixture examples
3. Run `pytest -v` to see all available tests
4. Start writing tests for new features

### For CI/CD

The GitHub Actions workflow already includes pytest:
- Runs all tests
- Generates coverage reports
- Uploads to Codecov
- Blocks PR merge if tests fail

### For Team

- Use fixtures from `conftest.py` to avoid repetitive setup
- Run `pytest -m "not slow"` for quick local feedback
- Follow the testing workflow in development
- Add markers to categorize new tests

## Common Commands

```bash
# Quick testing
pytest                          # Run all tests
pytest -v                       # Verbose
pytest -k "rsi"                # Run tests matching "rsi"
pytest tests/test_example.py    # Specific file
pytest --cov=quantagent        # With coverage

# Filtering
pytest -m "not slow"            # Exclude slow
pytest -m integration           # Only integration
pytest -m "api and not slow"    # Combined

# Debugging
pytest -v --tb=long            # Long traceback
pytest --pdb                    # Drop to debugger on fail
pytest --lf                     # Run last failed

# Reporting
pytest --cov=quantagent --cov-report=html
pytest --cov=quantagent --cov-report=term-missing
```

## Troubleshooting

**pytest not found:**
```bash
pip install -r requirements-dev.txt
```

**Fixtures not working:**
```bash
pytest --fixtures  # List all available fixtures
```

**Tests discovery issues:**
```bash
pytest --collect-only  # See what tests are discovered
```

**Import errors:**
```bash
pip install -e .  # Install package in development mode
```

## Support

- Pytest docs: https://docs.pytest.org/
- pytest-mock: https://github.com/pytest-dev/pytest-mock
- pytest-cov: https://github.com/pytest-dev/pytest-cov
- Local guide: `tests/README.md`

---

**Setup Date:** 2025-11-25
**Status:** ✅ Complete and Ready to Use
