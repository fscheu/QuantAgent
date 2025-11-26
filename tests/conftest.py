"""
Pytest configuration and shared fixtures for QuantAgent testing.

This module provides pytest fixtures for:
- Sample OHLCV data (kline_data)
- Mock LLM configurations (supporting with_structured_output pattern)
- Agent state objects
- Mocked API keys
- Temporary directories for test outputs
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from quantagent.agent_models import (
    IndicatorReport,
    PatternReport,
    TrendReport,
    TradingDecision,
)


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def sample_ohlcv_data() -> Dict[str, List[float]]:
    """
    Provide sample OHLCV data for testing.

    Returns:
        Dictionary with 'open', 'high', 'low', 'close', 'volume' keys
        containing 30 candlesticks of realistic price data.
    """
    np.random.seed(42)
    num_candles = 30

    # Generate realistic price movements
    prices = np.cumsum(np.random.randn(num_candles)) + 100

    return {
        "open": [float(p) for p in prices],
        "high": [float(p + abs(np.random.randn())) for p in prices],
        "low": [float(p - abs(np.random.randn())) for p in prices],
        "close": [float(p + np.random.randn() * 0.5) for p in prices],
        "volume": [float(1000000 + np.random.randint(-100000, 100000))
                   for _ in range(num_candles)],
    }


@pytest.fixture
def sample_ohlcv_dataframe(sample_ohlcv_data) -> pd.DataFrame:
    """
    Provide sample OHLCV data as a pandas DataFrame.

    Returns:
        DataFrame with datetime index and OHLCV columns.
    """
    df = pd.DataFrame(sample_ohlcv_data)
    df.index = pd.date_range(end=datetime.now(), periods=len(df), freq='1h')
    df.index.name = 'datetime'
    return df

@pytest.fixture
def sample_state_inicial() -> Dict:
    return {
        "kline_data": {
            "open": [100.0] * 30,
            "high": [101.0] * 30,
            "low": [99.0] * 30,
            "close": [100.5] * 30,
            "volume": [1000000.0] * 30,
        },
        "analysis_results": None,
        "messages": [],
        "time_frame": "1hour",
        "stock_name": "BTC",
        "pattern_image": "base64_encoded_image_data",
        "trend_image": "base64_encoded_image_data",
    }

@pytest.fixture
def sample_state() -> Dict:
    """
    Provide a minimal valid IndicatorAgentState for testing.

    Returns:
        Dictionary matching IndicatorAgentState structure with sample data.
    """
    return {
        "kline_data": {
            "open": [100.0] * 30,
            "high": [101.0] * 30,
            "low": [99.0] * 30,
            "close": [100.5] * 30,
            "volume": [1000000.0] * 30,
        },
        "time_frame": "1hour",
        "stock_name": "BTC",
        "rsi": [50.0] * 30,
        "macd": [0.0] * 30,
        "macd_signal": [0.0] * 30,
        "macd_hist": [0.0] * 30,
        "stoch_k": [50.0] * 30,
        "stoch_d": [50.0] * 30,
        "roc": [0.0] * 30,
        "willr": [-50.0] * 30,
        "indicator_report": "Sample indicator report",
        "pattern_image": "base64_encoded_image_data",
        "pattern_image_filename": "/tmp/pattern.png",
        "pattern_image_description": "Sample pattern description",
        "pattern_report": "Sample pattern report",
        "trend_image": "base64_encoded_image_data",
        "trend_image_filename": "/tmp/trend.png",
        "trend_image_description": "Sample trend description",
        "trend_report": "Sample trend report",
        "analysis_results": None,
        "messages": [],
        "decision_prompt": "Sample decision prompt",
        "final_trade_decision": "",
    }


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def mock_config() -> Dict:
    """
    Provide mock LLM configuration for testing.

    Returns:
        Dictionary with mock API keys and model settings.
    """
    return {
        "agent_llm_model": "gpt-4o-mini",
        "graph_llm_model": "gpt-4o",
        "agent_llm_provider": "openai",
        "graph_llm_provider": "openai",
        "agent_llm_temperature": 0.1,
        "graph_llm_temperature": 0.1,
        "api_key": "sk-test-key-12345",
        "anthropic_api_key": "sk-ant-test-key",
        "qwen_api_key": "sk-qwen-test-key",
    }


@pytest.fixture
def mock_env_vars(monkeypatch):
    """
    Set up mock environment variables for testing.

    Args:
        monkeypatch: pytest's monkeypatch fixture.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-qwen-test-key")


# ============================================================================
# LLM Mock Fixtures
# ============================================================================

class StructuredMockLLM:
    """
    Mock LLM supporting modern LangChain patterns.

    Supports both legacy invoke() and modern with_structured_output() patterns.
    Returns actual Pydantic model instances when used with structured output.
    """

    def __init__(self, default_overrides=None):
        """Initialize with optional field overrides."""
        self.default_overrides = default_overrides or {}

    def invoke(self, messages):
        return  "Mock LLM response"

    def with_structured_output(self, schema):
        """
        Modern pattern: return mock LLM that produces Pydantic instances.

        Args:
            schema: Pydantic model class (IndicatorReport, PatternReport, etc.)

        Returns:
            Mock LLM with invoke() that returns schema instance
        """
        structured_mock = Mock()

        # Create default values based on schema type
        if schema == IndicatorReport:
            defaults = {
                "macd": 0.5,
                "macd_signal": 0.3,
                "macd_histogram": 0.2,
                "rsi": 65.0,
                "rsi_level": "overbought",
                "roc": 2.5,
                "stochastic": 75.0,
                "willr": -25.0,
                "trend_direction": "bullish",
                "confidence": 0.75,
                "reasoning": "Mock indicator analysis"
            }
        elif schema == PatternReport:
            defaults = {
                "patterns_detected": ["double_bottom"],
                "primary_pattern": "double_bottom",
                "confidence": 0.75,
                "breakout_probability": 0.65,
                "reasoning": "Mock pattern detected"
            }
        elif schema == TrendReport:
            defaults = {
                "support_level": 99.5,
                "resistance_level": 101.5,
                "trend_direction": "upward",
                "trend_strength": 0.75,
                "reasoning": "Mock trend analysis"
            }
        elif schema == TradingDecision:
            defaults = {
                "decision": "LONG",
                "entry_price": 100.5,
                "take_profit": 102.5,
                "stop_loss": 98.5,
                "confidence": 0.75,
                "risk_level": "medium",
                "reasoning": "Mock trading decision"
            }
        else:
            defaults = {}

        # Merge with custom overrides
        defaults.update(self.default_overrides)

        # Create Pydantic instance
        instance = schema(**defaults)
        structured_mock.invoke = Mock(return_value=instance)

        return structured_mock

    def bind_tools(self, tools):
        """Mock tool binding."""
        return self


@pytest.fixture
def mock_llm():
    """
    Provide a StructuredMockLLM for agent testing.

    Supports both legacy and modern patterns.
    """
    return StructuredMockLLM()


@pytest.fixture
def mock_llm_custom():
    """
    Factory fixture for creating StructuredMockLLM with custom field values.

    Usage:
        def test_something(mock_llm_custom):
            llm = mock_llm_custom({"confidence": 0.9, "trend_direction": "bearish"})
    """
    def create_with_overrides(overrides):
        return StructuredMockLLM(default_overrides=overrides)
    return create_with_overrides


@pytest.fixture
def mock_openai_llm():
    """
    Provide a mock OpenAI LLM for testing.

    Returns:
        StructuredMockLLM object configured like OpenAI LLM.
    """
    return StructuredMockLLM()


@pytest.fixture
def mock_anthropic_llm():
    """
    Provide a mock Anthropic LLM for testing.

    Returns:
        StructuredMockLLM object configured like Anthropic LLM.
    """
    return StructuredMockLLM()


@pytest.fixture
def mock_vision_llm():
    """
    Provide a mock vision-capable LLM for pattern/trend analysis.

    Returns:
        StructuredMockLLM object configured for vision tasks.
    """
    return StructuredMockLLM()


# ============================================================================
# Toolkit Mock Fixtures
# ============================================================================

class MockToolkit:
    """Mock toolkit with indicator and chart generation functions."""

    def __init__(self):
        """Initialize all tool mocks with default return values."""
        self.compute_macd = Mock(name="compute_macd")
        self.compute_rsi = Mock(name="compute_rsi")
        self.compute_roc = Mock(name="compute_roc")
        self.compute_stoch = Mock(name="compute_stoch")
        self.compute_willr = Mock(name="compute_willr")

        # Setup default return values for tool invocation
        self.compute_macd.invoke = Mock(return_value={
            "macd": 0.5,
            "signal": 0.3,
            "histogram": 0.2
        })
        self.compute_rsi.invoke = Mock(return_value={"rsi": 65.0})
        self.compute_roc.invoke = Mock(return_value={"roc": 2.5})
        self.compute_stoch.invoke = Mock(return_value={"stochastic": 75.0})
        self.compute_willr.invoke = Mock(return_value={"willr": -25.0})
        self.generate_kline_image = Mock(name="generate_kline_image")
        self.generate_kline_image.invoke = Mock(return_value={"pattern_image": "mock_b64_encoded_png_pattern"})
        self.generate_trend_image = Mock(name="generate_trend_image")
        self.generate_trend_image.invoke = Mock(return_value={"trend_image": "mock_b64_encoded_png_trend"})

    # def generate_kline_image(self):
    #     """Mock K-line image generation tool."""
    #     mock_tool = Mock()
    #     mock_tool.invoke = Mock(return_value={
    #         "pattern_image": "mock_b64_encoded_png_pattern"
    #     })
    #     return mock_tool

    # def generate_trend_image(self):
    #     """Mock trend image generation tool."""
    #     mock_tool = Mock()
    #     mock_tool.invoke = Mock(return_value={
    #         "trend_image": "mock_b64_encoded_png_trend"
    #     })
    #     return mock_tool


@pytest.fixture
def mock_toolkit():
    """
    Provide a MockToolkit with all indicator and chart functions.

    Used by indicator_agent, pattern_agent, and trend_agent tests.
    """
    return MockToolkit()


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """
    Provide a temporary directory for test outputs.

    Args:
        tmp_path: pytest's tmp_path fixture.

    Returns:
        Path object pointing to temporary output directory.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def temp_chart_dir(tmp_path) -> Path:
    """
    Provide a temporary directory for chart outputs.

    Args:
        tmp_path: pytest's tmp_path fixture.

    Returns:
        Path object pointing to temporary chart directory.
    """
    chart_dir = tmp_path / "charts"
    chart_dir.mkdir()
    return chart_dir


# ============================================================================
# Mocking Utilities
# ============================================================================

@pytest.fixture
def patch_yfinance():
    """
    Patch yfinance to avoid real API calls during testing.

    Yields:
        MagicMock configured to return sample OHLCV data.
    """
    with patch("yfinance.download") as mock_download:
        # Create sample dataframe
        dates = pd.date_range(end=datetime.now(), periods=30, freq='1h')
        mock_df = pd.DataFrame({
            "Open": np.linspace(100, 105, 30),
            "High": np.linspace(101, 106, 30),
            "Low": np.linspace(99, 104, 30),
            "Close": np.linspace(100.5, 105.5, 30),
            "Volume": [1000000] * 30,
        }, index=dates)

        mock_download.return_value = mock_df
        yield mock_download


@pytest.fixture
def patch_talib():
    """
    Patch TA-Lib functions to avoid errors during testing without TA-Lib.

    Yields:
        Dictionary of mocked indicator functions.
    """
    mocked_indicators = {
        "RSI": lambda data, period=14: np.array([50.0] * len(data)),
        "MACD": lambda data, fastperiod=12, slowperiod=26, signalperiod=9: (
            np.array([0.0] * len(data)),
            np.array([0.0] * len(data)),
            np.array([0.0] * len(data)),
        ),
        "STOCH": lambda high, low, close, fastk_period=5, slowk_period=3, slowd_period=3: (
            np.array([50.0] * len(close)),
            np.array([50.0] * len(close)),
        ),
        "ROC": lambda data, period=12: np.array([0.0] * len(data)),
        "WILLR": lambda high, low, close, period=14: np.array([-50.0] * len(close)),
    }

    with patch("talib.RSI", side_effect=mocked_indicators["RSI"]), \
         patch("talib.MACD", side_effect=mocked_indicators["MACD"]), \
         patch("talib.STOCH", side_effect=mocked_indicators["STOCH"]), \
         patch("talib.ROC", side_effect=mocked_indicators["ROC"]), \
         patch("talib.WILLR", side_effect=mocked_indicators["WILLR"]):
        yield mocked_indicators


# ============================================================================
# Session-scoped Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """
    Provide the project root directory path.

    Returns:
        Path object pointing to the project root.
    """
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def benchmark_data_dir(project_root) -> Path:
    """
    Provide path to benchmark data directory.

    Returns:
        Path object pointing to benchmark/ directory.
    """
    benchmark_dir = project_root / "benchmark"
    if not benchmark_dir.exists():
        pytest.skip("Benchmark data directory not found")
    return benchmark_dir


# ============================================================================
# Pytest Configuration Hooks
# ============================================================================

def pytest_configure(config):
    """
    Configure pytest with custom markers and settings.

    Args:
        config: pytest Config object.
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests requiring API calls"
    )
    config.addinivalue_line(
        "markers", "vision: marks tests using vision-capable LLMs"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to apply default markers.

    Args:
        config: pytest Config object.
        items: List of test items.
    """
    for item in items:
        # Mark tests with 'integration' in the name
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark tests with 'api' in the name
        if "api" in item.nodeid:
            item.add_marker(pytest.mark.api)

        # Mark tests with 'vision' in the name
        if "vision" in item.nodeid:
            item.add_marker(pytest.mark.vision)
