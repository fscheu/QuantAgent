"""
Example tests demonstrating the use of conftest fixtures.

This module shows how to use the various fixtures provided by conftest.py
for testing QuantAgent components.
"""

import pytest


class TestSampleOHLCVData:
    """Tests demonstrating fixture usage for OHLCV data."""

    def test_sample_ohlcv_data_structure(self, sample_ohlcv_data):
        """
        Test that sample OHLCV data has the expected structure.

        Args:
            sample_ohlcv_data: Fixture providing OHLCV dictionary.
        """
        assert "open" in sample_ohlcv_data
        assert "high" in sample_ohlcv_data
        assert "low" in sample_ohlcv_data
        assert "close" in sample_ohlcv_data
        assert "volume" in sample_ohlcv_data

    def test_sample_ohlcv_data_length(self, sample_ohlcv_data):
        """
        Test that sample data contains 30 candlesticks.

        Args:
            sample_ohlcv_data: Fixture providing OHLCV dictionary.
        """
        assert len(sample_ohlcv_data["open"]) == 30
        assert len(sample_ohlcv_data["close"]) == 30

    def test_sample_ohlcv_dataframe_index(self, sample_ohlcv_dataframe):
        """
        Test that sample dataframe has datetime index.

        Args:
            sample_ohlcv_dataframe: Fixture providing OHLCV dataframe.
        """
        assert sample_ohlcv_dataframe.index.name == "datetime"
        assert len(sample_ohlcv_dataframe) == 30


class TestMockConfiguration:
    """Tests demonstrating fixture usage for configuration."""

    def test_mock_config_has_required_keys(self, mock_config):
        """
        Test that mock configuration has all required keys.

        Args:
            mock_config: Fixture providing mock configuration.
        """
        required_keys = [
            "agent_llm_model",
            "graph_llm_model",
            "agent_llm_provider",
            "graph_llm_provider",
            "agent_llm_temperature",
            "graph_llm_temperature",
        ]
        for key in required_keys:
            assert key in mock_config

    def test_mock_config_provider_values(self, mock_config):
        """
        Test that provider values are valid.

        Args:
            mock_config: Fixture providing mock configuration.
        """
        assert mock_config["agent_llm_provider"] in ["openai", "anthropic", "qwen"]
        assert mock_config["graph_llm_provider"] in ["openai", "anthropic", "qwen"]

    def test_mock_env_vars_set(self, mock_env_vars):
        """
        Test that mock environment variables are properly set.

        Args:
            mock_env_vars: Fixture setting environment variables.
        """
        import os
        assert os.getenv("OPENAI_API_KEY") == "sk-test-openai-key"
        assert os.getenv("ANTHROPIC_API_KEY") == "sk-ant-test-key"


class TestAgentState:
    """Tests demonstrating fixture usage for agent state."""

    def test_sample_state_structure(self, sample_state):
        """
        Test that sample state has required fields.

        Args:
            sample_state: Fixture providing IndicatorAgentState.
        """
        required_fields = [
            "kline_data",
            "time_frame",
            "stock_name",
            "rsi",
            "macd",
            "indicator_report",
            "final_trade_decision",
        ]
        for field in required_fields:
            assert field in sample_state

    def test_sample_state_kline_data_valid(self, sample_state):
        """
        Test that kline_data in state is valid.

        Args:
            sample_state: Fixture providing IndicatorAgentState.
        """
        kline = sample_state["kline_data"]
        assert len(kline["open"]) == 30
        assert len(kline["close"]) == 30
        assert all(isinstance(p, float) for p in kline["open"])


class TestMockLLMs:
    """Tests demonstrating fixture usage for mock LLMs."""

    def test_mock_openai_llm_invoke(self, mock_openai_llm):
        """
        Test that mock OpenAI LLM can be invoked.

        Args:
            mock_openai_llm: Fixture providing mock OpenAI LLM.
        """
        response = mock_openai_llm.invoke("test prompt")
        assert response == "Mock LLM response"

    def test_mock_anthropic_llm_bind_tools(self, mock_anthropic_llm):
        """
        Test that mock Anthropic LLM supports tool binding.

        Args:
            mock_anthropic_llm: Fixture providing mock Anthropic LLM.
        """
        result = mock_anthropic_llm.bind_tools([])
        assert result is not None

    def test_mock_vision_llm_invoke(self, mock_vision_llm):
        """
        Test that mock vision LLM can be invoked.

        Args:
            mock_vision_llm: Fixture providing mock vision LLM.
        """
        response = mock_vision_llm.invoke("analyze image")
        assert response == "Mock LLM response"


class TestTemporaryDirectories:
    """Tests demonstrating fixture usage for temporary directories."""

    def test_temp_output_dir_exists(self, temp_output_dir):
        """
        Test that temporary output directory exists.

        Args:
            temp_output_dir: Fixture providing temporary directory.
        """
        assert temp_output_dir.exists()
        assert temp_output_dir.is_dir()

    def test_temp_output_dir_writable(self, temp_output_dir):
        """
        Test that temporary output directory is writable.

        Args:
            temp_output_dir: Fixture providing temporary directory.
        """
        test_file = temp_output_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.read_text() == "test content"

    def test_temp_chart_dir_isolated(self, temp_chart_dir):
        """
        Test that temporary chart directory is separate from others.

        Args:
            temp_chart_dir: Fixture providing temporary chart directory.
        """
        assert temp_chart_dir.exists()
        assert temp_chart_dir.name == "charts"


class TestYfinancePatch:
    """Tests demonstrating yfinance mocking."""

    @pytest.mark.api
    def test_patch_yfinance_returns_dataframe(self, patch_yfinance):
        """
        Test that patched yfinance returns valid dataframe.

        Args:
            patch_yfinance: Fixture mocking yfinance.download.
        """
        import yfinance
        df = yfinance.download("BTC-USD", period="30h")
        assert df is not None
        assert len(df) == 30


class TestMarkers:
    """Tests demonstrating custom pytest markers."""

    @pytest.mark.slow
    def test_marked_as_slow(self):
        """Test marked as slow - can be excluded with -m 'not slow'."""
        assert True

    @pytest.mark.integration
    def test_marked_as_integration(self):
        """Test marked as integration test."""
        assert True

    @pytest.mark.vision
    def test_marked_as_vision(self):
        """Test marked as using vision LLMs."""
        assert True
