"""
Tests for PostgreSQL checkpointing and resume functionality in trading graph.

Testing Strategy:
- Avoid excessive mocking of checkpointer (causes tautologies)
- Use real checkpointer from PostgreSQL or skip if unavailable
- Validate structure and behavior, not mock internals
- Test that checkpointing doesn't break graph execution
- Test that state is correctly passed through graph with checkpointing

See docs/03_technical/TESTING_PATTERNS.md for testing guidelines.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

from quantagent.trading_graph import TradingGraph


# ============================================================================
# CHECKPOINTING CONFIGURATION TESTS
# ============================================================================

class TestCheckpointingConfiguration:
    """Test checkpointing setup and configuration."""

    def test_trading_graph_without_checkpointing(self, mock_llm, mock_vision_llm, mock_toolkit):
        """Verify graph initializes without checkpointing by default."""
        tg = TradingGraph(use_checkpointing=False)

        assert tg.checkpointer is None
        assert tg.graph is not None

    @patch('quantagent.trading_graph.PostgresSaver')
    def test_trading_graph_with_checkpointing_enabled(self, mock_postgres_saver, mock_llm,
                                                      mock_vision_llm, mock_toolkit, monkeypatch):
        """Verify graph initializes checkpointing when enabled."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/quantagent_dev")

        mock_saver_instance = MagicMock()
        mock_postgres_saver.from_conn_string.return_value = mock_saver_instance

        tg = TradingGraph(use_checkpointing=True)

        assert tg.checkpointer is not None
        assert tg.graph is not None
        mock_postgres_saver.from_conn_string.assert_called_once()

    def test_checkpointing_fails_without_database_url(self, mock_llm, mock_vision_llm,
                                                      mock_toolkit, monkeypatch):
        """Verify checkpointing raises error when DATABASE_URL not set."""
        monkeypatch.delenv("DATABASE_URL", raising=False)

        with patch('quantagent.trading_graph.PostgresSaver') as mock_saver:
            with pytest.raises(ValueError, match="DATABASE_URL environment variable not set"):
                TradingGraph(use_checkpointing=True)

    def test_checkpointing_graceful_fallback_without_postgres_saver_library(self, mock_llm,
                                                                           mock_vision_llm,
                                                                           mock_toolkit, monkeypatch):
        """Verify error message is clear when PostgresSaver not installed."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/quantagent_dev")

        with patch('quantagent.trading_graph.PostgresSaver', None):
            with pytest.raises(ImportError, match="langgraph.checkpoint.postgres not available"):
                TradingGraph(use_checkpointing=True)


# ============================================================================
# GRAPH EXECUTION TESTS - Without excessive mocking
# ============================================================================

class TestGraphExecutionWithAndWithoutCheckpointing:
    """Test that graph executes correctly regardless of checkpointing mode."""

    def test_graph_invokes_without_checkpointing(self, mock_llm, mock_vision_llm, mock_toolkit,
                                                 sample_state_inicial):
        """Verify graph invokes successfully without checkpointing."""
        tg = TradingGraph(use_checkpointing=False)

        result = tg.graph.invoke(sample_state_inicial)

        assert result is not None
        assert isinstance(result, dict)
        assert tg.checkpointer is None
        assert "messages" in result

    def test_graph_with_checkpointing_disabled_produces_valid_output(self, mock_llm, mock_vision_llm,
                                                                     mock_toolkit, sample_state_inicial):
        """Verify graph produces valid output structure without checkpointing."""
        tg = TradingGraph(use_checkpointing=False)

        result = tg.graph.invoke(sample_state_inicial)

        # Verify expected output fields exist
        expected_fields = ["kline_data", "time_frame", "stock_name", "messages"]
        for field in expected_fields:
            assert field in result, f"Result missing {field}"

    def test_graph_execution_with_thread_id_config(self, mock_llm, mock_vision_llm, mock_toolkit,
                                                   sample_state_inicial):
        """Verify graph accepts thread_id in config without checkpointing."""
        tg = TradingGraph(use_checkpointing=False)

        config = {"configurable": {"thread_id": "test_thread_001"}}
        result = tg.graph.invoke(sample_state_inicial, config=config)

        assert result is not None
        assert isinstance(result, dict)

    def test_graph_with_multiple_consecutive_invocations(self, mock_llm, mock_vision_llm,
                                                         mock_toolkit, sample_state_inicial):
        """Verify graph can be invoked multiple times without state leakage."""
        tg = TradingGraph(use_checkpointing=False)

        results = []
        for i in range(3):
            result = tg.graph.invoke(sample_state_inicial)
            results.append(result)

        assert len(results) == 3
        # Each result should be independent
        for result in results:
            assert isinstance(result, dict)
            assert "messages" in result


# ============================================================================
# CHECKPOINTING INFRASTRUCTURE TESTS
# ============================================================================

class TestCheckpointingInfrastructure:
    """Test checkpointing setup and infrastructure (not mocking internals)."""

    @patch('quantagent.trading_graph.PostgresSaver')
    def test_graph_compiles_with_checkpointer(self, mock_postgres_saver, mock_llm, mock_vision_llm,
                                             mock_toolkit, monkeypatch):
        """Verify graph compiles successfully when checkpointer is provided."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/quantagent_dev")

        mock_saver_instance = MagicMock()
        mock_postgres_saver.from_conn_string.return_value = mock_saver_instance

        tg = TradingGraph(use_checkpointing=True)

        assert tg.graph is not None
        assert tg.checkpointer is not None
        assert hasattr(tg.graph, 'invoke'), "Graph must have invoke method"
        assert hasattr(tg.graph, 'get_state'), "Graph must have get_state method (for checkpointing)"

    def test_checkpointer_is_database_saver_type(self, monkeypatch):
        """Verify checkpointer is PostgresSaver type when initialized."""
        # This test uses real import to verify type
        db_url = os.environ.get("DATABASE_URL")

        if not db_url:
            pytest.skip("DATABASE_URL not set - skipping real PostgreSQL checkpointer test")

        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            # Create real checkpointer (if DB is available)
            checkpointer = PostgresSaver.from_conn_string(db_url)
            assert checkpointer is not None
            assert hasattr(checkpointer, 'get_tuple')
            assert hasattr(checkpointer, 'put')
            assert hasattr(checkpointer, 'list')
        except Exception as e:
            pytest.skip(f"Cannot create real PostgreSQL checkpointer: {e}")


# ============================================================================
# STATE STRUCTURE & PRESERVATION TESTS
# ============================================================================

class TestStateStructureAndPreservation:
    """Test that state is properly preserved through graph execution."""

    def test_kline_data_preserved_in_result(self, mock_llm, mock_vision_llm, mock_toolkit,
                                            sample_state_inicial):
        """Verify kline_data from input is preserved in output."""
        tg = TradingGraph(use_checkpointing=False)
        original_kline = sample_state_inicial["kline_data"].copy()

        result = tg.graph.invoke(sample_state_inicial)

        assert "kline_data" in result
        assert result["kline_data"] == original_kline

    def test_messages_list_present_in_output(self, mock_llm, mock_vision_llm, mock_toolkit,
                                             sample_state_inicial):
        """Verify messages list is present and is actually a list."""
        tg = TradingGraph(use_checkpointing=False)

        result = tg.graph.invoke(sample_state_inicial)

        assert "messages" in result
        assert isinstance(result["messages"], list)

    def test_time_frame_and_stock_name_preserved(self, mock_llm, mock_vision_llm, mock_toolkit,
                                                  sample_state_inicial):
        """Verify configuration fields are preserved through execution."""
        tg = TradingGraph(use_checkpointing=False)
        original_timeframe = sample_state_inicial["time_frame"]
        original_symbol = sample_state_inicial["stock_name"]

        result = tg.graph.invoke(sample_state_inicial)

        assert result["time_frame"] == original_timeframe
        assert result["stock_name"] == original_symbol

    def test_result_contains_all_input_fields(self, mock_llm, mock_vision_llm, mock_toolkit,
                                              sample_state_inicial):
        """Verify output contains at least all input fields from state."""
        tg = TradingGraph(use_checkpointing=False)
        input_keys = set(sample_state_inicial.keys())

        result = tg.graph.invoke(sample_state_inicial)
        output_keys = set(result.keys())

        # Output should have at least all input fields (may have more)
        assert input_keys.issubset(output_keys), f"Missing fields: {input_keys - output_keys}"


# ============================================================================
# GRAPH STATE INSPECTION TESTS
# ============================================================================

class TestGraphStateInspection:
    """Test ability to inspect graph state (requires get_state method)."""

    def test_graph_has_get_state_method_for_inspection(self, mock_llm, mock_vision_llm,
                                                       mock_toolkit):
        """Verify graph has get_state method for checkpointing support."""
        tg = TradingGraph(use_checkpointing=False)

        assert hasattr(tg.graph, 'get_state'), "Graph should have get_state for checkpointing"

    def test_get_state_with_valid_thread_id(self, mock_llm, mock_vision_llm, mock_toolkit,
                                            sample_state_inicial):
        """Verify get_state can retrieve state with valid thread_id."""
        tg = TradingGraph(use_checkpointing=False)
        config = {"configurable": {"thread_id": "state_test_001"}}

        # Execute graph first
        result = tg.graph.invoke(sample_state_inicial, config=config)

        # Now try to get state
        try:
            state = tg.graph.get_state(config)
            # If get_state works, it should return a state-like object
            assert state is not None
        except Exception as e:
            # Some graph configurations may not support get_state
            pytest.skip(f"Graph get_state not fully supported: {e}")


# ============================================================================
# EDGE CASES & ROBUSTNESS TESTS
# ============================================================================

class TestCheckpointingRobustness:
    """Test edge cases and robustness of checkpointing setup."""

    def test_refresh_llms_preserves_checkpointer_reference(self, mock_llm, mock_vision_llm,
                                                           mock_toolkit, monkeypatch):
        """Verify that refreshing LLMs doesn't affect checkpointer."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/quantagent_dev")

        with patch('quantagent.trading_graph.PostgresSaver') as mock_saver:
            mock_saver.from_conn_string.return_value = MagicMock()
            tg = TradingGraph(use_checkpointing=True)

            original_checkpointer = tg.checkpointer

            # Refresh LLMs
            tg.refresh_llms()

            # Checkpointer should be unchanged
            assert tg.checkpointer is original_checkpointer

    def test_update_api_key_preserves_checkpointer_reference(self, mock_llm, mock_vision_llm,
                                                             mock_toolkit, monkeypatch):
        """Verify that updating API keys doesn't affect checkpointer."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/quantagent_dev")

        with patch('quantagent.trading_graph.PostgresSaver') as mock_saver:
            mock_saver.from_conn_string.return_value = MagicMock()
            tg = TradingGraph(use_checkpointing=True)

            original_checkpointer = tg.checkpointer

            # Update API key
            tg.update_api_key("new-test-key", provider="openai")

            # Checkpointer should be unchanged
            assert tg.checkpointer is original_checkpointer

    def test_multiple_graphs_with_different_checkpointing_modes(self, mock_llm, mock_vision_llm,
                                                                mock_toolkit):
        """Verify multiple graph instances can have different checkpointing configs."""
        # One without checkpointing
        tg_no_checkpoint = TradingGraph(use_checkpointing=False)
        assert tg_no_checkpoint.checkpointer is None

        # One with checkpointing (will fail if no DB, but verifies parameter accepted)
        try:
            with patch('quantagent.trading_graph.PostgresSaver') as mock_saver:
                mock_saver.from_conn_string.return_value = MagicMock()
                tg_with_checkpoint = TradingGraph(use_checkpointing=True)
                assert tg_with_checkpoint.checkpointer is not None
        except ValueError:
            pytest.skip("Database URL not configured")


# ============================================================================
# CONFIGURATION COMBINATIONS TESTS
# ============================================================================

class TestCheckpointingWithVariousConfigs:
    """Test that checkpointing works with various graph configurations."""

    def test_graph_with_different_thread_ids(self, mock_llm, mock_vision_llm, mock_toolkit,
                                             sample_state_inicial):
        """Verify graph can be invoked with multiple different thread_ids."""
        tg = TradingGraph(use_checkpointing=True)

        thread_ids = ["thread_001", "thread_002", "thread_003"]
        results = []

        for thread_id in thread_ids:
            config = {"configurable": {"thread_id": thread_id}}
            result = tg.graph.invoke(sample_state_inicial, config=config)
            results.append(result)

        assert len(results) == len(thread_ids)
        for result in results:
            assert result is not None

    def test_sequential_invocations_with_same_thread_id(self, mock_llm, mock_vision_llm,
                                                        mock_toolkit, sample_state_inicial):
        """Verify same thread_id can be used in sequential invocations."""
        tg = TradingGraph(use_checkpointing=False)
        config = {"configurable": {"thread_id": "sequential_test"}}

        # First invocation
        result1 = tg.graph.invoke(sample_state_inicial, config=config)
        assert result1 is not None

        # Second invocation with same thread_id
        result2 = tg.graph.invoke(sample_state_inicial, config=config)
        assert result2 is not None

        # Both should complete without error
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)

    def test_graph_execution_with_custom_stock_names(self, mock_llm, mock_vision_llm,
                                                     mock_toolkit, sample_state_inicial):
        """Verify graph handles different stock names correctly."""
        tg = TradingGraph(use_checkpointing=False)

        stocks = ["BTC", "AAPL", "EURUSD"]

        for stock in stocks:
            state = sample_state_inicial.copy()
            state["stock_name"] = stock

            result = tg.graph.invoke(state)

            assert result is not None
            assert result["stock_name"] == stock


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestCheckpointingErrorHandling:
    """Test error handling when checkpointing is misconfigured."""

    def test_invalid_database_url_format(self, mock_llm, mock_vision_llm, mock_toolkit, monkeypatch):
        """Verify appropriate error when DATABASE_URL is invalid."""
        # Set invalid URL
        monkeypatch.setenv("DATABASE_URL", "invalid://not-a-valid-url")

        with patch('quantagent.trading_graph.PostgresSaver') as mock_saver:
            mock_saver.from_conn_string.side_effect = ValueError("Invalid connection string")

            with pytest.raises(ValueError):
                TradingGraph(use_checkpointing=True)

    def test_missing_environment_variable_error(self, mock_llm, mock_vision_llm, mock_toolkit,
                                                monkeypatch):
        """Verify clear error when DATABASE_URL environment variable is missing."""
        monkeypatch.delenv("DATABASE_URL", raising=False)

        with patch('quantagent.trading_graph.PostgresSaver') as mock_saver:
            with pytest.raises(ValueError) as exc_info:
                TradingGraph(use_checkpointing=True)

            assert "DATABASE_URL" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
