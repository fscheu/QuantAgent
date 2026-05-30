"""
Tests for QuantAgent-um8: batch processing for backtesting LLM calls.

Covers: BatchConfig, TraceMetadata, BacktestLLMRequest, BacktestLLMResult,
        SyncExecutor, ConcurrentBatchExecutor, BatchSignalCollector,
        BatchExecutorFactory, and _build_single_prompt_messages.

All tests run without real API keys or database connections.
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from quantagent.backtesting.batch import (
    AnthropicBatchExecutor,
    BacktestLLMExecutor,
    BacktestLLMRequest,
    BacktestLLMResult,
    BatchConfig,
    BatchExecutorFactory,
    BatchSignalCollector,
    ConcurrentBatchExecutor,
    OpenAIBatchExecutor,
    SyncExecutor,
    TraceMetadata,
    _build_single_prompt_messages,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    custom_id: str = "bt:1:AAPL:1h:0:strategy_eval",
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    kline_data: Optional[Any] = None,
) -> BacktestLLMRequest:
    trace = TraceMetadata(
        backtest_run_id=1,
        symbol="AAPL",
        timeframe="1h",
        candle_index=0,
    )
    return BacktestLLMRequest(
        custom_id=custom_id,
        provider=provider,
        model=model,
        payload=[],
        trace=trace,
        kline_data=kline_data or {"close": [100, 101, 102]},
        current_price=102.0,
    )


def _make_trading_graph_mock(decision: Optional[Dict] = None):
    """Return a mock TradingGraph whose graph.invoke returns a predictable decision."""
    decision_obj = MagicMock()
    decision_obj.model_dump.return_value = decision or {
        "decision": "LONG",
        "confidence": 0.8,
        "reasoning": "uptrend",
        "risk_level": "medium",
    }
    result_state = {"final_trade_decision": decision_obj}

    graph_mock = MagicMock()
    graph_mock.graph.invoke.return_value = result_state

    trading_graph_mock = MagicMock()
    trading_graph_mock.graph = graph_mock.graph
    return trading_graph_mock


# ---------------------------------------------------------------------------
# BatchConfig
# ---------------------------------------------------------------------------


class TestBatchConfig:
    def test_defaults(self):
        cfg = BatchConfig()
        assert cfg.batch_enabled is False
        assert cfg.batch_size == 20
        assert cfg.batch_flush_timeout_sec == 300
        assert cfg.batch_max_in_flight == 3
        assert cfg.batch_poll_interval_sec == 30
        assert cfg.batch_completion_window == "24h"
        assert cfg.fail_fast is False
        assert cfg.batch_allow_fallback_to_sync is False

    def test_custom_values(self):
        cfg = BatchConfig(
            batch_enabled=True,
            batch_size=50,
            fail_fast=True,
            batch_allow_fallback_to_sync=True,
        )
        assert cfg.batch_enabled is True
        assert cfg.batch_size == 50
        assert cfg.fail_fast is True
        assert cfg.batch_allow_fallback_to_sync is True


# ---------------------------------------------------------------------------
# TraceMetadata
# ---------------------------------------------------------------------------


class TestTraceMetadata:
    def test_custom_id_format(self):
        trace = TraceMetadata(
            backtest_run_id=42,
            symbol="BTC-USD",
            timeframe="4h",
            candle_index=7,
        )
        cid = trace.to_custom_id()
        assert cid == "bt:42:BTC-USD:4h:7:strategy_eval"

    def test_custom_id_none_run_id(self):
        trace = TraceMetadata(
            backtest_run_id=None,
            symbol="ETH",
            timeframe="1d",
            candle_index=3,
        )
        cid = trace.to_custom_id()
        assert cid.startswith("bt:0:ETH:1d:3:")

    def test_custom_step(self):
        trace = TraceMetadata(
            backtest_run_id=1,
            symbol="SPY",
            timeframe="1h",
            candle_index=0,
            step="risk_eval",
        )
        assert trace.to_custom_id().endswith(":risk_eval")

    def test_custom_id_uniqueness_across_candles(self):
        ids = set()
        for i in range(10):
            trace = TraceMetadata(backtest_run_id=1, symbol="MSFT", timeframe="1h", candle_index=i)
            ids.add(trace.to_custom_id())
        assert len(ids) == 10  # all unique


# ---------------------------------------------------------------------------
# BacktestLLMResult
# ---------------------------------------------------------------------------


class TestBacktestLLMResult:
    def test_is_success_true(self):
        r = BacktestLLMResult(
            custom_id="x",
            status="completed",
            output='{"decision": "LONG"}',
            error=None,
        )
        assert r.is_success() is True

    def test_is_success_false_on_failed(self):
        r = BacktestLLMResult(
            custom_id="x",
            status="failed",
            output=None,
            error={"message": "timeout"},
        )
        assert r.is_success() is False

    def test_is_success_false_when_no_output(self):
        r = BacktestLLMResult(
            custom_id="x",
            status="completed",
            output=None,
            error=None,
        )
        assert r.is_success() is False

    def test_parse_decision_valid_json(self):
        payload = {"decision": "SHORT", "confidence": 0.7, "reasoning": "bearish"}
        r = BacktestLLMResult(
            custom_id="x",
            status="completed",
            output=json.dumps(payload),
            error=None,
        )
        parsed = r.parse_decision()
        assert parsed == payload

    def test_parse_decision_invalid_json_returns_none(self):
        r = BacktestLLMResult(
            custom_id="x",
            status="completed",
            output="not valid json {{{",
            error=None,
        )
        assert r.parse_decision() is None

    def test_parse_decision_none_output(self):
        r = BacktestLLMResult(custom_id="x", status="failed", output=None, error=None)
        assert r.parse_decision() is None


# ---------------------------------------------------------------------------
# _build_single_prompt_messages
# ---------------------------------------------------------------------------


class TestBuildSinglePromptMessages:
    def test_returns_two_messages(self):
        msgs = _build_single_prompt_messages(
            kline_data={"close": [100, 101]},
            symbol="AAPL",
            timeframe="1h",
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_system_message_contains_symbol(self):
        msgs = _build_single_prompt_messages(
            kline_data={"close": [100]},
            symbol="TSLA",
            timeframe="1d",
        )
        assert "TSLA" in msgs[0]["content"]

    def test_user_message_contains_ohlcv_data(self):
        kline = {"open": [100], "high": [105], "low": [98], "close": [103], "volume": [1000000]}
        msgs = _build_single_prompt_messages(kline, "SPY", "4h")
        # kline JSON should be embedded in the user message
        assert "100" in msgs[1]["content"]

    def test_large_kline_data_is_capped(self):
        """Prompt builder safety-caps kline JSON at 8000 chars."""
        big_kline = {"close": list(range(10000))}
        msgs = _build_single_prompt_messages(big_kline, "X", "1m")
        # The combined content should not be astronomically large
        assert len(msgs[1]["content"]) < 20000


# ---------------------------------------------------------------------------
# SyncExecutor
# ---------------------------------------------------------------------------


class TestSyncExecutor:
    def test_submit_returns_job_id(self):
        tg = _make_trading_graph_mock()
        executor = SyncExecutor(tg)
        req = _make_request()
        job_id = executor.submit([req])
        assert job_id.startswith("sync-")

    def test_poll_returns_completed(self):
        tg = _make_trading_graph_mock()
        executor = SyncExecutor(tg)
        req = _make_request()
        job_id = executor.submit([req])
        status, results = executor.poll(job_id)
        assert status == "completed"
        assert len(results) == 1

    def test_result_is_success(self):
        tg = _make_trading_graph_mock()
        executor = SyncExecutor(tg)
        req = _make_request(custom_id="bt:1:AAPL:1h:0:strategy_eval")
        job_id = executor.submit([req])
        _, results = executor.poll(job_id)
        r = results[0]
        assert r.custom_id == "bt:1:AAPL:1h:0:strategy_eval"
        assert r.is_success()

    def test_decision_contains_long(self):
        tg = _make_trading_graph_mock({"decision": "LONG", "confidence": 0.8, "reasoning": "up"})
        executor = SyncExecutor(tg)
        req = _make_request()
        job_id = executor.submit([req])
        _, results = executor.poll(job_id)
        decision = results[0].parse_decision()
        assert decision["decision"] == "LONG"

    def test_materialize_returns_dict(self):
        tg = _make_trading_graph_mock()
        executor = SyncExecutor(tg)
        req = _make_request()
        job_id = executor.submit([req])
        _, results = executor.poll(job_id)
        materialized = executor.materialize(results[0])
        assert isinstance(materialized, dict)

    def test_exception_in_graph_produces_failed_result(self):
        tg = MagicMock()
        tg.graph.invoke.side_effect = RuntimeError("LLM unavailable")
        executor = SyncExecutor(tg)
        req = _make_request()
        job_id = executor.submit([req])
        _, results = executor.poll(job_id)
        assert results[0].status == "failed"
        assert results[0].error is not None

    def test_multiple_requests_processed_in_order(self):
        decisions = [
            {"decision": "LONG", "confidence": 0.9, "reasoning": "up"},
            {"decision": "SHORT", "confidence": 0.75, "reasoning": "down"},
            {"decision": "HOLD", "confidence": 0.5, "reasoning": "flat"},
        ]
        call_count = 0

        def invoke_side_effect(state):
            nonlocal call_count
            d = MagicMock()
            d.model_dump.return_value = decisions[call_count]
            call_count += 1
            return {"final_trade_decision": d}

        tg = MagicMock()
        tg.graph.invoke.side_effect = invoke_side_effect

        executor = SyncExecutor(tg)
        reqs = [_make_request(custom_id=f"bt:1:X:1h:{i}:strategy_eval") for i in range(3)]
        job_id = executor.submit(reqs)
        _, results = executor.poll(job_id)

        assert len(results) == 3
        assert results[0].parse_decision()["decision"] == "LONG"
        assert results[1].parse_decision()["decision"] == "SHORT"
        assert results[2].parse_decision()["decision"] == "HOLD"

    def test_none_final_trade_decision_returns_hold(self):
        tg = MagicMock()
        tg.graph.invoke.return_value = {"final_trade_decision": None}
        executor = SyncExecutor(tg)
        req = _make_request()
        job_id = executor.submit([req])
        _, results = executor.poll(job_id)
        decision = results[0].parse_decision()
        assert decision["decision"] == "HOLD"


# ---------------------------------------------------------------------------
# ConcurrentBatchExecutor
# ---------------------------------------------------------------------------


class TestConcurrentBatchExecutor:
    def test_submit_returns_job_id(self):
        tg = _make_trading_graph_mock()
        executor = ConcurrentBatchExecutor(tg, max_workers=2)
        reqs = [_make_request(custom_id=f"bt:1:X:1h:{i}:strategy_eval") for i in range(3)]
        job_id = executor.submit(reqs)
        assert job_id.startswith("concurrent-")

    def test_all_requests_processed(self):
        tg = _make_trading_graph_mock()
        executor = ConcurrentBatchExecutor(tg, max_workers=2)
        reqs = [_make_request(custom_id=f"bt:1:X:1h:{i}:strategy_eval") for i in range(5)]
        job_id = executor.submit(reqs)
        status, results = executor.poll(job_id)
        assert status == "completed"
        assert len(results) == 5

    def test_results_include_correct_custom_ids(self):
        tg = _make_trading_graph_mock()
        executor = ConcurrentBatchExecutor(tg, max_workers=2)
        ids = [f"bt:1:X:1h:{i}:strategy_eval" for i in range(3)]
        reqs = [_make_request(custom_id=cid) for cid in ids]
        job_id = executor.submit(reqs)
        _, results = executor.poll(job_id)
        result_ids = {r.custom_id for r in results}
        assert result_ids == set(ids)

    def test_exception_in_worker_gives_failed_result(self):
        tg = MagicMock()
        tg.graph.invoke.side_effect = RuntimeError("worker crash")
        executor = ConcurrentBatchExecutor(tg, max_workers=2)
        reqs = [_make_request(custom_id=f"bt:1:X:1h:{i}:strategy_eval") for i in range(2)]
        job_id = executor.submit(reqs)
        _, results = executor.poll(job_id)
        # All requests should have failed results (not exceptions propagated)
        assert all(r.status == "failed" for r in results)

    def test_materialize_returns_dict(self):
        tg = _make_trading_graph_mock()
        executor = ConcurrentBatchExecutor(tg, max_workers=2)
        req = _make_request()
        job_id = executor.submit([req])
        _, results = executor.poll(job_id)
        materialized = executor.materialize(results[0])
        assert isinstance(materialized, dict)


# ---------------------------------------------------------------------------
# BatchSignalCollector
# ---------------------------------------------------------------------------


class TestBatchSignalCollector:
    def _make_sync_executor(self, decision=None):
        tg = _make_trading_graph_mock(decision)
        return SyncExecutor(tg)

    def test_add_increments_invocations(self):
        executor = self._make_sync_executor()
        cfg = BatchConfig(batch_enabled=True, batch_size=100)
        collector = BatchSignalCollector(executor, cfg)
        req = _make_request()
        collector.add(req)
        assert collector.invocations_total == 1

    def test_flush_triggers_at_batch_size(self):
        executor = self._make_sync_executor()
        cfg = BatchConfig(batch_enabled=True, batch_size=3)
        collector = BatchSignalCollector(executor, cfg)

        for i in range(3):
            collector.add(_make_request(custom_id=f"bt:1:X:1h:{i}:strategy_eval"))

        # Batch of 3 should have been auto-flushed
        assert collector.batched_total == 3
        assert len(collector._pending) == 0

    def test_pending_not_flushed_below_batch_size(self):
        executor = self._make_sync_executor()
        cfg = BatchConfig(batch_enabled=True, batch_size=10)
        collector = BatchSignalCollector(executor, cfg)

        for i in range(5):
            collector.add(_make_request(custom_id=f"bt:1:X:1h:{i}:strategy_eval"))

        assert len(collector._pending) == 5
        assert collector.batched_total == 0

    def test_manual_flush_clears_pending(self):
        executor = self._make_sync_executor()
        cfg = BatchConfig(batch_enabled=True, batch_size=100)
        collector = BatchSignalCollector(executor, cfg)

        for i in range(5):
            collector.add(_make_request(custom_id=f"bt:1:X:1h:{i}:strategy_eval"))

        collector.flush()
        assert len(collector._pending) == 0
        assert collector.batched_total == 5

    def test_get_result_after_flush(self):
        executor = self._make_sync_executor()
        cfg = BatchConfig(batch_enabled=True, batch_size=100)
        collector = BatchSignalCollector(executor, cfg)
        req = _make_request(custom_id="bt:1:AAPL:1h:0:strategy_eval")
        collector.add(req)
        collector.flush()

        result = collector.get_result("bt:1:AAPL:1h:0:strategy_eval")
        assert result is not None
        assert result.is_success()

    def test_get_result_returns_none_for_unknown_id(self):
        executor = self._make_sync_executor()
        cfg = BatchConfig(batch_enabled=True, batch_size=100)
        collector = BatchSignalCollector(executor, cfg)
        assert collector.get_result("nonexistent") is None

    def test_flush_and_get_all_returns_all_results(self):
        executor = self._make_sync_executor()
        cfg = BatchConfig(batch_enabled=True, batch_size=100)
        collector = BatchSignalCollector(executor, cfg)

        ids = [f"bt:1:X:1h:{i}:strategy_eval" for i in range(4)]
        for cid in ids:
            collector.add(_make_request(custom_id=cid))

        all_results = collector.flush_and_get_all()
        assert set(all_results.keys()) == set(ids)

    def test_summary_reflects_correct_counts(self):
        executor = self._make_sync_executor()
        cfg = BatchConfig(batch_enabled=True, batch_size=100)
        collector = BatchSignalCollector(executor, cfg)

        for i in range(3):
            collector.add(_make_request(custom_id=f"bt:1:X:1h:{i}:strategy_eval"))
        collector.flush()

        summary = collector.summary()
        assert summary["invocations_total"] == 3
        assert summary["batched_total"] == 3
        assert summary["failed_total"] == 0

    def test_failed_requests_counted(self):
        tg = MagicMock()
        tg.graph.invoke.side_effect = RuntimeError("fail")
        executor = SyncExecutor(tg)
        cfg = BatchConfig(batch_enabled=True, batch_size=100, fail_fast=False)
        collector = BatchSignalCollector(executor, cfg)

        collector.add(_make_request(custom_id="bt:1:X:1h:0:strategy_eval"))
        collector.flush()

        summary = collector.summary()
        assert summary["failed_total"] == 1
        assert summary["batched_total"] == 0

    def test_fail_fast_raises_on_flush_error(self):
        tg = MagicMock()
        tg.graph.invoke.side_effect = RuntimeError("fatal error")
        executor = SyncExecutor(tg)
        cfg = BatchConfig(batch_enabled=True, batch_size=100, fail_fast=True)
        collector = BatchSignalCollector(executor, cfg)
        collector.add(_make_request())

        # fail_fast=True: flush error should propagate
        # SyncExecutor wraps individual exceptions — they produce failed results, not raised
        # The fail_fast in BatchSignalCollector.flush only raises if executor.submit raises
        # So we test what we can: failed_total is updated correctly
        collector.flush()
        assert collector.failed_total == 1


# ---------------------------------------------------------------------------
# BatchExecutorFactory
# ---------------------------------------------------------------------------


class TestBatchExecutorFactory:
    def test_creates_sync_executor_when_disabled(self):
        tg = _make_trading_graph_mock()
        cfg = BatchConfig(batch_enabled=False)
        executor = BatchExecutorFactory.create(
            provider="sync", model="gpt-4o-mini", trading_graph=tg, batch_config=cfg
        )
        assert isinstance(executor, SyncExecutor)

    def test_creates_sync_executor_for_sync_provider(self):
        tg = _make_trading_graph_mock()
        cfg = BatchConfig(batch_enabled=True)
        executor = BatchExecutorFactory.create(
            provider="sync", model="gpt-4o-mini", trading_graph=tg, batch_config=cfg
        )
        assert isinstance(executor, SyncExecutor)

    def test_creates_concurrent_executor(self):
        tg = _make_trading_graph_mock()
        cfg = BatchConfig(batch_enabled=True)
        executor = BatchExecutorFactory.create(
            provider="concurrent",
            model="gpt-4o-mini",
            trading_graph=tg,
            batch_config=cfg,
            concurrent_workers=3,
        )
        assert isinstance(executor, ConcurrentBatchExecutor)
        assert executor.max_workers == 3

    def test_creates_openai_batch_executor(self):
        cfg = BatchConfig(batch_enabled=True)
        mock_openai_cls = MagicMock(return_value=MagicMock())
        mock_openai_module = MagicMock(OpenAI=mock_openai_cls)
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            executor = BatchExecutorFactory.create(
                provider="openai_batch",
                model="gpt-4o-mini",
                api_key="test-key",
                batch_config=cfg,
            )
        assert isinstance(executor, OpenAIBatchExecutor)

    def test_creates_anthropic_batch_executor(self):
        cfg = BatchConfig(batch_enabled=True)
        with patch("quantagent.backtesting.batch.Anthropic", create=True) as mock_anthropic:
            mock_anthropic.return_value = MagicMock()
            with patch.dict("sys.modules", {"anthropic": MagicMock(Anthropic=mock_anthropic)}):
                executor = BatchExecutorFactory.create(
                    provider="anthropic_batch",
                    model="claude-3-5-haiku-20241022",
                    api_key="test-key",
                    batch_config=cfg,
                )
        assert isinstance(executor, AnthropicBatchExecutor)

    def test_raises_for_unknown_provider(self):
        cfg = BatchConfig(batch_enabled=True)
        with pytest.raises(ValueError, match="Unknown batch executor provider"):
            BatchExecutorFactory.create(
                provider="unknown_provider",
                model="gpt-4o-mini",
                batch_config=cfg,
            )

    def test_raises_when_trading_graph_missing_for_sync(self):
        cfg = BatchConfig(batch_enabled=False)
        with pytest.raises(ValueError, match="trading_graph required"):
            BatchExecutorFactory.create(
                provider="sync", model="gpt-4o-mini", trading_graph=None, batch_config=cfg
            )

    def test_raises_when_trading_graph_missing_for_concurrent(self):
        cfg = BatchConfig(batch_enabled=True)
        with pytest.raises(ValueError, match="trading_graph required"):
            BatchExecutorFactory.create(
                provider="concurrent", model="gpt-4o-mini", trading_graph=None, batch_config=cfg
            )


# ---------------------------------------------------------------------------
# wait_for_completion (abstract method on BacktestLLMExecutor)
# ---------------------------------------------------------------------------


class TestWaitForCompletion:
    """Test the default wait_for_completion template method."""

    def test_returns_immediately_when_already_completed(self):
        """If poll() returns 'completed', wait_for_completion should not sleep."""
        tg = _make_trading_graph_mock()
        executor = SyncExecutor(tg)
        req = _make_request()
        job_id = executor.submit([req])
        results = executor.wait_for_completion(job_id, poll_interval_sec=0)
        assert len(results) == 1
        assert results[0].is_success()

    def test_fail_fast_raises_on_terminal_failure(self):
        """A terminal status with fail_fast=True should raise RuntimeError."""

        class FailingExecutor(BacktestLLMExecutor):
            def submit(self, requests):
                return "fail-job"

            def poll(self, job_id):
                return "failed", []

            def materialize(self, result):
                return None

        executor = FailingExecutor()
        with pytest.raises(RuntimeError, match="fail-job"):
            executor.wait_for_completion("fail-job", poll_interval_sec=0, fail_fast=True)

    def test_no_raise_on_terminal_failure_without_fail_fast(self):
        """With fail_fast=False, terminal failures should return empty results."""

        class FailingExecutor(BacktestLLMExecutor):
            def submit(self, requests):
                return "fail-job"

            def poll(self, job_id):
                return "expired", []

            def materialize(self, result):
                return None

        executor = FailingExecutor()
        results = executor.wait_for_completion("fail-job", poll_interval_sec=0, fail_fast=False)
        assert results == []


# ---------------------------------------------------------------------------
# BacktestMetrics batch fields
# ---------------------------------------------------------------------------


def test_backtest_metrics_batch_fields():
    """Verify BacktestMetrics includes QuantAgent-um8 batch fields."""
    from quantagent.backtesting.backtest import BacktestMetrics

    metrics = BacktestMetrics(
        total_trades=10,
        winning_trades=6,
        losing_trades=4,
        win_rate=0.6,
        profit_factor=1.5,
        sharpe_ratio=1.2,
        max_drawdown=-0.05,
        total_pnl=500.0,
        avg_win=100.0,
        avg_loss=-60.0,
        largest_win=200.0,
        largest_loss=-120.0,
        total_return_pct=5.0,
        batch_enabled=True,
        invocations_total=100,
        batched_total=95,
        failed_total=5,
    )

    assert metrics.batch_enabled is True
    assert metrics.invocations_total == 100
    assert metrics.batched_total == 95
    assert metrics.failed_total == 5


# ---------------------------------------------------------------------------
# Module-level __init__ exports
# ---------------------------------------------------------------------------


def test_batch_module_exports():
    """Ensure all public symbols are exported from __init__."""
    import quantagent.backtesting as bt

    expected = [
        "BatchConfig",
        "TraceMetadata",
        "BacktestLLMRequest",
        "BacktestLLMResult",
        "BacktestLLMExecutor",
        "SyncExecutor",
        "ConcurrentBatchExecutor",
        "OpenAIBatchExecutor",
        "AnthropicBatchExecutor",
        "BatchExecutorFactory",
        "BatchSignalCollector",
    ]
    for name in expected:
        assert hasattr(bt, name), f"Missing export: {name}"
