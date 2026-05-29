"""
Batch processing for backtesting LLM calls.

Provides a pluggable executor interface for:
- SyncExecutor: current sequential behavior (baseline)
- ConcurrentBatchExecutor: ThreadPoolExecutor parallelism (no extra cost)
- OpenAIBatchExecutor: OpenAI Batch API (~50% discount, async)
- AnthropicBatchExecutor: Anthropic Message Batches API (~50% discount, async)

Usage (from Backtest):
    config = BatchConfig(batch_enabled=True, batch_size=50)
    # Executor is created by BatchExecutorFactory based on provider + config
"""

import io
import json
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BatchConfig:
    """Configuration for batch processing mode."""

    batch_enabled: bool = False
    # Number of requests to accumulate before submitting a batch
    batch_size: int = 20
    # Max seconds to wait before flushing a partial batch
    batch_flush_timeout_sec: int = 300
    # Max concurrent in-flight batches (for provider batch modes)
    batch_max_in_flight: int = 3
    # Polling interval for provider batch status checks (seconds)
    batch_poll_interval_sec: int = 30
    # Completion window for OpenAI batch API
    batch_completion_window: str = "24h"
    # If True, abort entire backtest on any batch failure; False = continue
    fail_fast: bool = False
    # If True, fall back to sync when provider batch is unavailable
    batch_allow_fallback_to_sync: bool = False


# ---------------------------------------------------------------------------
# Trace / request / result data structures
# ---------------------------------------------------------------------------


@dataclass
class TraceMetadata:
    """Traceability metadata per batch request."""

    backtest_run_id: Optional[int]
    symbol: str
    timeframe: str
    candle_index: int
    step: str = "strategy_eval"

    def to_custom_id(self) -> str:
        """Stable custom_id for provider correlation."""
        run_id = self.backtest_run_id or 0
        return f"bt:{run_id}:{self.symbol}:{self.timeframe}:{self.candle_index}:{self.step}"


@dataclass
class BacktestLLMRequest:
    """Atomic unit of work for a single LLM strategy evaluation."""

    custom_id: str
    provider: str  # openai | anthropic
    model: str
    # messages in provider-native format (list of dicts with role/content)
    payload: List[Dict[str, Any]]
    trace: TraceMetadata
    # Original OHLCV kline_data for SyncExecutor / ConcurrentBatchExecutor
    kline_data: Optional[Any] = None
    current_price: float = 0.0


@dataclass
class BacktestLLMResult:
    """Result of a single LLM batch request."""

    custom_id: str
    # completed | failed | cancelled | expired
    status: str
    # Raw JSON string of TradingDecision-compatible dict; None on failure
    output: Optional[str]
    error: Optional[Dict[str, Any]]
    submitted_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def is_success(self) -> bool:
        return self.status == "completed" and self.output is not None

    def parse_decision(self) -> Optional[Dict[str, Any]]:
        """Parse output JSON into a decision dict."""
        if not self.output:
            return None
        try:
            return json.loads(self.output)
        except (json.JSONDecodeError, TypeError):
            return None


# ---------------------------------------------------------------------------
# Abstract executor interface
# ---------------------------------------------------------------------------


class BacktestLLMExecutor(ABC):
    """
    Abstract executor for backtesting LLM requests.

    Subclasses implement submit/poll/materialize for different providers.
    """

    @abstractmethod
    def submit(self, requests: List[BacktestLLMRequest]) -> str:
        """Submit a batch of requests and return a job handle (batch_id or local token)."""

    @abstractmethod
    def poll(self, job_id: str) -> Tuple[str, List[BacktestLLMResult]]:
        """
        Check batch status.

        Returns:
            (status, results) where status is one of:
            validating | in_progress | finalizing | completed | failed | expired | cancelled
            results is non-empty only when status == "completed"
        """

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval_sec: int = 30,
        fail_fast: bool = False,
    ) -> List[BacktestLLMResult]:
        """Block until batch is done, polling every poll_interval_sec."""
        while True:
            status, results = self.poll(job_id)
            logger.info(
                "Batch status check",
                extra={
                    "event_type": "batch_poll",
                    "batch_id": job_id,
                    "status": status,
                },
            )
            if status == "completed":
                return results
            if status in ("failed", "expired", "cancelled"):
                if fail_fast:
                    raise RuntimeError(f"Batch {job_id} ended with status={status}")
                logger.warning(f"Batch {job_id} ended with status={status}")
                return results
            time.sleep(poll_interval_sec)

    @abstractmethod
    def materialize(self, result: BacktestLLMResult) -> Optional[Dict[str, Any]]:
        """Parse a BacktestLLMResult into a TradingDecision-compatible dict."""


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _build_single_prompt_messages(
    kline_data: Any,
    symbol: str,
    timeframe: str,
) -> List[Dict[str, str]]:
    """
    Build a single comprehensive prompt for provider batch mode.

    Returns messages in OpenAI-compatible format (also used by Anthropic after
    stripping the system message into the separate `system` parameter).
    """
    kline_json = json.dumps(kline_data, default=str)[:8000]  # safety cap

    system_content = (
        f"You are a quantitative trading analyst specializing in {timeframe} charts for {symbol}. "
        "Analyze the provided OHLCV data and output a single JSON trading decision. "
        "You must respond with ONLY valid JSON — no markdown, no explanation."
    )

    human_content = f"""Analyze the following OHLCV data for {symbol} ({timeframe} timeframe) and provide a trading decision.

OHLCV Data:
{kline_json}

Output ONLY this JSON structure:
{{
  "decision": "<LONG|SHORT|HOLD>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<brief explanation>",
  "risk_level": "<low|medium|high>",
  "entry_price": <float or null>,
  "stop_loss": <float or null>,
  "take_profit": <float or null>
}}

Rules:
- LONG if price likely to rise next {timeframe} candle
- SHORT if price likely to fall next {timeframe} candle
- HOLD if signals are unclear or mixed
- confidence >= 0.6 to recommend LONG/SHORT; otherwise HOLD"""

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": human_content},
    ]


# ---------------------------------------------------------------------------
# SyncExecutor: wraps existing graph.invoke() behavior
# ---------------------------------------------------------------------------


class SyncExecutor(BacktestLLMExecutor):
    """
    Synchronous executor — wraps existing TradingGraph behavior.

    Does not actually batch; processes each request sequentially.
    Used as baseline and fallback.
    """

    def __init__(self, trading_graph):
        self.trading_graph = trading_graph
        self._results_cache: Dict[str, BacktestLLMResult] = {}

    def submit(self, requests: List[BacktestLLMRequest]) -> str:
        """Process all requests synchronously (one graph.invoke per request)."""
        job_id = f"sync-{int(time.time()*1000)}"
        results = []
        for req in requests:
            result = self._invoke_one(req)
            results.append(result)
            self._results_cache[job_id] = results  # store last batch only
        self._results_cache[job_id] = results
        return job_id

    def _invoke_one(self, req: BacktestLLMRequest) -> BacktestLLMResult:
        submitted_at = datetime.utcnow()
        try:
            initial_state: Dict = {
                "kline_data": req.kline_data,
                "time_frame": req.trace.timeframe,
                "stock_name": req.trace.symbol,
                "messages": [],
            }
            graph_result = self.trading_graph.graph.invoke(initial_state)
            decision_raw = graph_result.get("final_trade_decision")

            if decision_raw is None:
                output_dict = {"decision": "HOLD", "confidence": 0.0, "reasoning": "No result"}
            elif hasattr(decision_raw, "model_dump"):
                output_dict = decision_raw.model_dump()
            else:
                output_dict = {
                    "decision": str(getattr(decision_raw, "decision", "HOLD")),
                    "confidence": float(getattr(decision_raw, "confidence", 0.0)),
                    "reasoning": str(getattr(decision_raw, "reasoning", "")),
                    "risk_level": str(getattr(decision_raw, "risk_level", "high")),
                }

            return BacktestLLMResult(
                custom_id=req.custom_id,
                status="completed",
                output=json.dumps(output_dict),
                error=None,
                submitted_at=submitted_at,
                completed_at=datetime.utcnow(),
            )
        except Exception as exc:
            logger.error(
                f"SyncExecutor failed for {req.custom_id}: {exc}",
                exc_info=True,
                extra={"event_type": "batch_request_error", "custom_id": req.custom_id},
            )
            return BacktestLLMResult(
                custom_id=req.custom_id,
                status="failed",
                output=None,
                error={"message": str(exc), "retryable": True},
                submitted_at=submitted_at,
                completed_at=datetime.utcnow(),
            )

    def poll(self, job_id: str) -> Tuple[str, List[BacktestLLMResult]]:
        results = self._results_cache.get(job_id, [])
        return "completed", results

    def materialize(self, result: BacktestLLMResult) -> Optional[Dict[str, Any]]:
        return result.parse_decision()


# ---------------------------------------------------------------------------
# ConcurrentBatchExecutor: ThreadPoolExecutor parallelism
# ---------------------------------------------------------------------------


class ConcurrentBatchExecutor(BacktestLLMExecutor):
    """
    Concurrent executor using ThreadPoolExecutor.

    Parallelizes graph.invoke() calls across requests in a batch.
    Uses standard sync API pricing (no discount), but much faster throughput.
    """

    def __init__(self, trading_graph, max_workers: int = 4):
        self.trading_graph = trading_graph
        self.max_workers = max_workers
        self._sync = SyncExecutor(trading_graph)
        self._results_cache: Dict[str, List[BacktestLLMResult]] = {}

    def submit(self, requests: List[BacktestLLMRequest]) -> str:
        job_id = f"concurrent-{int(time.time()*1000)}"
        results: List[BacktestLLMResult] = [None] * len(requests)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_idx = {
                pool.submit(self._sync._invoke_one, req): i
                for i, req in enumerate(requests)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    req = requests[idx]
                    results[idx] = BacktestLLMResult(
                        custom_id=req.custom_id,
                        status="failed",
                        output=None,
                        error={"message": str(exc)},
                        submitted_at=datetime.utcnow(),
                        completed_at=datetime.utcnow(),
                    )

        self._results_cache[job_id] = results
        logger.info(
            f"ConcurrentBatchExecutor: processed {len(requests)} requests",
            extra={
                "event_type": "batch_submitted",
                "batch_id": job_id,
                "total": len(requests),
                "completed": sum(1 for r in results if r and r.is_success()),
                "failed": sum(1 for r in results if r and not r.is_success()),
            },
        )
        return job_id

    def poll(self, job_id: str) -> Tuple[str, List[BacktestLLMResult]]:
        results = self._results_cache.get(job_id, [])
        return "completed", results

    def materialize(self, result: BacktestLLMResult) -> Optional[Dict[str, Any]]:
        return result.parse_decision()


# ---------------------------------------------------------------------------
# OpenAIBatchExecutor: OpenAI Batch API
# ---------------------------------------------------------------------------


class OpenAIBatchExecutor(BacktestLLMExecutor):
    """
    OpenAI Batch API executor.

    Submits /v1/chat/completions requests via the Batch API for ~50% cost reduction.
    Completion window: up to 24h (typically completes in minutes).

    Reference: https://platform.openai.com/docs/guides/batch
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        completion_window: str = "24h",
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package required for OpenAIBatchExecutor") from exc

        self._client = OpenAI(api_key=api_key)  # type: ignore[arg-type]
        self.model = model
        self.completion_window = completion_window

    def submit(self, requests: List[BacktestLLMRequest]) -> str:
        """Build JSONL, upload, and create batch. Returns OpenAI batch_id."""
        lines = []
        for req in requests:
            # Use payload if pre-built, otherwise generate from kline_data
            messages = req.payload if req.payload else _build_single_prompt_messages(
                req.kline_data, req.trace.symbol, req.trace.timeframe
            )
            body = {
                "model": self.model,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "max_tokens": 512,
            }
            line = {
                "custom_id": req.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            lines.append(json.dumps(line))

        jsonl_bytes = "\n".join(lines).encode("utf-8")
        jsonl_file = io.BytesIO(jsonl_bytes)
        jsonl_file.name = "batch_input.jsonl"

        submitted_at = datetime.utcnow()
        upload = self._client.files.create(file=jsonl_file, purpose="batch")

        batch = self._client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window=self.completion_window,
            metadata={"submitted_at": submitted_at.isoformat()},
        )

        logger.info(
            "OpenAI batch submitted",
            extra={
                "event_type": "batch_submitted",
                "batch_id": batch.id,
                "total_requests": len(requests),
                "input_file_id": upload.id,
            },
        )
        return batch.id

    def poll(self, job_id: str) -> Tuple[str, List[BacktestLLMResult]]:
        batch = self._client.batches.retrieve(job_id)
        status = batch.status  # validating|in_progress|finalizing|completed|failed|expired|cancelled

        if status != "completed":
            return status, []

        # Download and parse output file
        results = self._download_results(batch)
        logger.info(
            "OpenAI batch completed",
            extra={
                "event_type": "batch_completed",
                "batch_id": job_id,
                "total": batch.request_counts.total if batch.request_counts else 0,
                "completed": batch.request_counts.completed if batch.request_counts else 0,
                "failed": batch.request_counts.failed if batch.request_counts else 0,
            },
        )
        return "completed", results

    def _download_results(self, batch) -> List[BacktestLLMResult]:
        results = []
        completed_at = datetime.utcnow()

        if not batch.output_file_id:
            return results

        raw = self._client.files.content(batch.output_file_id).text
        for line in raw.strip().split("\n"):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                custom_id = obj.get("custom_id", "")
                response = obj.get("response", {})
                error_obj = obj.get("error")

                if error_obj or response.get("status_code", 200) >= 400:
                    results.append(
                        BacktestLLMResult(
                            custom_id=custom_id,
                            status="failed",
                            output=None,
                            error=error_obj or {"message": "HTTP error", "status_code": response.get("status_code")},
                            completed_at=completed_at,
                        )
                    )
                    continue

                # Extract message content
                body = response.get("body", {})
                choices = body.get("choices", [])
                content = choices[0]["message"]["content"] if choices else "{}"

                results.append(
                    BacktestLLMResult(
                        custom_id=custom_id,
                        status="completed",
                        output=content,
                        error=None,
                        completed_at=completed_at,
                    )
                )
            except Exception as exc:
                logger.warning(f"Failed to parse output line: {exc}")

        return results

    def materialize(self, result: BacktestLLMResult) -> Optional[Dict[str, Any]]:
        return result.parse_decision()


# ---------------------------------------------------------------------------
# AnthropicBatchExecutor: Anthropic Message Batches API
# ---------------------------------------------------------------------------


class AnthropicBatchExecutor(BacktestLLMExecutor):
    """
    Anthropic Message Batches API executor.

    Submits batch of messages for ~50% cost reduction.
    Reference: https://docs.anthropic.com/en/docs/build-with-claude/message-batches
    """

    def __init__(self, api_key: str, model: str):
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError("anthropic package required for AnthropicBatchExecutor") from exc

        self._client = Anthropic(api_key=api_key)  # type: ignore[arg-type]
        self.model = model

    def submit(self, requests: List[BacktestLLMRequest]) -> str:
        """Build MessageBatchRequestParam list and create batch."""
        batch_requests = []
        for req in requests:
            messages = req.payload if req.payload else _build_single_prompt_messages(
                req.kline_data, req.trace.symbol, req.trace.timeframe
            )

            # Separate system message (Anthropic requires it as a top-level param)
            system_msg = None
            user_msgs = []
            for msg in messages:
                if msg.get("role") == "system":
                    system_msg = msg["content"]
                else:
                    user_msgs.append({"role": msg["role"], "content": msg["content"]})

            params: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": 512,
                "messages": user_msgs,
            }
            if system_msg:
                params["system"] = system_msg

            batch_requests.append({"custom_id": req.custom_id, "params": params})

        batch = self._client.messages.batches.create(requests=batch_requests)

        logger.info(
            "Anthropic batch submitted",
            extra={
                "event_type": "batch_submitted",
                "batch_id": batch.id,
                "total_requests": len(requests),
            },
        )
        return batch.id

    def poll(self, job_id: str) -> Tuple[str, List[BacktestLLMResult]]:
        batch = self._client.messages.batches.retrieve(job_id)
        status = batch.processing_status  # in_progress | ended

        if status != "ended":
            return "in_progress", []

        results = self._collect_results(job_id)
        logger.info(
            "Anthropic batch completed",
            extra={
                "event_type": "batch_completed",
                "batch_id": job_id,
                "succeeded": batch.request_counts.succeeded if batch.request_counts else 0,
                "errored": batch.request_counts.errored if batch.request_counts else 0,
            },
        )
        return "completed", results

    def _collect_results(self, job_id: str) -> List[BacktestLLMResult]:
        results = []
        completed_at = datetime.utcnow()
        for item in self._client.messages.batches.results(job_id):
            custom_id = item.custom_id
            result = item.result
            result_type = result.type  # succeeded | errored | canceled | expired

            if result_type == "succeeded":
                content_blocks = result.message.content if result.message else []
                text = content_blocks[0].text if content_blocks else "{}"
                results.append(
                    BacktestLLMResult(
                        custom_id=custom_id,
                        status="completed",
                        output=text,
                        error=None,
                        completed_at=completed_at,
                    )
                )
            else:
                error_info = {}
                if hasattr(result, "error") and result.error:
                    error_info = {
                        "type": result.error.type,
                        "message": str(result.error),
                    }
                results.append(
                    BacktestLLMResult(
                        custom_id=custom_id,
                        status=result_type if result_type != "canceled" else "cancelled",
                        output=None,
                        error=error_info or {"message": f"Result type: {result_type}"},
                        completed_at=completed_at,
                    )
                )
        return results

    def materialize(self, result: BacktestLLMResult) -> Optional[Dict[str, Any]]:
        return result.parse_decision()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class BatchExecutorFactory:
    """Creates the appropriate executor based on provider and config."""

    @staticmethod
    def create(
        provider: str,
        model: str,
        trading_graph=None,
        api_key: str = "",
        batch_config: Optional[BatchConfig] = None,
        concurrent_workers: int = 4,
    ) -> BacktestLLMExecutor:
        """
        Create executor for given provider.

        Args:
            provider: "sync" | "concurrent" | "openai_batch" | "anthropic_batch"
            model: model name (used for provider batch executors)
            trading_graph: TradingGraph instance (for sync/concurrent modes)
            api_key: provider API key (for batch modes)
            batch_config: BatchConfig instance
            concurrent_workers: max threads for ConcurrentBatchExecutor
        """
        cfg = batch_config or BatchConfig()

        if not cfg.batch_enabled or provider == "sync":
            if trading_graph is None:
                raise ValueError("trading_graph required for SyncExecutor")
            return SyncExecutor(trading_graph)

        if provider == "concurrent":
            if trading_graph is None:
                raise ValueError("trading_graph required for ConcurrentBatchExecutor")
            return ConcurrentBatchExecutor(trading_graph, max_workers=concurrent_workers)

        if provider == "openai_batch":
            return OpenAIBatchExecutor(
                api_key=api_key,
                model=model,
                completion_window=cfg.batch_completion_window,
            )

        if provider == "anthropic_batch":
            return AnthropicBatchExecutor(api_key=api_key, model=model)

        raise ValueError(
            f"Unknown batch executor provider '{provider}'. "
            "Use: sync | concurrent | openai_batch | anthropic_batch"
        )


# ---------------------------------------------------------------------------
# BatchSignalCollector: collects and dispatches signal requests
# ---------------------------------------------------------------------------


class BatchSignalCollector:
    """
    Accumulates BacktestLLMRequests and dispatches in batches.

    Used by the backtest engine to buffer requests and submit when
    batch_size or flush_timeout is reached.
    """

    def __init__(
        self,
        executor: BacktestLLMExecutor,
        config: BatchConfig,
    ):
        self.executor = executor
        self.config = config
        self._pending: List[BacktestLLMRequest] = []
        self._results: Dict[str, BacktestLLMResult] = {}
        self._last_flush: float = time.monotonic()

        # Counters
        self.invocations_total: int = 0
        self.batched_total: int = 0
        self.failed_total: int = 0

    def add(self, request: BacktestLLMRequest) -> None:
        """Add a request to the pending queue."""
        self._pending.append(request)
        self.invocations_total += 1

        should_flush = len(self._pending) >= self.config.batch_size or (
            time.monotonic() - self._last_flush >= self.config.batch_flush_timeout_sec
        )
        if should_flush:
            self.flush()

    def flush(self) -> None:
        """Submit all pending requests."""
        if not self._pending:
            return

        batch = list(self._pending)
        self._pending.clear()
        self._last_flush = time.monotonic()

        try:
            job_id = self.executor.submit(batch)
            results = self.executor.wait_for_completion(
                job_id,
                poll_interval_sec=self.config.batch_poll_interval_sec,
                fail_fast=self.config.fail_fast,
            )
            for r in results:
                self._results[r.custom_id] = r
                if r.is_success():
                    self.batched_total += 1
                else:
                    self.failed_total += 1
                    logger.warning(
                        f"Batch request failed: {r.custom_id} — {r.error}",
                        extra={
                            "event_type": "batch_request_error",
                            "custom_id": r.custom_id,
                            "error": r.error,
                        },
                    )

            logger.info(
                "Batch flush complete",
                extra={
                    "event_type": "batch_flush",
                    "job_id": job_id,
                    "submitted": len(batch),
                    "succeeded": self.batched_total,
                    "failed": self.failed_total,
                },
            )
        except Exception as exc:
            self.failed_total += len(batch)
            logger.error(
                f"Batch flush error: {exc}",
                exc_info=True,
                extra={"event_type": "batch_flush_error"},
            )
            if self.config.fail_fast:
                raise

    def get_result(self, custom_id: str) -> Optional[BacktestLLMResult]:
        """Retrieve cached result by custom_id."""
        return self._results.get(custom_id)

    def flush_and_get_all(self) -> Dict[str, BacktestLLMResult]:
        """Flush remaining pending requests and return all results."""
        self.flush()
        return dict(self._results)

    def summary(self) -> Dict[str, int]:
        return {
            "invocations_total": self.invocations_total,
            "batched_total": self.batched_total,
            "failed_total": self.failed_total,
        }
