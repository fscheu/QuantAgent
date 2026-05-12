"""
Agent for trend analysis in high-frequency trading (HFT) context.
Uses LLM and toolkit to generate and interpret trendline charts for short-term prediction.
"""

import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from quantagent.agent_models import TrendReport
from quantagent.agent_utils import invoke_with_retry
from quantagent.llm_telemetry import TelemetryCtx

logger = logging.getLogger(__name__)


def create_trend_agent(tool_llm, graph_llm, toolkit):
    """
    Create a trend analysis agent node for HFT.

    Returns structured TrendReport instead of string output.
    Uses centralized retry logic for both tool and LLM calls.
    """

    def trend_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        # --- Tool definitions ---
        time_frame = state["time_frame"]
        symbol = state.get("stock_name", "UNKNOWN")
        thread_id = state.get("thread_id")

        logger.info(
            f"Starting trend agent for {symbol}",
            extra={
                "event_type": "agent_start",
                "symbol": symbol,
                "thread_id": thread_id,
            },
        )

        # --- Check for precomputed image in state ---
        trend_image_b64 = state.get("trend_image")

        # Initialize agent_messages list (will only populate if we do LLM analysis)
        agent_messages = []

        # --- Generate image if not precomputed ---
        if not trend_image_b64:
            logger.info(
                "No precomputed trend image found, generating with tool...",
                extra={"event_type": "trend_image_generation"},
            )

            try:
                # Call tool with retry wrapper
                tool_result = invoke_with_retry(
                    toolkit.generate_trend_image.invoke,
                    {"kline_data": state["kline_data"]},
                    retries=3,
                    base_wait=4,
                )
                trend_image_b64 = tool_result.get("trend_image")
            except Exception as e:
                logger.error(
                    f"Failed to generate trend image: {e}",
                    extra={"event_type": "trend_image_generation_failed"},
                    exc_info=True,
                )
                trend_image_b64 = None

        # --- Initialize trend analysis output ---
        reasoning = "Trend analysis could not be completed"
        trend_report = None

        # --- Vision analysis with image ---
        if trend_image_b64:
            image_prompt = [
                {
                    "type": "text",
                    "text": (
                        f"This is a {time_frame} candlestick chart with automated trendlines:\n"
                        "- **Blue line**: Support (derived from recent closing prices)\n"
                        "- **Red line**: Resistance (derived from recent closing prices)\n\n"
                        "Analyze how price interacts with these lines:\n"
                        "- Are candles bouncing off support/resistance?\n"
                        "- Is price breaking through the lines?\n"
                        "- Is price compressing between them?\n\n"
                        "Based on trendline slope, spacing, and recent K-line behavior, predict the likely short-term trend."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{trend_image_b64}"},
                },
            ]

            human_msg = HumanMessage(content=image_prompt)
            system_msg = SystemMessage(
                content="You are a K-line trend analysis assistant. Analyze candlestick charts with support/resistance trendlines."
            )

            # Create agent-specific messages (will be added to state via reducer)
            agent_messages = [system_msg, human_msg]

            trend_telemetry_ctx = TelemetryCtx(
                operation="trend_agent",
                symbol=symbol,
                thread_id=thread_id,
                environment=state.get("environment"),
                backtest_run_id=state.get("backtest_run_id"),
            )

            try:
                structured_graph_llm = graph_llm.with_structured_output(TrendReport)
                trend_report = invoke_with_retry(
                    structured_graph_llm.invoke, agent_messages, retries=3, base_wait=4,
                    telemetry_ctx=trend_telemetry_ctx,
                )
            except Exception as e:
                # Fallback: retry without system message for Anthropic compatibility
                if "at least one message" in str(e).lower():
                    logger.info(
                        "Retrying without system message for Anthropic compatibility...",
                        extra={"event_type": "llm_retry_no_system_msg"},
                    )
                    try:
                        structured_graph_llm = graph_llm.with_structured_output(
                            TrendReport
                        )
                        trend_report = invoke_with_retry(
                            structured_graph_llm.invoke, [human_msg], retries=3, base_wait=4,
                            telemetry_ctx=trend_telemetry_ctx,
                        )
                    except Exception as retry_error:
                        reasoning = f"LLM error: {str(retry_error)}"
                else:
                    reasoning = f"LLM error: {str(e)}"

        if not isinstance(trend_report, TrendReport):
            try:
                trend_report = TrendReport(
                    support_level=0.0,
                    resistance_level=0.0,
                    trend_direction="sideways",
                    trend_strength=0.0,
                    reasoning=reasoning,
                )
            except Exception as e:
                trend_report = TrendReport(
                    support_level=0.0,
                    resistance_level=0.0,
                    trend_direction="sideways",
                    trend_strength=0.0,
                    reasoning=f"Failed to create report: {str(e)}",
                )

        # Don't add messages to shared state - each agent only needs them for its LLM call
        # Agents work independently and communicate via structured reports, not messages

        logger.info(
            f"Trend agent completed for {symbol}",
            extra={
                "event_type": "agent_end",
                "symbol": symbol,
                "thread_id": thread_id,
                "extra_data": {
                    "trend": trend_report.trend_direction,
                    "trend_strength": trend_report.trend_strength,
                    "support_level": trend_report.support_level,
                    "resistance_level": trend_report.resistance_level,
                },
            },
        )

        return {
            "trend_report": trend_report,
            "trend_image": trend_image_b64,
            "trend_image_filename": "trend_graph.png",
            "trend_image_description": (
                "Trend-enhanced candlestick chart with support/resistance lines"
                if trend_image_b64
                else None
            ),
        }

    return trend_agent_node
