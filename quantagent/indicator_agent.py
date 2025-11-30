"""
Agent for technical indicator analysis in high-frequency trading (HFT) context.
Uses LLM and toolkit to compute and interpret indicators like MACD, RSI, ROC, Stochastic, and Williams %R.
"""

import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from quantagent.agent_models import IndicatorReport
from quantagent.agent_utils import invoke_with_retry


def create_indicator_agent(llm, toolkit):
    """
    Create an indicator analysis node for HFT graph.

    This is a node (not a standalone agent) that:
    1. Calls technical indicator tools to compute values
    2. Uses LLM with structured output to analyze results
    3. Returns typed IndicatorReport

    Note: This is a node in the graph because it's part of a multi-agent pipeline.
    The graph itself (trading_graph.py) orchestrates the full flow.
    """

    # --- Tool definitions ---
    tools = [
        toolkit.compute_macd,
        toolkit.compute_rsi,
        toolkit.compute_roc,
        toolkit.compute_stoch,
        toolkit.compute_willr,
    ]

    def indicator_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        time_frame = state["time_frame"]
        kline_data = state["kline_data"]

        # --- System prompt for LLM with tool instructions ---
        system_prompt = (
            "You are a high-frequency trading (HFT) analyst assistant. "
            "You must analyze technical indicators to support fast-paced trading execution.\n\n"
            "You have access to tools: compute_rsi, compute_macd, compute_roc, compute_stoch, and compute_willr. "
            "Use them by providing appropriate arguments like `kline_data` and the respective periods.\n\n"
            f"⚠️ The OHLC data provided is from {time_frame} intervals, reflecting recent market behavior. "
            "You must interpret this data quickly and accurately.\n\n"
            "Call necessary tools to compute indicator values, then analyze the results.\n"
        )

        user_message = f"Analyze these indicators from the OHLC data:\n{json.dumps(kline_data, indent=2)}"

        # Create messages for this agent's analysis
        # Don't modify existing messages, just create new ones for this specific call
        agent_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        # Bind tools to LLM and use structured output
        llm_with_tools = llm.bind_tools(tools)
        structured_llm = llm_with_tools.with_structured_output(IndicatorReport)

        try:
            # LLM call with tools and structured output
            indicator_report = invoke_with_retry(
                structured_llm.invoke,
                agent_messages,
                retries=3,
                wait_sec=2
            )

            # Ensure we got a valid IndicatorReport
            if not isinstance(indicator_report, IndicatorReport):
                indicator_report = IndicatorReport(
                    macd=0.0,
                    macd_signal=0.0,
                    macd_histogram=0.0,
                    rsi=50.0,
                    rsi_level="neutral",
                    roc=0.0,
                    stochastic=50.0,
                    willr=-50.0,
                    trend_direction="neutral",
                    confidence=0.0,
                    reasoning="Output validation failed"
                )
        except Exception as e:
            # Fallback to minimal valid report
            indicator_report = IndicatorReport(
                macd=0.0,
                macd_signal=0.0,
                macd_histogram=0.0,
                rsi=50.0,
                rsi_level="neutral",
                roc=0.0,
                stochastic=50.0,
                willr=-50.0,
                trend_direction="neutral",
                confidence=0.0,
            reasoning=f"Analysis failed: {str(e)}"
            )

        # Don't add messages to shared state - each agent only needs them for its LLM call
        # Agents work independently and communicate via structured reports, not messages
        return {
            "indicator_report": indicator_report,
        }

    return indicator_agent_node
