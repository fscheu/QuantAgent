"""
Agent for trend analysis in high-frequency trading (HFT) context.
Uses LLM and toolkit to generate and interpret trendline charts for short-term prediction.
"""

import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from quantagent.agent_models import TrendReport
from quantagent.agent_utils import invoke_with_retry


def create_trend_agent(tool_llm, graph_llm, toolkit):
    """
    Create a trend analysis agent node for HFT.

    Returns structured TrendReport instead of string output.
    Uses centralized retry logic for both tool and LLM calls.
    """

    def trend_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        # --- Tool definitions ---
        tools = [toolkit.generate_trend_image]
        time_frame = state["time_frame"]

        # --- Check for precomputed image in state ---
        trend_image_b64 = state.get("trend_image")
        messages = state.get("messages", [])

        # --- Generate image if not precomputed ---
        if not trend_image_b64:
            print("No precomputed trend image found, generating with tool...")

            try:
                # Call tool with retry wrapper
                tool_result = invoke_with_retry(
                    toolkit.generate_trend_image.invoke,
                    {"kline_data": state["kline_data"]},
                    retries=3,
                    wait_sec=4
                )
                trend_image_b64 = tool_result.get("trend_image")
            except Exception as e:
                print(f"Failed to generate trend image: {e}")
                trend_image_b64 = None

        # --- Initialize trend analysis output ---
        support_level = 0.0
        resistance_level = 0.0
        trend_direction = "sideways"
        trend_strength = 0.0
        reasoning = "Trend analysis could not be completed"

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
                        "Based on trendline slope, spacing, and recent K-line behavior, predict the likely short-term trend.\n"
                        "Respond in JSON format:\n"
                        "{\n"
                        '  "support_level": <float>,\n'
                        '  "resistance_level": <float>,\n'
                        '  "trend_direction": "<upward|downward|sideways>",\n'
                        '  "trend_strength": <float 0.0-1.0>,\n'
                        '  "reasoning": "<brief explanation>"\n'
                        "}"
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
            messages = [system_msg, human_msg]

            try:
                # Try with system message
                final_response = invoke_with_retry(
                    graph_llm.invoke,
                    messages,
                    retries=3,
                    wait_sec=4
                )
            except Exception as e:
                # Fallback: retry without system message for Anthropic compatibility
                if "at least one message" in str(e).lower():
                    print("Retrying without system message for Anthropic compatibility...")
                    try:
                        final_response = invoke_with_retry(
                            graph_llm.invoke,
                            [human_msg],
                            retries=3,
                            wait_sec=4
                        )
                    except Exception as retry_error:
                        reasoning = f"LLM error: {str(retry_error)}"
                        final_response = None
                else:
                    reasoning = f"LLM error: {str(e)}"
                    final_response = None

            # Parse structured output
            if final_response and hasattr(final_response, "content"):
                try:
                    # Extract JSON from response (handle markdown)
                    response_text = final_response.content
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0]
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0]

                    response_dict = json.loads(response_text.strip())
                    support_level = float(response_dict.get("support_level", 0.0))
                    resistance_level = float(response_dict.get("resistance_level", 0.0))
                    trend_direction = response_dict.get("trend_direction", "sideways")
                    trend_strength = float(response_dict.get("trend_strength", 0.0))
                    reasoning = response_dict.get("reasoning", "Trend analysis completed")
                except (json.JSONDecodeError, ValueError, KeyError) as parse_err:
                    reasoning = f"Trend analysis completed with fallback parsing: {final_response.content[:100]}"

        # Build structured report
        try:
            trend_report = TrendReport(
                support_level=support_level,
                resistance_level=resistance_level,
                trend_direction=trend_direction,
                trend_strength=min(1.0, max(0.0, trend_strength)),
                reasoning=reasoning
            )
        except Exception as e:
            # Fallback valid report
            trend_report = TrendReport(
                support_level=0.0,
                resistance_level=0.0,
                trend_direction="sideways",
                trend_strength=0.0,
                reasoning=f"Failed to create report: {str(e)}"
            )

        return {
            "messages": messages,
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
