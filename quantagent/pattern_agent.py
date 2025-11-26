import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from quantagent.agent_models import PatternReport
from quantagent.agent_utils import invoke_with_retry


def create_pattern_agent(tool_llm, graph_llm, toolkit):
    """
    Create a pattern recognition agent node for candlestick pattern analysis.

    Returns structured PatternReport instead of string output.
    Uses centralized retry logic for both tool and LLM calls.
    """

    def pattern_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        # --- Tool definitions ---
        tools = [toolkit.generate_kline_image]
        time_frame = state["time_frame"]

        pattern_descriptions = {
            "inverse_head_and_shoulders": "Three lows with the middle one being the lowest, symmetrical structure, typically indicates upward trend",
            "double_bottom": "Two similar low points with a rebound in between, forming a 'W' shape",
            "rounded_bottom": "Gradual price decline followed by gradual rise, forming a 'U' shape",
            "hidden_base": "Horizontal consolidation followed by sudden upward breakout",
            "falling_wedge": "Price narrows downward, usually breaks out upward",
            "rising_wedge": "Price rises slowly but converges, often breaks down",
            "ascending_triangle": "Rising support line with flat resistance on top, breakout occurs upward",
            "descending_triangle": "Falling resistance line with flat support at bottom, typically breaks down",
            "bullish_flag": "After sharp rise, price consolidates downward briefly before continuing upward",
            "bearish_flag": "After sharp drop, price consolidates upward briefly before continuing downward",
            "rectangle": "Price fluctuates between horizontal support and resistance",
            "island_reversal": "Two price gaps in opposite directions forming isolated price island",
            "v_shaped_reversal": "Sharp decline followed by sharp recovery, or vice versa",
            "rounded_top": "Gradual peaking, forming an arc-shaped pattern",
            "expanding_triangle": "Highs and lows increasingly wider, indicating volatile swings",
            "symmetrical_triangle": "Highs and lows converge toward apex, usually followed by breakout",
        }

        # --- Check for precomputed image in state ---
        pattern_image_b64 = state.get("pattern_image")
        messages = state.get("messages", [])

        # --- Generate image if not precomputed ---
        if not pattern_image_b64:
            print("No precomputed pattern image found, generating with tool...")

            try:
                # Call tool with retry wrapper
                tool_result = invoke_with_retry(
                    toolkit.generate_kline_image.invoke,
                    {"kline_data": state["kline_data"]},
                    retries=3,
                    wait_sec=4
                )
                pattern_image_b64 = tool_result.get("pattern_image")
            except Exception as e:
                print(f"Failed to generate pattern image: {e}")
                pattern_image_b64 = None

        # --- Vision analysis with image ---
        pattern_detected = None
        confidence = 0.0
        breakout_probability = 0.0
        reasoning = "Pattern analysis could not be completed"

        if pattern_image_b64:
            image_prompt = [
                {
                    "type": "text",
                    "text": (
                        f"This is a {time_frame} candlestick chart from recent OHLC market data.\n\n"
                        "Known patterns to identify:\n"
                        + "\n".join([f"- {k}: {v}" for k, v in pattern_descriptions.items()])
                        + "\n\nDetermine which patterns (if any) match the chart. "
                        "Respond in JSON format:\n"
                        "{\n"
                        '  "patterns_detected": ["pattern1", "pattern2"],\n'
                        '  "primary_pattern": "pattern1",\n'
                        '  "confidence": 0.75,\n'
                        '  "breakout_probability": 0.65,\n'
                        '  "reasoning": "explanation"\n'
                        "}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{pattern_image_b64}"},
                },
            ]

            human_msg = HumanMessage(content=image_prompt)
            system_msg = SystemMessage(content="You are a pattern recognition assistant analyzing candlestick charts.")

            try:
                # Try with system message
                final_response = invoke_with_retry(
                    graph_llm.invoke,
                    [system_msg, human_msg],
                    retries=3,
                    wait_sec=8
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
                            wait_sec=8
                        )
                    except Exception as retry_error:
                        reasoning = f"Vision LLM failed: {str(retry_error)}"
                        final_response = None
                else:
                    reasoning = f"Vision LLM error: {str(e)}"
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
                    pattern_detected = response_dict.get("patterns_detected", [])
                    confidence = float(response_dict.get("confidence", 0.0))
                    breakout_probability = float(response_dict.get("breakout_probability", 0.0))
                    reasoning = response_dict.get("reasoning", "Pattern detected")
                except (json.JSONDecodeError, ValueError, KeyError) as parse_err:
                    reasoning = f"Pattern analysis completed with fallback parsing: {final_response.content[:100]}"

        # Build structured report
        try:
            pattern_report = PatternReport(
                patterns_detected=pattern_detected or [],
                primary_pattern=pattern_detected[0] if pattern_detected else "failed to analyze",
                confidence=min(1.0, max(0.0, confidence)),
                breakout_probability=min(1.0, max(0.0, breakout_probability)),
                reasoning=reasoning
            )
        except Exception as e:
            # Fallback valid report
            pattern_report = PatternReport(
                patterns_detected=[],
                primary_pattern="failed to analyze",
                confidence=0.0,
                breakout_probability=0.0,
                reasoning=f"Failed to create report: {str(e)}"
            )

        return {
            "messages": messages,
            "pattern_report": pattern_report,
        }

    return pattern_agent_node
