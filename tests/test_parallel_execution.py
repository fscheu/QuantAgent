"""
Verify that the trading graph runs the three analysis agents in parallel.
Data comes from the versioned fixture tests/fixtures/spy-smoke.csv; LLMs are the
conftest mocks, so the test needs no network or API keys.
"""

import pandas as pd
import pytest

from quantagent.backtesting.fixtures import load_fixture
from quantagent.models import MarketData
from quantagent.static_util import read_and_format_ohlcv
from quantagent.trading_graph import TradingGraph

ANALYSIS_AGENTS = {"Indicator Agent", "Pattern Agent", "Trend Agent"}


@pytest.mark.integration
def test_parallel_execution(db_session):
    """
    Validates the graph topology on real fixture candles: Indicator, Pattern and
    Trend agents run in the same LangGraph superstep (fan-out from START), the
    Decision Maker runs in the next one (fan-in), and every agent leaves its report.
    Fails if the agents are chained sequentially or one is dropped from the graph.
    """
    row_count = load_fixture(db_session, "spy-smoke")
    candles = db_session.query(MarketData).order_by(MarketData.timestamp).all()
    assert len(candles) == row_count

    df = pd.DataFrame(
        {
            "Datetime": [c.timestamp for c in candles],
            "Open": [float(c.open) for c in candles],
            "High": [float(c.high) for c in candles],
            "Low": [float(c.low) for c in candles],
            "Close": [float(c.close) for c in candles],
        }
    )
    initial_state = {
        "kline_data": read_and_format_ohlcv(df),
        "time_frame": candles[0].timeframe,
        "stock_name": candles[0].symbol,
        "messages": [],
    }

    tg = TradingGraph()
    step_by_node = {}
    result = {}
    for event in tg.graph.stream(initial_state, stream_mode="debug"):
        if event["type"] != "task_result":
            continue
        payload = event["payload"]
        assert not payload.get("error"), f"{payload['name']} failed: {payload['error']}"
        step_by_node[payload["name"]] = event["step"]
        result.update(dict(payload["result"]))

    assert ANALYSIS_AGENTS <= set(step_by_node), f"Agents that ran: {step_by_node}"
    agent_steps = {step_by_node[name] for name in ANALYSIS_AGENTS}
    assert len(agent_steps) == 1, f"Agents ran in different supersteps: {step_by_node}"
    assert step_by_node["Decision Maker"] == agent_steps.pop() + 1

    for key in ("indicator_report", "pattern_report", "trend_report"):
        assert result.get(key), f"Missing {key}"
    assert result["final_trade_decision"]
