from typing import Dict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from quantagent.agent_state import IndicatorAgentState
from quantagent.decision_agent import create_final_trade_decider
from quantagent.graph_util import TechnicalTools
from quantagent.indicator_agent import create_indicator_agent
from quantagent.pattern_agent import create_pattern_agent
from quantagent.trend_agent import create_trend_agent


class SetGraph:
    def __init__(
        self,
        agent_llm: ChatOpenAI,
        graph_llm: ChatOpenAI,
        toolkit: TechnicalTools,
        # tool_nodes: Dict[str, ToolNode],
    ):
        self.agent_llm = agent_llm
        self.graph_llm = graph_llm
        self.toolkit = toolkit
        # self.tool_nodes = tool_nodes

    def set_graph(self, checkpointer=None):
        # Create analyst nodes
        agent_nodes = {}

        # create nodes for indicator agent
        agent_nodes["indicator"] = create_indicator_agent(self.graph_llm, self.toolkit)
        # tool_nodes["indicator"] = self.tool_nodes["indicator"]

        # create nodes for pattern agent
        agent_nodes["pattern"] = create_pattern_agent(
            self.agent_llm, self.graph_llm, self.toolkit
        )
        # tool_nodes["pattern"] = self.tool_nodes["pattern"]

        # create nodes for trend agent
        agent_nodes["trend"] = create_trend_agent(
            self.agent_llm, self.graph_llm, self.toolkit
        )
        # tool_nodes["trend"] = self.tool_nodes["trend"]

        # create nodes for decision agent
        decision_agent_node = create_final_trade_decider(self.graph_llm)

        # create graph
        graph = StateGraph(IndicatorAgentState)

        # add agent nodes to graph
        for agent_type, cur_node in agent_nodes.items():
            graph.add_node(f"{agent_type.capitalize()} Agent", cur_node)

        # add rest of the nodes
        graph.add_node("Decision Maker", decision_agent_node)

        # Parallelization: Fan-out from START to all three agents
        # All agents (Indicator, Pattern, Trend) are independent and analyze raw kline_data
        graph.add_edge(START, "Indicator Agent")
        graph.add_edge(START, "Pattern Agent")
        graph.add_edge(START, "Trend Agent")

        # Convergence: Fan-in from all three agents to Decision Maker
        # Decision agent receives reports from all three analysis agents
        graph.add_edge("Indicator Agent", "Decision Maker")
        graph.add_edge("Pattern Agent", "Decision Maker")
        graph.add_edge("Trend Agent", "Decision Maker")

        # Final decision output
        graph.add_edge("Decision Maker", END)

        return graph.compile(checkpointer=checkpointer) if checkpointer else graph.compile()
