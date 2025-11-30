"""
TradingGraph: Orchestrates the multi-agent trading system using LangChain and LangGraph.
Initializes LLMs, toolkits, and agent nodes for indicator, pattern, and trend analysis.
Supports PostgreSQL checkpointing for resilient backtest execution.
"""

import os
from typing import Dict, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_qwq import ChatQwen
from langgraph.prebuilt import ToolNode

from quantagent import settings
from quantagent.default_config import DEFAULT_MODELS, DEFAULT_TEMPERATURE
from quantagent.graph_setup import SetGraph
from quantagent.graph_util import TechnicalTools

try:
    from langgraph.checkpoint.postgres import PostgresSaver
except ImportError:
    PostgresSaver = None


class TradingGraph:
    """
    Main orchestrator for the multi-agent trading system.
    Sets up LLMs, toolkits, and agent nodes for indicator, pattern, and trend analysis.
    """

    def __init__(self, use_checkpointing: bool = False):
        """
        Initialize TradingGraph with configuration from environment variables.

        Args:
            use_checkpointing: Enable PostgreSQL checkpointing for resilience
        """
        # All configuration now comes from settings module (loaded from .env)
        self.agent_llm = self._create_llm(
            provider=settings.AGENT_LLM_PROVIDER,
            model=settings.AGENT_LLM_MODEL,
            temperature=settings.AGENT_LLM_TEMPERATURE,
        )
        self.graph_llm = self._create_llm(
            provider=settings.GRAPH_LLM_PROVIDER,
            model=settings.GRAPH_LLM_MODEL,
            temperature=settings.GRAPH_LLM_TEMPERATURE,
        )
        self.toolkit = TechnicalTools()

        # --- Setup PostgreSQL checkpointing if enabled ---
        self.checkpointer = None
        if use_checkpointing:
            self.checkpointer = self._setup_checkpointer()

        # --- Create tool nodes for each agent ---
        # self.tool_nodes = self._set_tool_nodes()

        # --- Graph logic and setup ---
        self.graph_setup = SetGraph(
            self.agent_llm,
            self.graph_llm,
            self.toolkit,
            # self.tool_nodes,
        )

        # --- The main LangGraph graph object ---
        self.graph = self.graph_setup.set_graph(checkpointer=self.checkpointer)

    def _get_api_key(self, provider: str = "openai") -> str:
        """
        Get API key from settings module.

        Args:
            provider: The provider name ("openai", "anthropic", or "qwen")

        Returns:
            str: The API key for the specified provider

        Raises:
            ValueError: If API key is missing or invalid
        """
        if provider == "openai":
            api_key = settings.OPENAI_API_KEY
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found in .env file. "
                    "Please set it in your .env file or use the web interface."
                )
        elif provider == "anthropic":
            api_key = settings.ANTHROPIC_API_KEY
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not found in .env file. "
                    "Please set it in your .env file or use the web interface."
                )
        elif provider == "qwen":
            api_key = settings.DASHSCOPE_API_KEY
            if not api_key:
                raise ValueError(
                    "DASHSCOPE_API_KEY not found in .env file. "
                    "Please set it in your .env file or use the web interface."
                )
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Must be 'openai', 'anthropic', or 'qwen'"
            )

        return api_key

    def _setup_checkpointer(self) -> Optional[PostgresSaver]:
        """
        Setup PostgreSQL checkpointer for graph resilience.

        Returns:
            PostgresSaver instance or None if PostgreSQL not available.

        Raises:
            ValueError: If DATABASE_URL not set and checkpointing requested.
        """
        if not PostgresSaver:
            raise ImportError(
                "langgraph.checkpoint.postgres not available. "
                "Install with: pip install langgraph-checkpoint-postgres"
            )

        db_url = settings.DATABASE_URL
        if not db_url:
            raise ValueError(
                "DATABASE_URL not set in .env file. "
                "Set it for PostgreSQL checkpointing, or use use_checkpointing=False"
            )

        try:
            checkpointer = PostgresSaver.from_conn_string(db_url)
            # checkpointer.setup()
            return checkpointer
        except Exception as e:
            raise ValueError(
                f"Failed to connect to PostgreSQL at {db_url}: {str(e)}"
            )

    def _create_llm(
        self, provider: str, model: str, temperature: float
    ) -> BaseChatModel:
        """
        Create an LLM instance based on the provider.

        Args:
            provider: The provider name ("openai", "anthropic", or "qwen")
            model: The model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022", "qwen-vl-max-latest")
            temperature: The temperature setting for the model

        Returns:
            BaseChatModel: An instance of the appropriate LLM class
        """
        api_key = self._get_api_key(provider)

        if provider == "openai":
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key,
            )
        elif provider == "anthropic":
            # ChatAnthropic handles SystemMessage extraction automatically
            # It extracts SystemMessage from the message list and passes it as 'system' parameter
            # The messages array should contain at least one non-SystemMessage
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                api_key=api_key,
            )
        elif provider == "qwen":
            return ChatQwen(
                model=model,
                temperature=temperature,
                api_key=api_key,
                max_retries=4,
            )
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Must be 'openai', 'anthropic', or 'qwen'"
            )

    # def _set_tool_nodes(self) -> Dict[str, ToolNode]:
    #     """
    #     Define tool nodes for each agent type (indicator, pattern, trend).
    #     """
    #     return {
    #         "indicator": ToolNode(
    #             [
    #                 self.toolkit.compute_macd,
    #                 self.toolkit.compute_roc,
    #                 self.toolkit.compute_rsi,
    #                 self.toolkit.compute_stoch,
    #                 self.toolkit.compute_willr,
    #             ]
    #         ),
    #         "pattern": ToolNode(
    #             [
    #                 self.toolkit.generate_kline_image,
    #             ]
    #         ),
    #         "trend": ToolNode([self.toolkit.generate_trend_image]),
    #     }

    def refresh_llms(self):
        """
        Refresh the LLM objects with the current API key from environment.
        This is called when the API key is updated.
        """
        # Recreate LLM objects with current config values from settings module
        self.agent_llm = self._create_llm(
            provider=settings.AGENT_LLM_PROVIDER,
            model=settings.AGENT_LLM_MODEL,
            temperature=settings.AGENT_LLM_TEMPERATURE,
        )
        self.graph_llm = self._create_llm(
            provider=settings.GRAPH_LLM_PROVIDER,
            model=settings.GRAPH_LLM_MODEL,
            temperature=settings.GRAPH_LLM_TEMPERATURE,
        )

        # Recreate the graph setup with new LLMs
        self.graph_setup = SetGraph(
            self.agent_llm,
            self.graph_llm,
            self.toolkit,
            # self.tool_nodes,
        )

        # Recreate the main graph
        self.graph = self.graph_setup.set_graph()

    def update_api_key(self, api_key: str, provider: str = "openai"):
        """
        Update API key in runtime and persist to .env file.
        This method is called by the web interface when API key is updated.

        Args:
            api_key: The new API key
            provider: The provider name ("openai", "anthropic", or "qwen")
        """
        # Map provider to env var name
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "qwen": "DASHSCOPE_API_KEY"
        }

        if provider not in env_var_map:
            raise ValueError(
                f"Unsupported provider: {provider}. Must be 'openai', 'anthropic', or 'qwen'"
            )

        env_var = env_var_map[provider]

        # Persist to .env file and update runtime
        settings.update_env_file(env_var, api_key)

        # Refresh the settings module attributes
        if provider == "openai":
            settings.OPENAI_API_KEY = api_key
        elif provider == "anthropic":
            settings.ANTHROPIC_API_KEY = api_key
        elif provider == "qwen":
            settings.DASHSCOPE_API_KEY = api_key

        # Refresh LLMs with new key
        self.refresh_llms()
