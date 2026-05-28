"""
TradingGraph: Orchestrates the multi-agent trading system using LangChain and LangGraph.
Initializes LLMs, toolkits, and agent nodes for indicator, pattern, and trend analysis.
Supports PostgreSQL checkpointing for resilient backtest execution.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_qwq import ChatQwen

from quantagent import settings
from quantagent.graph_setup import SetGraph
from quantagent.graph_util import TechnicalTools
from quantagent.llm.roles import ProviderRoleConfig
from quantagent.llm.routing import (
    ROLE_DEEP_REASONING,
    ROLE_LITE,
    ProviderRoutingPolicy,
)

logger = logging.getLogger(__name__)

try:
    from langgraph.checkpoint.postgres import PostgresSaver
except ImportError:
    PostgresSaver = None


class TradingGraph:
    """
    Main orchestrator for the multi-agent trading system.
    Sets up LLMs, toolkits, and agent nodes for indicator, pattern, and trend analysis.
    """

    def __init__(
        self,
        use_checkpointing: bool = False,
        routing_policy: Optional[ProviderRoutingPolicy] = None,
    ):
        """
        Initialize TradingGraph with configuration from environment variables.

        Args:
            use_checkpointing: Enable PostgreSQL checkpointing for resilience
        """
        logger.info("Initializing TradingGraph", extra={"event_type": "graph_init"})

        self._routing_policy = routing_policy or ProviderRoutingPolicy.from_legacy_settings()
        agent_config = self._routing_policy.resolve(ROLE_LITE)
        graph_config = self._routing_policy.resolve(ROLE_DEEP_REASONING)
        self.agent_llm = self._create_llm_from_config(agent_config)
        self.graph_llm = self._create_llm_from_config(graph_config)

        logger.info(
            f"LLM configuration: agent={agent_config.provider}/{agent_config.model_name}, "
            f"graph={graph_config.provider}/{graph_config.model_name}",
            extra={
                "event_type": "llm_config",
                "extra_data": {
                    "agent": {
                        "provider": agent_config.provider,
                        "model": agent_config.model_name,
                        "temperature": agent_config.temperature,
                        "role": ROLE_LITE,
                    },
                    "graph": {
                        "provider": graph_config.provider,
                        "model": graph_config.model_name,
                        "temperature": graph_config.temperature,
                        "role": ROLE_DEEP_REASONING,
                    },
                },
            },
        )

        self.toolkit = TechnicalTools()

        # --- Setup PostgreSQL checkpointing if enabled ---
        self.checkpointer = None
        self._checkpointer_context = None
        if use_checkpointing:
            self.checkpointer, self._checkpointer_context = self._setup_checkpointer()
            logger.info(
                "Checkpointing enabled", extra={"event_type": "checkpointer_init"}
            )
        else:
            logger.info(
                "Checkpointing disabled", extra={"event_type": "checkpointer_init"}
            )

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

        logger.info(
            "TradingGraph initialization complete", extra={"event_type": "graph_ready"}
        )

    def _render_stategraph_png(self) -> bytes:
        drawable_graph = self.graph.get_graph() if hasattr(self.graph, "get_graph") else self.graph
        return drawable_graph.draw_mermaid_png()

    def export_stategraph_image(
        self,
        artifacts_policy: str = "path-only",
        *,
        artifacts_root: str | Path | None = None,
        environment: str | None = None,
        run_id: str | None = None,
        thread_id: str | None = None,
        symbol: str | None = None,
    ) -> str | None:
        if artifacts_policy == "none":
            return None

        base_dir = Path(artifacts_root or "data/artifacts").resolve()
        context_parts = [part for part in (environment, run_id or thread_id, symbol) if part]
        output_dir = base_dir.joinpath(*context_parts)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"stategraph-{datetime.utcnow().strftime('%Y%m%dT%H%M%S%fZ')}.png"
        output_path = output_dir / filename
        output_path.write_bytes(self._render_stategraph_png())
        return str(output_path)

    def build_stategraph_artifact_metadata(
        self,
        artifacts_policy: str = "path-only",
        **export_kwargs,
    ) -> dict[str, str]:
        image_path = self.export_stategraph_image(
            artifacts_policy=artifacts_policy,
            **export_kwargs,
        )
        if image_path is None:
            return {}
        return {"stategraph_image_path": image_path}

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
        elif provider == "azure":
            api_key = settings.AZURE_OPENAI_API_KEY
            if not api_key:
                raise ValueError(
                    "AZURE_OPENAI_API_KEY not found in .env file. "
                    "Please set it in your .env file or use the web interface."
                )
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                "Must be 'openai', 'anthropic', 'qwen', or 'azure'"
            )

        return api_key

    def _setup_checkpointer(self):
        """
        Setup PostgreSQL checkpointer for graph resilience.

        Returns:
            Tuple of (checkpointer, context_manager) to keep connection alive.

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
            # PostgresSaver.from_conn_string returns a context manager
            # Enter it manually and keep it alive for the TradingGraph lifetime
            context_manager = PostgresSaver.from_conn_string(db_url)
            checkpointer = context_manager.__enter__()
            checkpointer.setup()
            return checkpointer, context_manager
        except Exception as e:
            raise ValueError(f"Failed to connect to PostgreSQL at {db_url}: {str(e)}")

    def _create_llm_from_config(self, config: ProviderRoleConfig) -> BaseChatModel:
        return self._create_llm(
            provider=config.provider,
            model=config.model_name,
            temperature=config.temperature,
        )

    def _create_llm(
        self, provider: str, model: str, temperature: float
    ) -> BaseChatModel:
        """
        Create an LLM instance based on the provider.

        Args:
            provider: The provider name ("openai", "anthropic", "qwen", or "azure")
            model: The model name
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
        elif provider == "azure":
            from langchain_openai import AzureChatOpenAI

            endpoint = settings.AZURE_OPENAI_ENDPOINT
            deployment = settings.AZURE_OPENAI_DEPLOYMENT
            api_version = settings.AZURE_OPENAI_API_VERSION

            if not endpoint:
                raise ValueError(
                    "AZURE_OPENAI_ENDPOINT not found in .env file. "
                    "Example: https://myresource.openai.azure.com/"
                )
            if not deployment:
                raise ValueError(
                    "AZURE_OPENAI_DEPLOYMENT not found in .env file. "
                    "Set this to your Azure deployment name (e.g., gpt-4o)"
                )

            return AzureChatOpenAI(
                azure_endpoint=endpoint,
                azure_deployment=deployment,
                api_version=api_version,
                api_key=api_key,
                temperature=temperature,
            )
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                "Must be 'openai', 'anthropic', 'qwen', or 'azure'"
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
        self.agent_llm = self._create_llm_from_config(
            self._routing_policy.resolve(ROLE_LITE)
        )
        self.graph_llm = self._create_llm_from_config(
            self._routing_policy.resolve(ROLE_DEEP_REASONING)
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
            "qwen": "DASHSCOPE_API_KEY",
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

    def close(self):
        """
        Close the checkpointer connection if it was opened.
        Call this when done with the TradingGraph to clean up resources.
        """
        if hasattr(self, "_checkpointer_context") and self._checkpointer_context:
            try:
                self._checkpointer_context.__exit__(None, None, None)
            except Exception:
                pass  # Ignore errors during cleanup
