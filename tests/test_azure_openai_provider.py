"""
Tests for Azure OpenAI provider support in TradingGraph.

Testing Strategy (QuantAgent-7bn):
- Test configuration validation and error handling
- Test Azure provider instantiation with required parameters  
- Test API version defaults
- Verify no regression in existing providers (openai, anthropic, qwen)
- Avoid tautological mocks; validate structure and constraints

See docs/03_design/TESTING_PATTERNS.md for guidelines.
See docs/05_acceptance_tests/QuantAgent-7bn-AC-azure-openai-support.md for AC.
"""

import os
import pytest
from unittest.mock import Mock, patch

from quantagent import settings
from quantagent.trading_graph import TradingGraph

pytestmark = pytest.mark.api

# ============================================================================
# AZURE CONFIGURATION TESTS  
# ============================================================================


class TestAzureConfiguration:
    """Test Azure OpenAI provider configuration and validation."""

    @patch.dict(
        os.environ,
        {
            "AGENT_LLM_PROVIDER": "azure",
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
            "AZURE_OPENAI_API_VERSION": "2024-02-01",
        },
        clear=False,
    )
    def test_azure_get_api_key_success(self, mock_llm, mock_vision_llm, mock_toolkit):
        """AC1: Verify _get_api_key returns Azure API key when provider is azure."""
        import importlib

        importlib.reload(settings)

        tg = TradingGraph(use_checkpointing=False)
        api_key = tg._get_api_key("azure")

        assert api_key == "test-key"
        assert api_key != ""

    @patch.dict(
        os.environ,
        {
            "AGENT_LLM_PROVIDER": "openai",  # Change to avoid Azure instantiation
            "AZURE_OPENAI_API_KEY": "",
        },
        clear=False,
    )
    def test_azure_missing_api_key_raises_error(
        self, mock_llm, mock_vision_llm, mock_toolkit
    ):
        """AC4: Verify ValueError is raised when AZURE_OPENAI_API_KEY is missing."""
        import importlib

        importlib.reload(settings)

        tg = TradingGraph(use_checkpointing=False)

        with pytest.raises(ValueError) as exc_info:
            tg._get_api_key("azure")

        assert "AZURE_OPENAI_API_KEY" in str(exc_info.value)

    @patch.dict(
        os.environ,
        {
            "AGENT_LLM_PROVIDER": "openai",  # Avoid Azure init
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "",
            "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
        },
        clear=False,
    )
    @patch("langchain_openai.AzureChatOpenAI")
    def test_azure_missing_endpoint_raises_error(
        self, mock_azure_llm, mock_llm, mock_vision_llm, mock_toolkit
    ):
        """AC3: Verify ValueError is raised when AZURE_OPENAI_ENDPOINT is missing."""
        import importlib

        importlib.reload(settings)

        tg = TradingGraph(use_checkpointing=False)

        with pytest.raises(ValueError) as exc_info:
            tg._create_llm("azure", "", 0.1)

        assert "AZURE_OPENAI_ENDPOINT" in str(exc_info.value)
        assert "https://" in str(exc_info.value).lower()

    @patch.dict(
        os.environ,
        {
            "AGENT_LLM_PROVIDER": "openai",  # Avoid Azure init
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_DEPLOYMENT": "",
        },
        clear=False,
    )
    @patch("langchain_openai.AzureChatOpenAI")
    def test_azure_missing_deployment_raises_error(
        self, mock_azure_llm, mock_llm, mock_vision_llm, mock_toolkit
    ):
        """AC5: Verify ValueError is raised when AZURE_OPENAI_DEPLOYMENT is missing."""
        import importlib

        importlib.reload(settings)

        tg = TradingGraph(use_checkpointing=False)

        with pytest.raises(ValueError) as exc_info:
            tg._create_llm("azure", "", 0.1)

        assert "AZURE_OPENAI_DEPLOYMENT" in str(exc_info.value)

    @patch.dict(
        os.environ,
        {
            "AGENT_LLM_PROVIDER": "openai",  # Avoid Azure init
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
            # Intentionally omit API_VERSION to test default
        },
        clear=False,
    )
    @patch("langchain_openai.AzureChatOpenAI")
    def test_azure_api_version_default(
        self, mock_azure_llm, mock_llm, mock_vision_llm, mock_toolkit
    ):
        """AC2: Verify API version defaults to '2024-02-01' when not specified."""
        import importlib

        importlib.reload(settings)

        tg = TradingGraph(use_checkpointing=False)
        tg._create_llm("azure", "", 0.1)

        # Verify AzureChatOpenAI was called with default api_version
        assert mock_azure_llm.called
        call_kwargs = mock_azure_llm.call_args.kwargs

        assert "api_version" in call_kwargs
        assert call_kwargs["api_version"] == "2024-02-01"


# ============================================================================
# AZURE LLM INSTANTIATION TESTS
# ============================================================================


class TestAzureLLMInstantiation:
    """Test Azure LLM instance creation with correct parameters."""

    @patch.dict(
        os.environ,
        {
            "AGENT_LLM_PROVIDER": "openai",  # Avoid init
            "AZURE_OPENAI_API_KEY": "test-key-123",
            "AZURE_OPENAI_ENDPOINT": "https://myresource.openai.azure.com/",
            "AZURE_OPENAI_DEPLOYMENT": "gpt-4o-deployment",
            "AZURE_OPENAI_API_VERSION": "2024-02-01",
        },
        clear=False,
    )
    @patch("langchain_openai.AzureChatOpenAI")
    def test_azure_llm_instantiation_with_correct_params(
        self, mock_azure_llm, mock_llm, mock_vision_llm, mock_toolkit
    ):
        """AC1: Verify AzureChatOpenAI is instantiated with correct parameters."""
        import importlib

        importlib.reload(settings)

        tg = TradingGraph(use_checkpointing=False)
        tg._create_llm("azure", "", 0.1)

        # Verify AzureChatOpenAI was called
        assert mock_azure_llm.called

        # Verify parameters passed
        call_kwargs = mock_azure_llm.call_args.kwargs
        assert call_kwargs["azure_endpoint"] == "https://myresource.openai.azure.com/"
        assert call_kwargs["azure_deployment"] == "gpt-4o-deployment"
        assert call_kwargs["api_version"] == "2024-02-01"
        assert call_kwargs["api_key"] == "test-key-123"
        assert call_kwargs["temperature"] == 0.1


# ============================================================================
# REGRESSION TESTS - EXISTING PROVIDERS
# ============================================================================


class TestExistingProvidersRegression:
    """Verify no breaking changes to existing providers (OpenAI, Anthropic, Qwen)."""

    @patch.dict(
        os.environ,
        {
            "AGENT_LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-openai-key",
        },
        clear=False,
    )
    def test_openai_provider_unchanged(
        self, mock_llm, mock_vision_llm, mock_toolkit
    ):
        """REG1: Verify OpenAI provider behavior is unchanged."""
        import importlib

        importlib.reload(settings)

        tg = TradingGraph(use_checkpointing=False)
        api_key = tg._get_api_key("openai")

        assert api_key == "test-openai-key"
        # Verify TradingGraph instantiates successfully
        assert tg.agent_llm is not None

    @patch.dict(
        os.environ,
        {
            "AGENT_LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
        },
        clear=False,
    )
    def test_anthropic_provider_unchanged(
        self, mock_llm, mock_vision_llm, mock_toolkit
    ):
        """REG2: Verify Anthropic provider behavior is unchanged."""
        import importlib

        importlib.reload(settings)

        tg = TradingGraph(use_checkpointing=False)
        api_key = tg._get_api_key("anthropic")

        assert api_key == "test-anthropic-key"
        assert tg.agent_llm is not None

    @patch.dict(
        os.environ,
        {
            "AGENT_LLM_PROVIDER": "qwen",
            "DASHSCOPE_API_KEY": "test-qwen-key",
        },
        clear=False,
    )
    def test_qwen_provider_unchanged(
        self, mock_llm, mock_vision_llm, mock_toolkit
    ):
        """REG3: Verify Qwen provider behavior is unchanged."""
        import importlib

        importlib.reload(settings)

        tg = TradingGraph(use_checkpointing=False)
        api_key = tg._get_api_key("qwen")

        assert api_key == "test-qwen-key"
        assert tg.agent_llm is not None


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestAzureErrorHandling:
    """Test error messages are clear and helpful."""

    def test_unsupported_provider_error_message(
        self, mock_llm, mock_vision_llm, mock_toolkit
    ):
        """Verify error message for unsupported provider includes 'azure'."""
        tg = TradingGraph(use_checkpointing=False)

        with pytest.raises(ValueError) as exc_info:
            tg._get_api_key("invalid_provider")

        error_msg = str(exc_info.value).lower()
        assert "azure" in error_msg
        assert "openai" in error_msg
        assert "anthropic" in error_msg
        assert "qwen" in error_msg


# ============================================================================
# SETTINGS MODULE TESTS
# ============================================================================


class TestAzureSettings:
    """Test Azure configuration variables in settings module."""

    def test_azure_settings_variables_exist(self):
        """Verify Azure settings variables are defined in settings module."""
        assert hasattr(settings, "AZURE_OPENAI_API_KEY")
        assert hasattr(settings, "AZURE_OPENAI_ENDPOINT")
        assert hasattr(settings, "AZURE_OPENAI_DEPLOYMENT")
        assert hasattr(settings, "AZURE_OPENAI_API_VERSION")

    def test_azure_api_version_has_default(self):
        """Verify AZURE_OPENAI_API_VERSION has default value."""
        # Even if not set in env, should have default
        assert settings.AZURE_OPENAI_API_VERSION is not None
        assert len(settings.AZURE_OPENAI_API_VERSION) > 0

    def test_get_default_model_supports_azure(self):
        """Verify get_default_model function supports azure provider."""
        from quantagent.settings import get_default_model

        agent_model = get_default_model("azure", is_agent=True)
        graph_model = get_default_model("azure", is_agent=False)

        # Azure returns empty strings (deployment name set by user)
        assert agent_model == ""
        assert graph_model == ""
