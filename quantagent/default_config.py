"""
Default configuration values for LLM models.
API keys and provider selection should be set via environment variables (.env file).
"""

# Default model configuration by provider
DEFAULT_MODELS = {
    "openai": {
        "agent": "gpt-4o-mini",
        "graph": "gpt-4o"
    },
    "anthropic": {
        "agent": "claude-haiku-4-5-20251001",
        "graph": "claude-haiku-4-5-20251001"
    },
    "qwen": {
        "agent": "qwen3-max",
        "graph": "qwen3-vl-plus"
    }
}

# Default temperature (professional, deterministic outputs)
DEFAULT_TEMPERATURE = 0.1
